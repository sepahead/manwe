use std::io::{Read, Write};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::PathBuf;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Condvar, Mutex,
};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use bevy::app::AppExit;
use bevy::asset::RenderAssetUsages;
use bevy::prelude::*;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};
use bevy::window::PrimaryWindow;
use clap::Parser;
use image::{DynamicImage, ImageBuffer, Rgb};
use manwe::native_runtime::NativeRuntime;
use manwe::runtime_contract::VerifiedModelPackage;
use manwe::secure_io::{resolve_executable, ResolvedExecutable};
use manwe::stream_url::{validate_rtsp_url, INVALID_RTSP_URL, MAX_STREAMS};

const MAX_FRAME_PIXELS: usize = 16_777_216;
const MAX_VIEWER_WORK_BYTES: u64 = 1024 * 1024 * 1024;
const VIEW_CELL_FILL: f32 = 0.96;
const INITIAL_RETRY_DELAY: Duration = Duration::from_secs(1);
const MAX_RETRY_DELAY: Duration = Duration::from_secs(30);

fn positive_dimension(value: &str) -> std::result::Result<usize, String> {
    let parsed = value
        .parse::<usize>()
        .map_err(|_| format!("{value:?} is not a positive integer"))?;
    if (1..=8192).contains(&parsed) {
        Ok(parsed)
    } else {
        Err("value must be between 1 and 8192".to_string())
    }
}

#[derive(Parser, Resource, Clone)]
#[command(
    author,
    version,
    about = "Experimental macOS multi-stream camera viewer",
    long_about = None
)]
struct Args {
    /// RTSP/video URLs. Prefer MANWE_RTSP_URLS for URLs containing credentials.
    #[arg(
        long = "url",
        env = "MANWE_RTSP_URLS",
        hide_env_values = true,
        value_delimiter = '\x1f',
        required = true,
        num_args = 1..
    )]
    urls: Vec<String>,

    /// ffmpeg executable or absolute path.
    #[arg(
        long,
        env = "MANWE_FFMPEG",
        default_value = "ffmpeg",
        hide_env_values = true
    )]
    ffmpeg: PathBuf,

    /// Schema-2 contract beside the exact model artifact.
    #[arg(long, env = "MANWE_CONTRACT", hide_env_values = true)]
    contract: PathBuf,

    /// Force CPU inference.
    #[arg(long)]
    cpu: bool,

    /// Decoded frame width.
    #[arg(long, default_value_t = 1280, value_parser = positive_dimension)]
    width: usize,

    /// Decoded frame height.
    #[arg(long, default_value_t = 720, value_parser = positive_dimension)]
    height: usize,

    /// Run inference on every Nth frame.
    #[arg(long, default_value_t = 3, value_parser = positive_dimension)]
    inference_every: usize,
}

#[derive(Resource, Clone)]
struct FrameBuffer {
    frames: Arc<Mutex<Vec<DisplaySlot>>>,
}

#[derive(Default)]
struct DisplaySlot {
    latest_capture_sequence: u64,
    latest_annotation_sequence: u64,
    annotation_ready: bool,
    pending: Option<Arc<DynamicImage>>,
}

struct ScheduledFrame {
    stream_index: usize,
    sequence: u64,
    image: Arc<DynamicImage>,
}

struct InferenceQueueState {
    slots: Vec<Option<ScheduledFrame>>,
    next_stream: usize,
}

struct InferenceQueueInner {
    state: Mutex<InferenceQueueState>,
    ready: Condvar,
}

#[derive(Resource, Clone)]
struct InferenceQueue(Arc<InferenceQueueInner>);

impl InferenceQueue {
    fn new(stream_count: usize) -> Self {
        Self(Arc::new(InferenceQueueInner {
            state: Mutex::new(InferenceQueueState {
                slots: (0..stream_count).map(|_| None).collect(),
                next_stream: 0,
            }),
            ready: Condvar::new(),
        }))
    }

    fn submit(&self, frame: ScheduledFrame) -> Result<()> {
        let mut state = self
            .0
            .state
            .lock()
            .map_err(|_| anyhow::anyhow!("inference queue was poisoned"))?;
        let stream_index = frame.stream_index;
        let slot = state
            .slots
            .get_mut(stream_index)
            .context("inference frame has an invalid stream index")?;
        // One replaceable slot per stream is the queue bound: capture never
        // waits for inference, and the scheduler never executes a stale backlog.
        if slot
            .as_ref()
            .is_some_and(|pending| pending.sequence >= frame.sequence)
        {
            anyhow::bail!("inference frames must advance monotonically per stream")
        }
        *slot = Some(frame);
        drop(state);
        self.0.ready.notify_one();
        Ok(())
    }

    fn take_next(&self, running: &AtomicBool) -> Result<Option<ScheduledFrame>> {
        let mut state = self
            .0
            .state
            .lock()
            .map_err(|_| anyhow::anyhow!("inference queue was poisoned"))?;
        loop {
            if !running.load(Ordering::Acquire) {
                return Ok(None);
            }
            let stream_count = state.slots.len();
            for offset in 0..stream_count {
                let index = (state.next_stream + offset) % stream_count;
                if let Some(frame) = state.slots[index].take() {
                    state.next_stream = (index + 1) % stream_count;
                    return Ok(Some(frame));
                }
            }
            state = self
                .0
                .ready
                .wait(state)
                .map_err(|_| anyhow::anyhow!("inference queue was poisoned"))?;
        }
    }

    fn wake_all(&self) {
        self.0.ready.notify_all();
    }
}

#[derive(Resource, Clone)]
struct FfmpegExecutable(ResolvedExecutable);

#[derive(Resource, Clone)]
struct WorkerControl {
    running: Arc<AtomicBool>,
    handles: Arc<Mutex<Vec<JoinHandle<()>>>>,
    failure: Arc<Mutex<Option<String>>>,
}

struct StreamConfig {
    index: usize,
    url: String,
    ffmpeg: ResolvedExecutable,
    width: usize,
    height: usize,
    inference_every: usize,
}

enum StreamAttemptFailure {
    /// The child process or remote stream failed and a bounded reconnect is useful.
    Retry(anyhow::Error),
    /// Manwe's own state or an authenticated local dependency violated an invariant.
    Fatal(anyhow::Error),
}

struct ChildGuard {
    child: Arc<Mutex<Option<Child>>>,
}

#[derive(Clone)]
struct ChildTerminator {
    child: Arc<Mutex<Option<Child>>>,
}

impl ChildTerminator {
    fn terminate(&self) {
        let child = match self.child.lock() {
            Ok(mut child) => child.take(),
            Err(poisoned) => poisoned.into_inner().take(),
        };
        if let Some(mut child) = child {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

impl ChildGuard {
    fn new(child: Child) -> Self {
        Self {
            child: Arc::new(Mutex::new(Some(child))),
        }
    }

    fn terminator(&self) -> ChildTerminator {
        ChildTerminator {
            child: Arc::clone(&self.child),
        }
    }

    fn take_stdin(&self) -> Result<ChildStdin> {
        let mut child = self
            .child
            .lock()
            .map_err(|_| anyhow::anyhow!("ffmpeg process lock was poisoned"))?;
        child
            .as_mut()
            .context("ffmpeg process is no longer available")?
            .stdin
            .take()
            .context("ffmpeg did not expose stdin")
    }

    fn take_stdout(&self) -> Result<ChildStdout> {
        let mut child = self
            .child
            .lock()
            .map_err(|_| anyhow::anyhow!("ffmpeg process lock was poisoned"))?;
        child
            .as_mut()
            .context("ffmpeg process is no longer available")?
            .stdout
            .take()
            .context("ffmpeg did not expose stdout")
    }

    fn wait(&mut self, timeout: Duration) -> Result<std::process::ExitStatus> {
        let deadline = Instant::now() + timeout;
        loop {
            {
                let mut child = self
                    .child
                    .lock()
                    .map_err(|_| anyhow::anyhow!("ffmpeg process lock was poisoned"))?;
                let process = child
                    .as_mut()
                    .context("ffmpeg process is no longer available")?;
                if let Some(status) = process.try_wait()? {
                    child.take();
                    return Ok(status);
                }
            }
            if Instant::now() >= deadline {
                self.terminate();
                anyhow::bail!("ffmpeg did not exit within the shutdown timeout")
            }
            thread::sleep(Duration::from_millis(20));
        }
    }

    fn terminate(&mut self) {
        self.terminator().terminate();
    }
}

struct CancellationWatchdog {
    done: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
}

impl CancellationWatchdog {
    fn spawn(running: Arc<AtomicBool>, terminator: ChildTerminator) -> Result<Self> {
        let done = Arc::new(AtomicBool::new(false));
        let watch_done = Arc::clone(&done);
        let handle = thread::Builder::new()
            .name("manwe-ffmpeg-watchdog".to_string())
            .spawn(move || {
                while running.load(Ordering::Acquire) && !watch_done.load(Ordering::Acquire) {
                    thread::sleep(Duration::from_millis(50));
                }
                if !running.load(Ordering::Acquire) {
                    terminator.terminate();
                }
            })
            .context("failed to start ffmpeg cancellation watchdog")?;
        Ok(Self {
            done,
            handle: Some(handle),
        })
    }
}

impl Drop for CancellationWatchdog {
    fn drop(&mut self) {
        self.done.store(true, Ordering::Release);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

impl Drop for ChildGuard {
    fn drop(&mut self) {
        self.terminate();
    }
}

#[derive(Component)]
struct CameraView {
    index: usize,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct GridPlacement {
    x: f32,
    y: f32,
    scale: f32,
}

fn grid_placements(
    count: usize,
    frame_width: f32,
    frame_height: f32,
    viewport_width: f32,
    viewport_height: f32,
) -> Vec<GridPlacement> {
    if count == 0
        || [frame_width, frame_height, viewport_width, viewport_height]
            .into_iter()
            .any(|value| !value.is_finite() || value <= 0.0)
    {
        return Vec::new();
    }

    let mut best_columns = 1_usize;
    let mut best_rows = count;
    let mut best_scale = 0.0_f32;
    for columns in 1..=count {
        let rows = count.div_ceil(columns);
        let cell_width = viewport_width / columns as f32;
        let cell_height = viewport_height / rows as f32;
        let scale = VIEW_CELL_FILL * (cell_width / frame_width).min(cell_height / frame_height);
        let scale_tie = (scale - best_scale).abs() <= f32::EPSILON;
        if scale > best_scale || (scale_tie && rows < best_rows) {
            best_columns = columns;
            best_rows = rows;
            best_scale = scale;
        }
    }

    let cell_width = viewport_width / best_columns as f32;
    let cell_height = viewport_height / best_rows as f32;
    (0..count)
        .map(|index| {
            let row = index / best_columns;
            let column = index % best_columns;
            let items_in_row = (count - row * best_columns).min(best_columns);
            GridPlacement {
                x: (column as f32 - (items_in_row - 1) as f32 / 2.0) * cell_width,
                y: ((best_rows - 1) as f32 / 2.0 - row as f32) * cell_height,
                scale: best_scale,
            }
        })
        .collect()
}

fn main() -> Result<()> {
    let args = Args::parse();
    for name in ["MANWE_RTSP_URLS", "MANWE_FFMPEG", "MANWE_CONTRACT"] {
        std::env::remove_var(name);
    }
    validate_stream_urls(&args.urls)?;
    if args.urls.len() > MAX_STREAMS {
        anyhow::bail!("at most {MAX_STREAMS} concurrent streams are supported")
    }
    let pixels = args
        .width
        .checked_mul(args.height)
        .context("frame dimensions overflowed")?;
    if pixels > MAX_FRAME_PIXELS {
        anyhow::bail!("decoded frames must not exceed {MAX_FRAME_PIXELS} pixels")
    }
    let stream_count = args.urls.len();
    let ffmpeg = resolve_executable(&args.ffmpeg)?;
    ffmpeg.require_native_executable()?;
    let model_byte_budget = viewer_model_byte_budget(pixels, stream_count)?;
    let package = VerifiedModelPackage::load_with_artifact_limit(
        &args.contract,
        u64::try_from(model_byte_budget).context("viewer model budget exceeds u64")?,
    )?;
    validate_viewer_work(pixels, stream_count, package.artifact_bytes().len())?;
    let runtime = NativeRuntime::from_verified_package(package, args.cpu)?;
    let running = Arc::new(AtomicBool::new(true));
    let handles = Arc::new(Mutex::new(Vec::with_capacity(stream_count + 1)));
    let failure = Arc::new(Mutex::new(None));
    let frame_buffer = FrameBuffer {
        frames: Arc::new(Mutex::new(
            (0..stream_count).map(|_| DisplaySlot::default()).collect(),
        )),
    };
    let inference_queue = InferenceQueue::new(stream_count);
    let inference_handle = spawn_inference_worker(
        runtime,
        inference_queue.clone(),
        frame_buffer.clone(),
        Arc::clone(&running),
        Arc::clone(&failure),
    )?;
    match handles.lock() {
        Ok(mut worker_handles) => worker_handles.push(inference_handle),
        Err(poisoned) => poisoned.into_inner().push(inference_handle),
    }

    // Bevy 0.14 made `App::run` return the terminal `AppExit`; surface a failing
    // exit instead of discarding it.
    let exit = App::new()
        .add_plugins(DefaultPlugins)
        .insert_resource(args)
        .insert_resource(FfmpegExecutable(ffmpeg))
        .insert_resource(inference_queue)
        .insert_resource(WorkerControl {
            running,
            handles,
            failure,
        })
        .insert_resource(frame_buffer)
        .add_systems(Startup, setup)
        .add_systems(
            Update,
            (layout_views, update_frame, propagate_worker_failure),
        )
        .add_systems(Last, shutdown_workers)
        .run();
    if let AppExit::Error(code) = exit {
        anyhow::bail!("viewer exited with status {code}")
    }
    Ok(())
}

fn setup(
    mut commands: Commands,
    args: Res<Args>,
    inference_queue: Res<InferenceQueue>,
    ffmpeg: Res<FfmpegExecutable>,
    frame_buffer: Res<FrameBuffer>,
    worker_control: Res<WorkerControl>,
) {
    // Bevy 0.15 replaced bundles with required components: `Camera2d` pulls in
    // `Camera`, the orthographic projection, and the 2D render graph by itself.
    commands.spawn(Camera2d);

    for (index, url) in args.urls.iter().cloned().enumerate() {
        let buffer = Arc::clone(&frame_buffer.frames);
        let worker_queue = inference_queue.clone();
        let running = Arc::clone(&worker_control.running);
        let failure = Arc::clone(&worker_control.failure);
        let config = StreamConfig {
            index,
            url,
            ffmpeg: ffmpeg.0.clone(),
            width: args.width,
            height: args.height,
            inference_every: args.inference_every,
        };
        let thread_name = format!("manwe-stream-{index}");
        let spawn_result = thread::Builder::new().name(thread_name).spawn(move || {
            let result = catch_unwind(AssertUnwindSafe(|| {
                let mut retry_delay = INITIAL_RETRY_DELAY;
                while running.load(Ordering::Acquire) {
                    let mut made_progress = false;
                    match run_stream(
                        &config,
                        &worker_queue,
                        &buffer,
                        &running,
                        &mut made_progress,
                    ) {
                        Ok(()) => {}
                        Err(StreamAttemptFailure::Retry(error)) => {
                            if running.load(Ordering::Acquire) {
                                eprintln!("stream {index} interrupted: {error:#}; retrying");
                            }
                        }
                        Err(StreamAttemptFailure::Fatal(error)) => {
                            record_worker_failure(
                                &running,
                                &failure,
                                format!("stream {index} worker failed: {error:#}"),
                            );
                            worker_queue.wake_all();
                            break;
                        }
                    }
                    // A stream that delivered a frame was healthy. Its next
                    // disconnect deserves the fast first retry instead of
                    // inheriting an old 30-second startup backoff forever.
                    if made_progress {
                        retry_delay = INITIAL_RETRY_DELAY;
                    }
                    if !sleep_while_running(&running, retry_delay) {
                        break;
                    }
                    retry_delay = retry_delay.saturating_mul(2).min(MAX_RETRY_DELAY);
                }
            }));
            if let Err(payload) = result {
                record_worker_failure(
                    &running,
                    &failure,
                    format!(
                        "stream {index} worker panicked: {}",
                        panic_payload_message(&payload)
                    ),
                );
                worker_queue.wake_all();
            }
        });
        let handle = match spawn_result {
            Ok(handle) => handle,
            Err(error) => {
                record_worker_failure(
                    &worker_control.running,
                    &worker_control.failure,
                    format!("failed to start stream {index} worker: {error}"),
                );
                inference_queue.wake_all();
                break;
            }
        };
        match worker_control.handles.lock() {
            Ok(mut handles) => handles.push(handle),
            Err(poisoned) => poisoned.into_inner().push(handle),
        }
    }

    for index in 0..args.urls.len() {
        // `Sprite` (0.15+) carries the texture handle and requires `Transform`
        // and `Visibility`, replacing `SpriteBundle`.
        commands.spawn((Sprite::default(), CameraView { index }));
    }
}

fn run_stream(
    config: &StreamConfig,
    inference_queue: &InferenceQueue,
    buffer: &Arc<Mutex<Vec<DisplaySlot>>>,
    running: &Arc<AtomicBool>,
    made_progress: &mut bool,
) -> std::result::Result<(), StreamAttemptFailure> {
    if !running.load(Ordering::Acquire) {
        return Ok(());
    }
    let frame_size = config
        .width
        .checked_mul(config.height)
        .and_then(|pixels| pixels.checked_mul(3))
        .ok_or_else(|| {
            StreamAttemptFailure::Fatal(anyhow::anyhow!("frame dimensions overflowed"))
        })?;
    config
        .ffmpeg
        .verify()
        .map_err(StreamAttemptFailure::Fatal)?;
    let mut command = Command::new(config.ffmpeg.path());
    command
        .args([
            "-nostdin",
            "-loglevel",
            "error",
            "-max_alloc",
            "268435456",
            "-threads",
            "1",
            "-filter_threads",
            "1",
            "-rw_timeout",
            "10000000",
            "-f",
            "concat",
            "-safe",
            "0",
            "-protocol_whitelist",
            "pipe,rtsp,rtsps,tcp,udp,rtp,tls",
            "-i",
            "pipe:0",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-vcodec",
            "rawvideo",
            "-vf",
            &format!(
                "scale={0}:{1}:force_original_aspect_ratio=decrease:flags=bicubic:param0=0:param1=0.5,setsar=1,pad={0}:{1}:(ow-iw)/2:(oh-ih)/2:color=0x727272",
                config.width, config.height,
            ),
            "-",
        ])
        .env_clear()
        .env("LANG", "C")
        .env("LC_ALL", "C")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null());
    let mut child = ChildGuard::new(
        command
            .spawn()
            .with_context(|| format!("failed to start {}", config.ffmpeg.path().display()))
            .map_err(StreamAttemptFailure::Retry)?,
    );
    let watchdog = CancellationWatchdog::spawn(Arc::clone(running), child.terminator())
        .map_err(StreamAttemptFailure::Fatal)?;
    let mut stdin = child.take_stdin().map_err(StreamAttemptFailure::Fatal)?;
    if let Err(error) = write!(stdin, "ffconcat version 1.0\nfile '{}'\n", config.url) {
        return Err(StreamAttemptFailure::Retry(
            anyhow::Error::from(error).context("failed to send the private stream URL to ffmpeg"),
        ));
    }
    drop(stdin);
    let mut stdout = child.take_stdout().map_err(StreamAttemptFailure::Fatal)?;
    let mut data = vec![0_u8; frame_size];
    let processing_result = (|| -> std::result::Result<(), StreamAttemptFailure> {
        loop {
            match stdout.read_exact(&mut data) {
                Ok(()) => {}
                Err(error) if error.kind() == std::io::ErrorKind::UnexpectedEof => break,
                Err(error) => return Err(StreamAttemptFailure::Retry(error.into())),
            }
            if !running.load(Ordering::Acquire) {
                break;
            }

            let image = ImageBuffer::<Rgb<u8>, _>::from_raw(
                config.width as u32,
                config.height as u32,
                data.clone(),
            )
            .ok_or_else(|| {
                StreamAttemptFailure::Fatal(anyhow::anyhow!(
                    "ffmpeg returned an incorrectly sized frame"
                ))
            })?;
            let image = Arc::new(DynamicImage::ImageRgb8(image));
            let frame_sequence = publish_captured_frame(buffer, config.index, Arc::clone(&image))
                .map_err(StreamAttemptFailure::Fatal)?;
            *made_progress = true;
            if frame_sequence.is_multiple_of(config.inference_every as u64) {
                inference_queue
                    .submit(ScheduledFrame {
                        stream_index: config.index,
                        sequence: frame_sequence,
                        image,
                    })
                    .map_err(StreamAttemptFailure::Fatal)?;
            }
        }
        Ok(())
    })();

    drop(stdout);
    drop(watchdog);
    if !running.load(Ordering::Acquire) {
        child.terminate();
        return Ok(());
    }
    processing_result?;
    let status = child
        .wait(Duration::from_secs(5))
        .map_err(StreamAttemptFailure::Retry)?;
    if !status.success() {
        return Err(StreamAttemptFailure::Retry(anyhow::anyhow!(
            "ffmpeg exited with {status}"
        )));
    }
    Ok(())
}

fn spawn_inference_worker(
    runtime: NativeRuntime,
    inference_queue: InferenceQueue,
    frame_buffer: FrameBuffer,
    running: Arc<AtomicBool>,
    failure: Arc<Mutex<Option<String>>>,
) -> Result<JoinHandle<()>> {
    thread::Builder::new()
        .name("manwe-inference".to_string())
        .spawn(move || {
            let result = catch_unwind(AssertUnwindSafe(|| {
                run_inference_worker(&runtime, &inference_queue, &frame_buffer.frames, &running)
            }));
            let message = match result {
                Ok(Ok(())) => None,
                Ok(Err(error)) => Some(format!("native inference worker failed: {error:#}")),
                Err(payload) => Some(format!(
                    "native inference worker panicked: {}",
                    panic_payload_message(&payload)
                )),
            };
            if let Some(message) = message {
                record_worker_failure(&running, &failure, message);
                inference_queue.wake_all();
            }
        })
        .context("failed to start native inference worker")
}

fn panic_payload_message(payload: &Box<dyn std::any::Any + Send>) -> &str {
    payload
        .downcast_ref::<&str>()
        .copied()
        .or_else(|| payload.downcast_ref::<String>().map(String::as_str))
        .unwrap_or("non-string panic payload")
}

fn record_worker_failure(running: &AtomicBool, failure: &Mutex<Option<String>>, message: String) {
    // A normal window close sets `running` first. Do not turn that shutdown into
    // a spurious error if a worker happens to unwind concurrently.
    if !running.swap(false, Ordering::AcqRel) {
        return;
    }
    match failure.lock() {
        Ok(mut failure) => *failure = Some(message),
        Err(poisoned) => *poisoned.into_inner() = Some(message),
    }
}

fn run_inference_worker(
    runtime: &NativeRuntime,
    inference_queue: &InferenceQueue,
    frame_buffer: &Arc<Mutex<Vec<DisplaySlot>>>,
    running: &AtomicBool,
) -> Result<()> {
    while let Some(frame) = inference_queue.take_next(running)? {
        let stream_index = frame.stream_index;
        let sequence = frame.sequence;
        let image = Arc::unwrap_or_clone(frame.image);
        let annotated = Arc::new(
            runtime
                .infer(image, 14)
                .with_context(|| format!("stream {stream_index} inference failed"))?
                .into_annotated_image(),
        );
        publish_annotated_monotonically(frame_buffer, stream_index, sequence, annotated)?;
    }
    Ok(())
}

fn publish_captured_frame(
    frame_buffer: &Arc<Mutex<Vec<DisplaySlot>>>,
    stream_index: usize,
    image: Arc<DynamicImage>,
) -> Result<u64> {
    let mut frames = frame_buffer
        .lock()
        .map_err(|_| anyhow::anyhow!("frame buffer was poisoned"))?;
    let slot = frames
        .get_mut(stream_index)
        .context("stream frame has an invalid display index")?;
    let sequence = slot
        .latest_capture_sequence
        .checked_add(1)
        .context("stream frame sequence overflowed")?;
    slot.latest_capture_sequence = sequence;
    // Show raw video only during inference warm-up. Once the first exact
    // annotated result arrives, later captures must not erase it before Bevy
    // can render it.
    if !slot.annotation_ready {
        slot.pending = Some(image);
    }
    Ok(sequence)
}

fn publish_annotated_monotonically(
    frame_buffer: &Arc<Mutex<Vec<DisplaySlot>>>,
    stream_index: usize,
    sequence: u64,
    annotated: Arc<DynamicImage>,
) -> Result<bool> {
    let mut frames = frame_buffer
        .lock()
        .map_err(|_| anyhow::anyhow!("frame buffer was poisoned"))?;
    let slot = frames
        .get_mut(stream_index)
        .context("inference result has an invalid stream index")?;
    // Results are exact rendered frames, not overlays transplanted onto a newer
    // capture. Publish them monotonically; the replaceable queue bounds latency
    // while preserving a usable annotated viewer at normal capture frame rates.
    if sequence > slot.latest_capture_sequence {
        anyhow::bail!("inference result is newer than the latest captured frame")
    }
    if sequence <= slot.latest_annotation_sequence {
        return Ok(false);
    }
    slot.latest_annotation_sequence = sequence;
    slot.annotation_ready = true;
    slot.pending = Some(annotated);
    Ok(true)
}

fn validate_stream_urls(urls: &[String]) -> Result<()> {
    if urls.is_empty() || urls.iter().any(|url| validate_rtsp_url(url).is_err()) {
        anyhow::bail!(INVALID_RTSP_URL)
    }
    Ok(())
}

fn validate_viewer_work(pixels: usize, streams: usize, model_bytes: usize) -> Result<()> {
    if model_bytes == 0 || model_bytes > viewer_model_byte_budget(pixels, streams)? {
        anyhow::bail!(
            "requested streams, frame size, and model exceed the {MAX_VIEWER_WORK_BYTES}-byte viewer work budget"
        )
    }
    Ok(())
}

fn viewer_model_byte_budget(pixels: usize, streams: usize) -> Result<usize> {
    if !(1..=MAX_STREAMS).contains(&streams) {
        anyhow::bail!("viewer stream count must be between 1 and {MAX_STREAMS}")
    }
    let frame_bytes = u64::try_from(pixels)?
        .checked_mul(3)
        .context("frame byte count overflowed")?;
    let per_stream = frame_bytes
        .checked_mul(10)
        .and_then(|bytes| bytes.checked_add(32 * 1024 * 1024))
        .context("viewer per-stream estimate overflowed")?;
    let stream_work = per_stream
        .checked_mul(u64::try_from(streams)?)
        .context("viewer aggregate work estimate overflowed")?;
    let fixed_work = stream_work
        .checked_add(64 * 1024 * 1024)
        .context("viewer aggregate work estimate overflowed")?;
    let model_budget = MAX_VIEWER_WORK_BYTES
        .checked_sub(fixed_work)
        .context("requested streams and frame size exhaust the viewer work budget")?
        / 3;
    usize::try_from(model_budget).context("viewer model budget exceeds usize")
}

fn sleep_while_running(running: &AtomicBool, duration: Duration) -> bool {
    let deadline = Instant::now() + duration;
    while running.load(Ordering::Acquire) {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return true;
        }
        thread::sleep(remaining.min(Duration::from_millis(100)));
    }
    false
}

// Bevy 0.17 split buffered events out of `Event` into `Message`; `AppExit` is a
// message, so it is drained with a `MessageReader` instead of an `EventReader`.
fn propagate_worker_failure(
    control: Res<WorkerControl>,
    mut exit_messages: MessageWriter<AppExit>,
) {
    let failure = match control.failure.lock() {
        Ok(mut failure) => failure.take(),
        Err(poisoned) => poisoned.into_inner().take(),
    };
    if let Some(failure) = failure {
        eprintln!("{failure}");
        exit_messages.write(AppExit::error());
    }
}

fn shutdown_workers(
    mut exit_messages: MessageReader<AppExit>,
    control: Res<WorkerControl>,
    inference_queue: Res<InferenceQueue>,
) {
    if exit_messages.read().next().is_none() {
        return;
    }
    control.running.store(false, Ordering::Release);
    inference_queue.wake_all();
    let handles = match control.handles.lock() {
        Ok(mut handles) => handles.drain(..).collect::<Vec<_>>(),
        Err(poisoned) => poisoned.into_inner().drain(..).collect::<Vec<_>>(),
    };
    for handle in handles {
        if handle.join().is_err() {
            eprintln!("a stream worker panicked during shutdown");
        }
    }
}

fn update_frame(
    frame_buffer: Res<FrameBuffer>,
    mut images: ResMut<Assets<Image>>,
    mut query: Query<(&CameraView, &mut Sprite)>,
) {
    let pending = {
        let Ok(mut frames) = frame_buffer.frames.lock() else {
            return;
        };
        frames
            .iter_mut()
            .enumerate()
            .filter_map(|(index, slot)| slot.pending.take().map(|image| (index, image)))
            .collect::<Vec<_>>()
    };

    for (view, mut sprite) in &mut query {
        let Some((_, frame)) = pending.iter().find(|(index, _)| *index == view.index) else {
            continue;
        };
        let rgba = frame.to_rgba8();
        let width = rgba.width();
        let height = rgba.height();
        let image = Image::new(
            Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            TextureDimension::D2,
            rgba.into_raw(),
            TextureFormat::Rgba8UnormSrgb,
            RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
        );
        // `Assets::get_mut` now hands back a change-tracking `AssetMut` guard that
        // borrows `images` until it is dropped, so the in-place update has to be
        // scoped before the fallback insert can borrow `images` again.
        if let Some(mut existing) = images.get_mut(&sprite.image) {
            *existing = image;
            continue;
        }
        sprite.image = images.add(image);
    }
}

fn layout_views(
    args: Res<Args>,
    windows: Query<&Window, With<PrimaryWindow>>,
    mut views: Query<(&CameraView, &mut Transform)>,
) {
    // `Query::get_single` was renamed to `Query::single` (returning `Result`).
    let Ok(window) = windows.single() else {
        return;
    };
    let placements = grid_placements(
        args.urls.len(),
        args.width as f32,
        args.height as f32,
        window.width(),
        window.height(),
    );
    for (view, mut transform) in &mut views {
        let Some(placement) = placements.get(view.index) else {
            continue;
        };
        transform.translation = Vec3::new(placement.x, placement.y, 0.0);
        transform.scale = Vec3::splat(placement.scale);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    fn scheduled_frame(stream_index: usize, sequence: u64) -> ScheduledFrame {
        ScheduledFrame {
            stream_index,
            sequence,
            image: Arc::new(DynamicImage::ImageRgb8(image::RgbImage::new(1, 1))),
        }
    }

    #[test]
    fn inference_scheduler_is_fair_and_replaces_per_stream_backlog() {
        let queue = InferenceQueue::new(3);
        let running = AtomicBool::new(true);
        queue.submit(scheduled_frame(0, 1)).unwrap();
        queue.submit(scheduled_frame(0, 2)).unwrap();
        queue.submit(scheduled_frame(1, 1)).unwrap();

        let first = queue.take_next(&running).unwrap().unwrap();
        assert_eq!((first.stream_index, first.sequence), (0, 2));

        queue.submit(scheduled_frame(0, 3)).unwrap();
        queue.submit(scheduled_frame(2, 1)).unwrap();
        let second = queue.take_next(&running).unwrap().unwrap();
        let third = queue.take_next(&running).unwrap().unwrap();
        let fourth = queue.take_next(&running).unwrap().unwrap();
        assert_eq!((second.stream_index, second.sequence), (1, 1));
        assert_eq!((third.stream_index, third.sequence), (2, 1));
        assert_eq!((fourth.stream_index, fourth.sequence), (0, 3));
    }

    #[test]
    fn exact_annotations_advance_monotonically_without_raw_frame_erasure() {
        let raw = Arc::new(DynamicImage::ImageRgb8(image::RgbImage::new(2, 2)));
        let frames = Arc::new(Mutex::new(vec![DisplaySlot {
            latest_capture_sequence: 8,
            latest_annotation_sequence: 6,
            annotation_ready: true,
            pending: Some(Arc::clone(&raw)),
        }]));
        let annotation = Arc::new(DynamicImage::ImageRgb8(image::RgbImage::new(3, 3)));

        assert!(publish_annotated_monotonically(&frames, 0, 7, Arc::clone(&annotation)).unwrap());
        assert!(Arc::ptr_eq(
            frames.lock().unwrap()[0].pending.as_ref().unwrap(),
            &annotation,
        ));

        let newer_raw = Arc::new(DynamicImage::ImageRgb8(image::RgbImage::new(4, 4)));
        assert_eq!(publish_captured_frame(&frames, 0, newer_raw).unwrap(), 9);
        assert!(Arc::ptr_eq(
            frames.lock().unwrap()[0].pending.as_ref().unwrap(),
            &annotation,
        ));

        let duplicate = Arc::new(DynamicImage::ImageRgb8(image::RgbImage::new(5, 5)));
        assert!(!publish_annotated_monotonically(&frames, 0, 7, duplicate).unwrap());
    }

    #[test]
    fn private_pipe_manifest_rejects_concat_metacharacters() {
        assert!(validate_rtsp_url("rtsp://example.invalid/camera'\nfile '/tmp/other").is_err());
        assert!(validate_rtsp_url("rtsp://example.invalid/live\\escape").is_err());
    }

    #[test]
    fn cli_has_one_contract_authority_for_the_model() {
        let args = Args::try_parse_from([
            "camera_view",
            "--url",
            "rtsp://example.invalid/live",
            "--contract",
            "/tmp/model.contract.json",
        ])
        .unwrap();

        assert_eq!(args.contract, PathBuf::from("/tmp/model.contract.json"));
    }

    #[test]
    fn invalid_url_is_rejected_after_clap_without_reflecting_its_value() {
        let sensitive_invalid = "credential-like-value";
        let args = Args::try_parse_from([
            "camera_view",
            "--url",
            sensitive_invalid,
            "--contract",
            "/tmp/model.contract.json",
        ])
        .unwrap();

        let error = validate_stream_urls(&args.urls).unwrap_err().to_string();

        assert_eq!(error, INVALID_RTSP_URL);
        assert!(!error.contains(sensitive_invalid));
    }

    #[test]
    fn credential_environment_values_are_hidden_from_help() {
        let command = Args::command();
        let urls = command
            .get_arguments()
            .find(|argument| argument.get_id() == "urls")
            .unwrap();

        assert!(urls.is_hide_env_values_set());
    }

    #[test]
    fn aggregate_work_budget_rejects_excessive_viewer_memory() {
        let error =
            validate_viewer_work(MAX_FRAME_PIXELS, MAX_STREAMS, 512 * 1024 * 1024).unwrap_err();

        assert!(error.to_string().contains("viewer work budget"));
        assert!(viewer_model_byte_budget(MAX_FRAME_PIXELS, MAX_STREAMS).is_err());
        assert!(viewer_model_byte_budget(1280 * 720, 1).unwrap() < MAX_VIEWER_WORK_BYTES as usize);
    }

    #[test]
    fn grid_layout_fits_and_separates_every_supported_stream_count() {
        let frame_width = 1280.0;
        let frame_height = 720.0;

        for (viewport_width, viewport_height) in [(1280.0, 720.0), (720.0, 1280.0)] {
            for count in 1..=MAX_STREAMS {
                let placements = grid_placements(
                    count,
                    frame_width,
                    frame_height,
                    viewport_width,
                    viewport_height,
                );
                assert_eq!(placements.len(), count);
                for placement in &placements {
                    let half_width = frame_width * placement.scale / 2.0;
                    let half_height = frame_height * placement.scale / 2.0;
                    assert!(placement.x.abs() + half_width <= viewport_width / 2.0);
                    assert!(placement.y.abs() + half_height <= viewport_height / 2.0);
                }
                for (index, first) in placements.iter().enumerate() {
                    for second in &placements[index + 1..] {
                        let overlaps_horizontally =
                            (first.x - second.x).abs() < frame_width * first.scale;
                        let overlaps_vertically =
                            (first.y - second.y).abs() < frame_height * first.scale;
                        assert!(!(overlaps_horizontally && overlaps_vertically));
                    }
                }
            }
        }

        let three_views = grid_placements(3, frame_width, frame_height, 1280.0, 720.0);
        assert!(three_views.iter().any(|view| view.y != three_views[0].y));
    }

    #[test]
    fn cancellation_watchdog_terminates_a_blocked_child() {
        let child = Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "tests::child_guard_blocking_helper"])
            .env_clear()
            .env("MANWE_CHILD_GUARD_TEST", "1")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .unwrap();
        let child = ChildGuard::new(child);
        let running = Arc::new(AtomicBool::new(true));
        let watchdog =
            CancellationWatchdog::spawn(Arc::clone(&running), child.terminator()).unwrap();
        let started = Instant::now();

        running.store(false, Ordering::Release);
        drop(watchdog);

        assert!(started.elapsed() < Duration::from_secs(5));
        drop(child);
    }

    #[test]
    fn child_guard_blocking_helper() {
        if std::env::var_os("MANWE_CHILD_GUARD_TEST").is_some() {
            loop {
                thread::sleep(Duration::from_secs(60));
            }
        }
    }
}
