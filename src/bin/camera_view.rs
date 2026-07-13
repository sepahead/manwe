use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Mutex,
};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use bevy::app::AppExit;
use bevy::asset::RenderAssetUsages;
use bevy::prelude::*;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};
use bevy::window::PrimaryWindow;
use candle::{DType, Device};
use candle_nn::{Module, VarBuilder};
use clap::Parser;
use image::{DynamicImage, ImageBuffer, Rgb};
use manwe::model::{Multiples, YoloV8};
use manwe::secure_io::{
    open_bounded_regular_file, read_bounded_open_file, resolve_executable, sha256_hex,
    ResolvedExecutable, MAX_MODEL_BYTES,
};
use manwe::stream_url::{validate_rtsp_url, INVALID_RTSP_URL};
use manwe::{device, prepare_image, report_detect};

const MAX_STREAMS: usize = 8;
const MAX_FRAME_PIXELS: usize = 16_777_216;
const MAX_VIEWER_WORK_BYTES: u64 = 1024 * 1024 * 1024;
const VIEW_CELL_FILL: f32 = 0.96;

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

fn sha256_digest(value: &str) -> std::result::Result<String, String> {
    if value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        Ok(value.to_ascii_lowercase())
    } else {
        Err("SHA-256 must contain exactly 64 hexadecimal characters".to_string())
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

    /// Local, non-symlinked YOLOv8n safetensors weights.
    #[arg(long, env = "MANWE_MODEL", hide_env_values = true)]
    model: PathBuf,

    /// Expected SHA-256 for the exact model artifact.
    #[arg(
        long,
        env = "MANWE_MODEL_SHA256",
        value_parser = sha256_digest,
        hide_env_values = true
    )]
    model_sha256: String,

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
    frames: Arc<Mutex<Vec<Option<DynamicImage>>>>,
}

#[derive(Resource, Clone)]
struct ModelBytes(Arc<[u8]>);

#[derive(Resource, Clone)]
struct FfmpegExecutable(ResolvedExecutable);

#[derive(Resource, Clone)]
struct WorkerControl {
    running: Arc<AtomicBool>,
    handles: Arc<Mutex<Vec<JoinHandle<()>>>>,
}

struct StreamWorker {
    device: Device,
    model: YoloV8,
}

impl StreamWorker {
    fn new(cpu: bool, model_bytes: &[u8]) -> Result<Self> {
        let device = device(cpu)?;
        let vb = VarBuilder::from_buffered_safetensors(model_bytes.to_vec(), DType::F32, &device)?;
        let model = YoloV8::load(vb, Multiples::n(), 80)?;
        Ok(Self { device, model })
    }
}

struct StreamConfig {
    index: usize,
    url: String,
    ffmpeg: ResolvedExecutable,
    width: usize,
    height: usize,
    inference_every: usize,
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
    fn spawn(running: Arc<AtomicBool>, terminator: ChildTerminator) -> Self {
        let done = Arc::new(AtomicBool::new(false));
        let watch_done = Arc::clone(&done);
        let handle = thread::spawn(move || {
            while running.load(Ordering::Acquire) && !watch_done.load(Ordering::Acquire) {
                thread::sleep(Duration::from_millis(50));
            }
            if !running.load(Ordering::Acquire) {
                terminator.terminate();
            }
        });
        Self {
            done,
            handle: Some(handle),
        }
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
    std::env::remove_var("MANWE_RTSP_URLS");
    std::env::remove_var("MANWE_MODEL");
    std::env::remove_var("MANWE_MODEL_SHA256");
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
    let model_bytes =
        load_verified_model_bytes(&args.model, &args.model_sha256, pixels, stream_count)?;

    // Bevy 0.14 made `App::run` return the terminal `AppExit`; surface a failing
    // exit instead of discarding it.
    let exit = App::new()
        .add_plugins(DefaultPlugins)
        .insert_resource(args)
        .insert_resource(FfmpegExecutable(ffmpeg))
        .insert_resource(ModelBytes(Arc::from(model_bytes)))
        .insert_resource(WorkerControl {
            running: Arc::new(AtomicBool::new(true)),
            handles: Arc::new(Mutex::new(Vec::with_capacity(stream_count))),
        })
        .insert_resource(FrameBuffer {
            frames: Arc::new(Mutex::new(vec![None; stream_count])),
        })
        .add_systems(Startup, setup)
        .add_systems(Update, (layout_views, update_frame))
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
    model_bytes: Res<ModelBytes>,
    ffmpeg: Res<FfmpegExecutable>,
    frame_buffer: Res<FrameBuffer>,
    worker_control: Res<WorkerControl>,
) {
    // Bevy 0.15 replaced bundles with required components: `Camera2d` pulls in
    // `Camera`, the orthographic projection, and the 2D render graph by itself.
    commands.spawn(Camera2d);

    for (index, url) in args.urls.iter().cloned().enumerate() {
        let buffer = Arc::clone(&frame_buffer.frames);
        let model_bytes = Arc::clone(&model_bytes.0);
        let running = Arc::clone(&worker_control.running);
        let cpu = args.cpu;
        let config = StreamConfig {
            index,
            url,
            ffmpeg: ffmpeg.0.clone(),
            width: args.width,
            height: args.height,
            inference_every: args.inference_every,
        };
        let handle = thread::spawn(move || {
            let worker = match StreamWorker::new(cpu, model_bytes.as_ref()) {
                Ok(worker) => worker,
                Err(error) => {
                    eprintln!("stream {index} model initialization failed: {error:#}");
                    return;
                }
            };
            let mut retry_delay = Duration::from_secs(1);
            while running.load(Ordering::Acquire) {
                if let Err(error) = run_stream(&config, &worker, &buffer, &running) {
                    if running.load(Ordering::Acquire) {
                        eprintln!("stream {index} interrupted: {error:#}; retrying");
                    }
                }
                if !sleep_while_running(&running, retry_delay) {
                    break;
                }
                retry_delay = retry_delay.saturating_mul(2).min(Duration::from_secs(30));
            }
        });
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
    worker: &StreamWorker,
    buffer: &Arc<Mutex<Vec<Option<DynamicImage>>>>,
    running: &Arc<AtomicBool>,
) -> Result<()> {
    if !running.load(Ordering::Acquire) {
        return Ok(());
    }
    let frame_size = config
        .width
        .checked_mul(config.height)
        .and_then(|pixels| pixels.checked_mul(3))
        .context("frame dimensions overflowed")?;
    config.ffmpeg.verify()?;
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
            .with_context(|| format!("failed to start {}", config.ffmpeg.path().display()))?,
    );
    let watchdog = CancellationWatchdog::spawn(Arc::clone(running), child.terminator());
    let mut stdin = child.take_stdin()?;
    if let Err(error) = write!(stdin, "ffconcat version 1.0\nfile '{}'\n", config.url) {
        return Err(error).context("failed to send the private stream URL to ffmpeg");
    }
    drop(stdin);
    let mut stdout = child.take_stdout()?;
    let mut data = vec![0_u8; frame_size];
    let mut frame_count = 0_usize;

    let processing_result = (|| -> Result<()> {
        loop {
            match stdout.read_exact(&mut data) {
                Ok(()) => {}
                Err(error) if error.kind() == std::io::ErrorKind::UnexpectedEof => break,
                Err(error) => return Err(error.into()),
            }
            if !running.load(Ordering::Acquire) {
                break;
            }

            let image = ImageBuffer::<Rgb<u8>, _>::from_raw(
                config.width as u32,
                config.height as u32,
                data.clone(),
            )
            .context("ffmpeg returned an incorrectly sized frame")?;
            let original = DynamicImage::ImageRgb8(image);
            frame_count += 1;

            let display = if frame_count.is_multiple_of(config.inference_every) {
                let (input, transform) = prepare_image(&original, 640, 32, &worker.device)?;
                let predictions = worker.model.forward(&input)?.squeeze(0)?;
                report_detect(&predictions, original, &transform, 0.25, 0.45, 14)?
            } else {
                original
            };
            let mut frames = buffer
                .lock()
                .map_err(|_| anyhow::anyhow!("frame buffer poisoned"))?;
            frames[config.index] = Some(display);
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
    let status = child.wait(Duration::from_secs(5))?;
    if !status.success() {
        anyhow::bail!("ffmpeg exited with {status}")
    }
    Ok(())
}

fn validate_stream_urls(urls: &[String]) -> Result<()> {
    if urls.is_empty() || urls.iter().any(|url| validate_rtsp_url(url).is_err()) {
        anyhow::bail!(INVALID_RTSP_URL)
    }
    Ok(())
}

fn load_verified_model_bytes(
    path: &Path,
    expected_sha256: &str,
    frame_pixels: usize,
    streams: usize,
) -> Result<Vec<u8>> {
    if path.extension().and_then(|ext| ext.to_str()) != Some("safetensors") {
        anyhow::bail!("model must be a .safetensors file")
    }
    let (mut file, identity) = open_bounded_regular_file(path, MAX_MODEL_BYTES)?;
    let model_bytes =
        usize::try_from(identity.len()).context("model is too large for this host")?;
    validate_viewer_work(frame_pixels, streams, model_bytes)?;
    let bytes = read_bounded_open_file(&mut file, identity, path, MAX_MODEL_BYTES)?;
    if sha256_hex(&bytes) != expected_sha256 {
        anyhow::bail!("model SHA-256 does not match the expected digest")
    }
    Ok(bytes)
}

fn validate_viewer_work(pixels: usize, streams: usize, model_bytes: usize) -> Result<()> {
    let frame_bytes = u64::try_from(pixels)?
        .checked_mul(3)
        .context("frame byte count overflowed")?;
    let per_stream = u64::try_from(model_bytes)?
        .checked_mul(3)
        .and_then(|bytes| {
            frame_bytes
                .checked_mul(10)
                .and_then(|frames| bytes.checked_add(frames))
        })
        .and_then(|bytes| bytes.checked_add(64 * 1024 * 1024))
        .context("viewer work estimate overflowed")?;
    let total = per_stream
        .checked_mul(u64::try_from(streams)?)
        .context("viewer aggregate work estimate overflowed")?;
    if total > MAX_VIEWER_WORK_BYTES {
        anyhow::bail!(
            "requested streams, frame size, and model exceed the {MAX_VIEWER_WORK_BYTES}-byte viewer work budget"
        )
    }
    Ok(())
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
fn shutdown_workers(mut exit_messages: MessageReader<AppExit>, control: Res<WorkerControl>) {
    if exit_messages.read().next().is_none() {
        return;
    }
    control.running.store(false, Ordering::Release);
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
            .filter_map(|(index, frame)| frame.take().map(|image| (index, image)))
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

    #[test]
    fn private_pipe_manifest_rejects_concat_metacharacters() {
        assert!(validate_rtsp_url("rtsp://example.invalid/camera'\nfile '/tmp/other").is_err());
        assert!(validate_rtsp_url("rtsp://example.invalid/live\\escape").is_err());
    }

    #[test]
    fn model_digest_parser_requires_exact_sha256() {
        assert!(sha256_digest(&"a".repeat(63)).is_err());
        assert_eq!(sha256_digest(&"A".repeat(64)).unwrap(), "a".repeat(64));
    }

    #[test]
    fn invalid_url_is_rejected_after_clap_without_reflecting_its_value() {
        let sensitive_invalid = "credential-like-value";
        let args = Args::try_parse_from([
            "camera_view",
            "--url",
            sensitive_invalid,
            "--model",
            "/tmp/model.safetensors",
            "--model-sha256",
            &"0".repeat(64),
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
        let watchdog = CancellationWatchdog::spawn(Arc::clone(&running), child.terminator());
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
