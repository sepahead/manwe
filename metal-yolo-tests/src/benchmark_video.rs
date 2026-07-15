use anyhow::Context;
use candle::{DType, Device, IndexOp, Tensor};
use candle_nn::{Module, VarBuilder};
use clap::Parser;
use image::{ImageBuffer, Rgb};
use std::fs::{self, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::{
    atomic::{AtomicBool, AtomicU64, AtomicU8, Ordering},
    Arc, Mutex,
};
use std::thread;
use std::time::{Duration, Instant};

use manwe::model::{Multiples, YoloV8};
use manwe::secure_io::{
    ensure_file_identity, open_bounded_regular_file, read_bounded_open_file,
    read_bounded_regular_file, resolve_executable, sha256_bounded_open_file, sha256_hex,
    BoundDirectory, FileIdentity, MAX_MODEL_BYTES, MAX_VIDEO_BYTES,
};
use manwe::{validate_coco_detection_output_schema, validate_coco_model_output};

const NUM_CLASSES: usize = 80;
const INPUT_W: usize = 640;
const INPUT_H: usize = 640;
const EXPECTED_COCO_PREDICTIONS: usize = (INPUT_W / 8) * (INPUT_H / 8)
    + (INPUT_W / 16) * (INPUT_H / 16)
    + (INPUT_W / 32) * (INPUT_H / 32);
const MAX_VIDEO_FRAMES: u64 = 1_000_000;
const MAX_PREDICTIONS: usize = 100_000;
const MAX_CANDIDATES: usize = 2_000;
const DEFAULT_MAX_DURATION_SECONDS: u64 = 7_200;
const MAX_BENCHMARK_DURATION_SECONDS: u64 = 30 * 24 * 60 * 60;
const MAX_RESULT_BYTES: u64 = 64 * 1024 * 1024;
const VIDEO_LETTERBOX_FILTER: &str = "scale=w='if(gte(iw,ih),640,max(1,floor(iw*640/ih)+gt(iw*640/ih-floor(iw*640/ih),0.5)+eq(iw*640/ih-floor(iw*640/ih),0.5)*mod(floor(iw*640/ih),2)))':h='if(gte(iw,ih),max(1,floor(ih*640/iw)+gt(ih*640/iw-floor(ih*640/iw),0.5)+eq(ih*640/iw-floor(ih*640/iw),0.5)*mod(floor(ih*640/iw),2)),640)':flags=bicubic:param0=0:param1=0.5,setsar=1,pad=640:640:(ow-iw)/2:(oh-ih)/2:color=0x727272";

fn bounded_fps(value: &str) -> std::result::Result<f64, String> {
    let parsed = value
        .parse::<f64>()
        .map_err(|_| format!("{value:?} is not a number"))?;
    if parsed == 0.0 || (parsed.is_finite() && (0.1..=1000.0).contains(&parsed)) {
        Ok(parsed)
    } else {
        Err("target FPS must be 0 (unthrottled) or between 0.1 and 1000".to_string())
    }
}

fn safe_run_id(value: &str) -> std::result::Result<String, String> {
    if !value.is_empty()
        && value.len() <= 64
        && value
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_'))
    {
        Ok(value.to_string())
    } else {
        Err("run id may contain only ASCII letters, digits, '-' and '_' (max 64)".to_string())
    }
}

fn sha256_digest(value: &str) -> std::result::Result<String, String> {
    if value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        Ok(value.to_ascii_lowercase())
    } else {
        Err("SHA-256 must contain exactly 64 hexadecimal characters".to_string())
    }
}

fn bounded_frame_count(value: &str) -> std::result::Result<u64, String> {
    let parsed = value
        .parse::<u64>()
        .map_err(|_| format!("{value:?} is not a positive integer"))?;
    if (1..=MAX_VIDEO_FRAMES).contains(&parsed) {
        Ok(parsed)
    } else {
        Err(format!(
            "frame count must be between 1 and {MAX_VIDEO_FRAMES}"
        ))
    }
}

fn bounded_duration_seconds(value: &str) -> std::result::Result<u64, String> {
    let parsed = value
        .parse::<u64>()
        .map_err(|_| format!("{value:?} is not a positive integer"))?;
    if (1..=MAX_BENCHMARK_DURATION_SECONDS).contains(&parsed) {
        Ok(parsed)
    } else {
        Err(format!(
            "duration must be between 1 and {MAX_BENCHMARK_DURATION_SECONDS} seconds"
        ))
    }
}

fn safe_video_name(path: &Path) -> anyhow::Result<String> {
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .context("video path must have a UTF-8 file name")?;
    if name.is_empty()
        || name.len() > 128
        || !name
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '.' | '-' | '_'))
    {
        anyhow::bail!(
            "video file name may contain only ASCII letters, digits, '.', '-' and '_' (max 128)"
        )
    }
    Ok(name.replace('.', "_"))
}

fn video_demuxer(path: &Path) -> anyhow::Result<&'static str> {
    let extension = path
        .extension()
        .and_then(|extension| extension.to_str())
        .map(str::to_ascii_lowercase)
        .context("video path must have a UTF-8 extension")?;
    match extension.as_str() {
        "mp4" | "m4v" | "mov" => Ok("mov"),
        "mkv" | "webm" => Ok("matroska"),
        "avi" => Ok("avi"),
        _ => anyhow::bail!("video must be MP4/M4V/MOV, MKV/WebM, or AVI"),
    }
}

#[derive(Clone)]
struct FramePacket {
    data: Vec<u8>,
    pipe_read_started_at: Instant,
    frame_read_wait_ms: f64,
    idx: u64,
}

#[derive(Parser, Clone)]
struct Args {
    #[arg(long)]
    video: PathBuf,
    #[arg(long, value_parser = bounded_fps)]
    target_fps: f64,
    #[arg(long, value_parser = safe_run_id)]
    run_id: String,
    #[arg(long)]
    save_video: bool,
    #[arg(long)]
    model: PathBuf,
    #[arg(long, value_parser = sha256_digest)]
    model_sha256: String,
    /// Maximum decoded frames accepted from this input.
    #[arg(long, default_value_t = 100_000, value_parser = bounded_frame_count)]
    max_frames: u64,
    /// Maximum wall-clock duration before the decoder is terminated.
    #[arg(
        long,
        default_value_t = DEFAULT_MAX_DURATION_SECONDS,
        value_parser = bounded_duration_seconds
    )]
    max_duration_seconds: u64,
    /// ffmpeg executable or absolute path.
    #[arg(long, default_value = "ffmpeg")]
    ffmpeg: PathBuf,
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

    fn take_stdin(&self) -> anyhow::Result<ChildStdin> {
        let mut child = self
            .child
            .lock()
            .map_err(|_| anyhow::anyhow!("child process lock was poisoned"))?;
        child
            .as_mut()
            .context("child process is no longer available")?
            .stdin
            .take()
            .context("child process did not expose stdin")
    }

    fn take_stdout(&self) -> anyhow::Result<ChildStdout> {
        let mut child = self
            .child
            .lock()
            .map_err(|_| anyhow::anyhow!("child process lock was poisoned"))?;
        child
            .as_mut()
            .context("child process is no longer available")?
            .stdout
            .take()
            .context("child process did not expose stdout")
    }

    fn wait(&mut self, timeout: Duration) -> anyhow::Result<std::process::ExitStatus> {
        let deadline = Instant::now() + timeout;
        loop {
            {
                let mut child = self
                    .child
                    .lock()
                    .map_err(|_| anyhow::anyhow!("child process lock was poisoned"))?;
                let process = child
                    .as_mut()
                    .context("child process is no longer available")?;
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

struct PipelineDeadlineState {
    value: AtomicU8,
}

impl PipelineDeadlineState {
    const PENDING: u8 = 0;
    const CANCELLED: u8 = 1;
    const TIMED_OUT: u8 = 2;

    fn new() -> Self {
        Self {
            value: AtomicU8::new(Self::PENDING),
        }
    }

    fn is_pending(&self) -> bool {
        self.value.load(Ordering::Acquire) == Self::PENDING
    }

    fn cancel(&self) -> bool {
        self.claim(Self::CANCELLED)
    }

    fn claim_timeout(&self) -> bool {
        self.claim(Self::TIMED_OUT)
    }

    fn timed_out(&self) -> bool {
        self.value.load(Ordering::Acquire) == Self::TIMED_OUT
    }

    fn claim(&self, outcome: u8) -> bool {
        self.value
            .compare_exchange(Self::PENDING, outcome, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
    }
}

struct PipelineDeadlineWatchdog {
    deadline: Instant,
    state: Arc<PipelineDeadlineState>,
    handle: Option<thread::JoinHandle<()>>,
}

impl PipelineDeadlineWatchdog {
    fn spawn(
        deadline: Instant,
        running: Arc<AtomicBool>,
        input: ChildTerminator,
        output: Option<ChildTerminator>,
    ) -> Self {
        let state = Arc::new(PipelineDeadlineState::new());
        let watch_state = Arc::clone(&state);
        let handle = thread::spawn(move || loop {
            if !watch_state.is_pending() {
                return;
            }
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                if !watch_state.claim_timeout() {
                    return;
                }
                running.store(false, Ordering::Release);
                if let Some(output) = &output {
                    output.terminate();
                }
                input.terminate();
                return;
            }
            thread::sleep(remaining.min(Duration::from_millis(50)));
        });
        Self {
            deadline,
            state,
            handle: Some(handle),
        }
    }

    fn finish(mut self) -> bool {
        // A completion observed before the deadline cancels the watchdog. Once
        // the deadline has passed, leave it armed so a delayed watchdog thread
        // still records the timeout and terminates the child processes.
        if Instant::now() < self.deadline {
            self.state.cancel();
        }
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
        self.state.timed_out()
    }
}

impl Drop for PipelineDeadlineWatchdog {
    fn drop(&mut self) {
        self.state.cancel();
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

fn read_complete_frame(reader: &mut impl Read, frame: &mut [u8]) -> std::io::Result<bool> {
    let mut offset = 0;
    while offset < frame.len() {
        match reader.read(&mut frame[offset..])? {
            0 if offset == 0 => return Ok(false),
            0 => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    "decoder returned a partial frame",
                ))
            }
            read => offset += read,
        }
    }
    Ok(true)
}

fn sleep_while_running(running: &AtomicBool, duration: Duration) -> bool {
    let deadline = Instant::now() + duration;
    while running.load(Ordering::Acquire) {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return true;
        }
        thread::sleep(remaining.min(Duration::from_millis(50)));
    }
    false
}

fn percentile(sorted_samples: &[f64], percentile: f64) -> f64 {
    if sorted_samples.is_empty() {
        return 0.0;
    }
    let index = ((percentile / 100.0) * (sorted_samples.len() - 1) as f64).round() as usize;
    sorted_samples[index.min(sorted_samples.len() - 1)]
}

fn create_private_directory(path: &Path) -> std::io::Result<()> {
    let mut builder = fs::DirBuilder::new();
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt;
        builder.mode(0o700);
    }
    builder.create(path)
}

fn verify_json_file(path: &Path, expected: &[u8]) -> anyhow::Result<FileIdentity> {
    let (mut file, identity) = open_bounded_regular_file(path, MAX_RESULT_BYTES)?;
    let actual = read_bounded_open_file(&mut file, identity, path, MAX_RESULT_BYTES)?;
    if actual != expected {
        anyhow::bail!("result JSON verification failed: {}", path.display())
    }
    serde_json::from_slice::<serde_json::Value>(&actual)
        .with_context(|| format!("result is not valid JSON: {}", path.display()))?;
    Ok(identity)
}

fn write_verified_json_once(path: &Path, value: &serde_json::Value) -> anyhow::Result<Vec<u8>> {
    let bytes = serde_json::to_vec_pretty(value)?;
    if bytes.is_empty() || bytes.len() as u64 > MAX_RESULT_BYTES {
        anyhow::bail!("result JSON must contain between 1 and {MAX_RESULT_BYTES} bytes")
    }
    let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
    file.write_all(&bytes)?;
    file.sync_all()?;
    drop(file);
    verify_json_file(path, &bytes)?;
    Ok(bytes)
}

fn verify_hard_link_identity(
    staged: &Path,
    published: &Path,
    max_bytes: u64,
) -> anyhow::Result<()> {
    let (staged_file, staged_identity) = open_bounded_regular_file(staged, max_bytes)?;
    drop(staged_file);
    let (published_file, published_identity) = open_bounded_regular_file(published, max_bytes)?;
    drop(published_file);
    if staged_identity != published_identity {
        anyhow::bail!("published file identity does not match its staged source")
    }
    Ok(())
}

#[derive(Debug)]
struct AuthenticatedFile {
    identity: FileIdentity,
    sha256: String,
}

fn digest_and_sync_regular_file(path: &Path, max_bytes: u64) -> anyhow::Result<AuthenticatedFile> {
    let (mut file, identity) = open_bounded_regular_file(path, max_bytes)?;
    let sha256 = sha256_bounded_open_file(&mut file, identity, path, max_bytes)?;
    file.sync_all()?;
    Ok(AuthenticatedFile { identity, sha256 })
}

fn verify_authenticated_file(
    path: &Path,
    expected: &AuthenticatedFile,
    max_bytes: u64,
) -> anyhow::Result<()> {
    let (mut file, identity) = open_bounded_regular_file(path, max_bytes)?;
    if identity != expected.identity {
        anyhow::bail!("staged artifact identity changed after authentication")
    }
    let sha256 = sha256_bounded_open_file(&mut file, identity, path, max_bytes)?;
    if sha256 != expected.sha256 {
        anyhow::bail!("staged artifact digest changed after authentication")
    }
    Ok(())
}

fn verify_expected_digest(
    path: &Path,
    expected_sha256: &str,
    max_bytes: u64,
) -> anyhow::Result<()> {
    let (mut file, identity) = open_bounded_regular_file(path, max_bytes)?;
    let sha256 = sha256_bounded_open_file(&mut file, identity, path, max_bytes)?;
    if sha256 != expected_sha256 {
        anyhow::bail!("published artifact digest does not match its authenticated digest")
    }
    Ok(())
}

fn path_occupied(path: &Path) -> anyhow::Result<bool> {
    match std::fs::symlink_metadata(path) {
        Ok(_) => Ok(true),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(error.into()),
    }
}

struct EvidenceRun {
    run_dir: BoundDirectory,
    stage_dir: BoundDirectory,
    stage_result: PathBuf,
    stage_video: Option<PathBuf>,
    result_path: PathBuf,
    output_path: Option<PathBuf>,
    final_link_created: bool,
    committed: bool,
}

impl EvidenceRun {
    fn acquire(
        run_dir: &Path,
        safe_name: &str,
        run_id: &str,
        save_video: bool,
    ) -> anyhow::Result<Self> {
        let run_dir = BoundDirectory::open(run_dir)?;
        let result_path = run_dir
            .path()
            .join(format!("res_rust_{safe_name}_{run_id}.json"));
        let output_path = save_video.then(|| {
            run_dir
                .path()
                .join(format!("video_rust_{safe_name}_{run_id}.mp4"))
        });
        let stage_path = run_dir
            .path()
            .join(format!(".manwe-benchmark-{safe_name}-{run_id}.in-progress"));
        run_dir.verify()?;
        match create_private_directory(&stage_path) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                anyhow::bail!("this benchmark run is already active or needs stale-run cleanup")
            }
            Err(error) => return Err(error.into()),
        }
        run_dir.verify()?;
        let stage_dir = match BoundDirectory::open(&stage_path) {
            Ok(directory) => directory,
            Err(error) => {
                if run_dir.verify().is_ok() {
                    let _ = fs::remove_dir_all(&stage_path);
                    let _ = run_dir.sync();
                }
                return Err(error);
            }
        };
        let run = Self {
            stage_result: stage_dir.path().join("result.json"),
            stage_video: save_video.then(|| stage_dir.path().join("output.mp4")),
            run_dir,
            stage_dir,
            result_path,
            output_path,
            final_link_created: false,
            committed: false,
        };
        run.run_dir.sync()?;
        run.stage_dir.verify()?;
        let output_occupied = match run.output_path.as_deref() {
            Some(path) => path_occupied(path)?,
            None => false,
        };
        if path_occupied(&run.result_path)? || output_occupied {
            anyhow::bail!("run id would overwrite existing benchmark evidence")
        }
        Ok(run)
    }

    fn staged_video_path(&self) -> Option<&Path> {
        self.stage_video.as_deref()
    }

    fn output_path(&self) -> Option<&Path> {
        self.output_path.as_deref()
    }

    fn publish(
        &mut self,
        results: &serde_json::Value,
        authenticated_video: Option<&AuthenticatedFile>,
    ) -> anyhow::Result<()> {
        let expected_result = write_verified_json_once(&self.stage_result, results)?;
        self.stage_dir.sync()?;
        self.run_dir.verify()?;
        match (
            self.stage_video.as_deref(),
            self.output_path.as_deref(),
            authenticated_video,
        ) {
            (Some(staged), Some(final_path), Some(authenticated)) => {
                self.stage_dir.verify()?;
                verify_authenticated_file(staged, authenticated, MAX_VIDEO_BYTES)?;
                self.stage_dir.verify()?;
                self.run_dir.verify()?;
                fs::hard_link(staged, final_path)?;
                self.final_link_created = true;
                self.run_dir.verify()?;
                self.stage_dir.verify()?;
                verify_hard_link_identity(staged, final_path, MAX_VIDEO_BYTES)?;
                verify_expected_digest(staged, &authenticated.sha256, MAX_VIDEO_BYTES)?;
                verify_expected_digest(final_path, &authenticated.sha256, MAX_VIDEO_BYTES)?;
            }
            (None, None, None) => {}
            _ => anyhow::bail!("video staging and authentication state is inconsistent"),
        }
        fs::hard_link(&self.stage_result, &self.result_path)?;
        self.final_link_created = true;
        self.run_dir.verify()?;
        self.stage_dir.verify()?;
        let stage_identity = verify_json_file(&self.stage_result, &expected_result)?;
        let result_identity = verify_json_file(&self.result_path, &expected_result)?;
        if stage_identity != result_identity {
            anyhow::bail!("published result identity does not match its staged source")
        }
        self.run_dir.sync()?;
        self.committed = true;
        match self
            .run_dir
            .verify()
            .and_then(|()| self.stage_dir.verify())
            .and_then(|()| fs::remove_dir_all(self.stage_dir.path()).map_err(Into::into))
        {
            Ok(()) => {
                if let Err(error) = self.run_dir.sync() {
                    eprintln!("benchmark evidence was published but cleanup sync failed: {error}");
                }
            }
            Err(error) => {
                eprintln!("benchmark evidence was published but staging cleanup failed: {error}");
            }
        }
        Ok(())
    }
}

impl Drop for EvidenceRun {
    fn drop(&mut self) {
        if (self.committed || !self.final_link_created)
            && self.run_dir.verify().is_ok()
            && self.stage_dir.verify().is_ok()
        {
            let _ = fs::remove_dir_all(self.stage_dir.path());
            let _ = self.run_dir.sync();
        }
    }
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let ffmpeg = resolve_executable(&args.ffmpeg)?;
    let run_dir = std::env::var("RUN_DIR").unwrap_or_else(|_| "video_results".to_string());
    let run_dir = std::path::PathBuf::from(run_dir);
    std::fs::create_dir_all(&run_dir)?;
    let safe_name = safe_video_name(&args.video)?;
    let input_demuxer = video_demuxer(&args.video)?;
    let mut evidence = EvidenceRun::acquire(&run_dir, &safe_name, &args.run_id, args.save_video)?;

    // 1. Load Model
    let device = Device::new_metal(0)?;
    println!("Loading Model on {device:?}...");

    let model_path = &args.model;
    if model_path.extension().and_then(|ext| ext.to_str()) != Some("safetensors") {
        anyhow::bail!("model must have a .safetensors extension")
    }
    let model_bytes = read_bounded_regular_file(model_path, MAX_MODEL_BYTES)?;
    let model_sha256 = sha256_hex(&model_bytes);
    if model_sha256 != args.model_sha256 {
        anyhow::bail!("model SHA-256 does not match the expected digest")
    }
    let (mut video_file, video_identity) = open_bounded_regular_file(&args.video, MAX_VIDEO_BYTES)?;
    let video_sha256 = sha256_bounded_open_file(
        &mut video_file,
        video_identity,
        &args.video,
        MAX_VIDEO_BYTES,
    )?;
    let mut video_verification_file = video_file.try_clone()?;
    let vb = VarBuilder::from_buffered_safetensors(model_bytes, DType::F32, &device)?;
    let model = YoloV8::load(vb, Multiples::s(), NUM_CLASSES)?;

    // Warm up before decoders start so setup work cannot pre-fill the input pipe.
    {
        let dummy = Tensor::zeros((1, 3, INPUT_H, INPUT_W), DType::F32, &device)?;
        let prediction = model.forward(&dummy)?;
        validate_coco_detection_output_schema(&prediction, EXPECTED_COCO_PREDICTIONS)?;
        device.synchronize()?;
        let _validated_prediction = validate_coco_model_output(&prediction)?;
    }

    // 2. Setup Input Pipe (FFmpeg -> Rust)
    // Decode into the same isotropic 640-square letterbox geometry as static inference.
    ffmpeg.verify()?;
    let mut input_command = Command::new(ffmpeg.path());
    input_command.args([
        "-nostdin",
        "-loglevel",
        "error",
        "-max_alloc",
        "268435456",
        "-threads",
        "1",
        "-filter_threads",
        "1",
        "-protocol_whitelist",
        "file,pipe",
        "-f",
        input_demuxer,
    ]);
    if input_demuxer == "mov" {
        input_command.args(["-use_absolute_path", "0", "-enable_drefs", "0"]);
    }
    input_command
        .args([
            "-i",
            "/dev/fd/0",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-vcodec",
            "rawvideo",
            "-vf",
            VIDEO_LETTERBOX_FILTER,
        ])
        .arg("-frames:v")
        .arg(args.max_frames.to_string())
        .arg("-")
        .env_clear()
        .env("LANG", "C")
        .env("LC_ALL", "C")
        .stdin(Stdio::from(video_file))
        .stdout(Stdio::piped())
        .stderr(Stdio::null());
    let mut input_process = ChildGuard::new(input_command.spawn()?);

    let mut input_stdout = input_process.take_stdout()?;

    // 3. Setup Output Pipe (Rust -> FFmpeg) if needed
    let (mut output_process, mut output_stdin) = if args.save_video {
        std::fs::create_dir_all(&run_dir)?;
        let output_fps = if args.target_fps > 0.0 {
            args.target_fps
        } else {
            30.0
        };
        ffmpeg.verify()?;
        let mut command = Command::new(ffmpeg.path());
        command
            .args([
                "-nostdin",
                "-loglevel",
                "error",
                "-threads",
                "1",
                "-filter_threads",
                "1",
                "-protocol_whitelist",
                "pipe,file",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                &format!("{INPUT_W}x{INPUT_H}"),
                "-r",
                &format!("{output_fps}"),
                "-i",
                "-",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-frames:v",
            ])
            .arg(args.max_frames.to_string())
            .args(["-n"])
            .arg(
                evidence
                    .staged_video_path()
                    .context("video staging path is unavailable")?,
            )
            .env_clear()
            .env("LANG", "C")
            .env("LC_ALL", "C")
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::null());
        let process = ChildGuard::new(command.spawn()?);
        let stdin = process.take_stdin()?;
        (Some(process), Some(stdin))
    } else {
        (None, None)
    };

    // 4. Reader thread: keep only the latest frame (drop older ones) to mirror RTSP simulation semantics.
    let running = Arc::new(AtomicBool::new(true));
    let frames_presented = Arc::new(AtomicU64::new(0));
    let frame_limit_reached = Arc::new(AtomicBool::new(false));
    let latest_frame: Arc<Mutex<Option<FramePacket>>> = Arc::new(Mutex::new(None));

    let reader_running = running.clone();
    let reader_presented = frames_presented.clone();
    let reader_limit_reached = frame_limit_reached.clone();
    let reader_latest = latest_frame.clone();
    let max_frames = args.max_frames;

    let reader_interval = if args.target_fps > 0.0 {
        Duration::from_secs_f64(1.0 / args.target_fps)
    } else {
        Duration::ZERO
    };
    let start_time = Instant::now();
    let benchmark_deadline = start_time + Duration::from_secs(args.max_duration_seconds);
    let watchdog = PipelineDeadlineWatchdog::spawn(
        benchmark_deadline,
        Arc::clone(&running),
        input_process.terminator(),
        output_process.as_ref().map(ChildGuard::terminator),
    );
    let reader_thread = thread::spawn(move || -> std::io::Result<()> {
        let frame_size = INPUT_W * INPUT_H * 3;
        let mut buffer = vec![0u8; frame_size];
        loop {
            let frame_start = Instant::now();
            match read_complete_frame(&mut input_stdout, &mut buffer) {
                Ok(true) => {
                    let frame_read_wait_ms = frame_start.elapsed().as_secs_f64() * 1000.0;
                    let idx = reader_presented.fetch_add(1, Ordering::Relaxed) + 1;
                    let packet = FramePacket {
                        data: buffer.clone(),
                        pipe_read_started_at: frame_start,
                        frame_read_wait_ms,
                        idx,
                    };
                    match reader_latest.lock() {
                        Ok(mut slot) => *slot = Some(packet),
                        Err(_) => {
                            reader_running.store(false, Ordering::Release);
                            return Err(std::io::Error::other("latest-frame buffer was poisoned"));
                        }
                    }
                    if idx >= max_frames {
                        reader_limit_reached.store(true, Ordering::Release);
                        reader_running.store(false, Ordering::Release);
                        break;
                    }
                    let elapsed = frame_start.elapsed();
                    if reader_interval > elapsed
                        && !sleep_while_running(&reader_running, reader_interval - elapsed)
                    {
                        break;
                    }
                }
                Ok(false) => {
                    reader_running.store(false, Ordering::Release);
                    break;
                }
                Err(error) => {
                    reader_running.store(false, Ordering::Release);
                    return Err(error);
                }
            }
        }
        Ok(())
    });

    // 5. Main Loop (Inference)
    let mut frames_processed: u64 = 0;
    let mut frame_read_wait_times = Vec::new();
    let mut preprocess_upload_sync_times = Vec::new();
    let mut model_forward_sync_times = Vec::new();
    let mut total_infer_times = Vec::new();
    let mut latencies = Vec::new();

    println!("Starting Rust Benchmark Loop...");

    let mut last_seen_idx: u64 = 0;

    let mut processing_result = (|| -> anyhow::Result<()> {
        while running.load(Ordering::Acquire)
            || last_seen_idx < frames_presented.load(Ordering::Acquire)
        {
            if Instant::now() >= benchmark_deadline {
                running.store(false, Ordering::Release);
                input_process.terminate();
                anyhow::bail!(
                    "benchmark exceeded the {}-second wall-clock limit",
                    args.max_duration_seconds
                )
            }
            let packet = {
                let slot = latest_frame
                    .lock()
                    .map_err(|_| anyhow::anyhow!("latest-frame buffer poisoned"))?;
                slot.as_ref().filter(|p| p.idx > last_seen_idx).cloned()
            };

            let Some(packet) = packet else {
                thread::sleep(Duration::from_millis(1));
                continue;
            };

            last_seen_idx = packet.idx;

            // Record raw-pipe read wait and increment processed count.
            frame_read_wait_times.push(packet.frame_read_wait_ms);
            frames_processed += 1;

            // 1. Preprocess/upload, synchronized before this stage is stopped.
            let t_preprocess_start = Instant::now();
            let tensor = Tensor::from_vec(packet.data.clone(), (INPUT_H, INPUT_W, 3), &device)?
                .permute((2, 0, 1))?
                .unsqueeze(0)?
                .to_dtype(DType::F32)?;
            let tensor = (tensor * (1.0 / 255.0))?;
            device.synchronize()?;
            let t_preprocess_end = Instant::now();
            let preprocess_upload_sync_ms = t_preprocess_end
                .duration_since(t_preprocess_start)
                .as_secs_f64()
                * 1000.0;
            preprocess_upload_sync_times.push(preprocess_upload_sync_ms);

            // 2. Time model forward through a transfer-free device synchronization barrier.
            let t_forward_start = Instant::now();
            let preds = model.forward(&tensor)?;
            validate_coco_detection_output_schema(&preds, EXPECTED_COCO_PREDICTIONS)?;
            device.synchronize()?;
            let t_forward_end = Instant::now();
            // A full validation copy gates the run but is not part of model-forward latency.
            let preds_host = validate_coco_model_output(&preds)?;
            let model_forward_sync_ms =
                t_forward_end.duration_since(t_forward_start).as_secs_f64() * 1000.0;
            model_forward_sync_times.push(model_forward_sync_ms);

            // Total inference includes both synchronized stage boundaries.
            let total_infer_dur = t_forward_end
                .duration_since(t_preprocess_start)
                .as_secs_f64()
                * 1000.0;
            total_infer_times.push(total_infer_dur);

            // Selected raw-frame pipe read call -> synchronized model result.
            let pipe_to_forward_latency_ms = t_forward_end
                .duration_since(packet.pipe_read_started_at)
                .as_secs_f64()
                * 1000.0;
            latencies.push(pipe_to_forward_latency_ms);

            // Video Writing (Optional)
            if let Some(ref mut writer) = output_stdin {
                use candle_transformers::object_detection::Bbox;
                use manwe::class_aware_non_maximum_suppression;

                let pred = preds_host.i(0)?;
                let (pred_size, npreds) = pred.dims2()?;
                if pred_size != 4 + NUM_CLASSES {
                    anyhow::bail!("video renderer requires the COCO-80 detection schema")
                }
                if npreds > MAX_PREDICTIONS {
                    anyhow::bail!("prediction count exceeds the renderer limit")
                }
                let nclasses = NUM_CLASSES;
                let mut bboxes: Vec<Vec<Bbox<Vec<()>>>> = (0..nclasses).map(|_| vec![]).collect();
                let mut candidate_count = 0_usize;

                for index in 0..npreds {
                    let p_vec = Vec::<f32>::try_from(pred.i((.., index))?)?;
                    if p_vec.iter().any(|value| !value.is_finite()) {
                        anyhow::bail!("model output must contain only finite values")
                    }
                    let Some(confidence) = p_vec[4..].iter().copied().max_by(f32::total_cmp) else {
                        continue;
                    };
                    if confidence > 0.25 {
                        if candidate_count >= MAX_CANDIDATES {
                            anyhow::bail!("detection candidate count exceeds the renderer limit")
                        }
                        let mut class_index = 0;
                        for i in 0..nclasses {
                            if p_vec[4 + i] > p_vec[4 + class_index] {
                                class_index = i;
                            }
                        }
                        let bbox = Bbox {
                            xmin: p_vec[0] - p_vec[2] / 2.,
                            ymin: p_vec[1] - p_vec[3] / 2.,
                            xmax: p_vec[0] + p_vec[2] / 2.,
                            ymax: p_vec[1] + p_vec[3] / 2.,
                            confidence,
                            data: vec![],
                        };
                        bboxes[class_index].push(bbox);
                        candidate_count += 1;
                    }
                }
                class_aware_non_maximum_suppression(&mut bboxes, 0.45)?;

                let mut img_buf = ImageBuffer::<Rgb<u8>, _>::from_raw(
                    INPUT_W as u32,
                    INPUT_H as u32,
                    packet.data.clone(),
                )
                .context("decoded frame had an invalid byte length")?;

                for class_boxes in bboxes.iter() {
                    for b in class_boxes {
                        let xmin = b.xmin.clamp(0.0, (INPUT_W - 1) as f32);
                        let ymin = b.ymin.clamp(0.0, (INPUT_H - 1) as f32);
                        let xmax = b.xmax.clamp(0.0, INPUT_W as f32);
                        let ymax = b.ymax.clamp(0.0, INPUT_H as f32);
                        if xmax <= xmin || ymax <= ymin {
                            continue;
                        }

                        imageproc::drawing::draw_hollow_rect_mut(
                            &mut img_buf,
                            imageproc::rect::Rect::at(xmin as i32, ymin as i32).of_size(
                                ((xmax - xmin).ceil() as u32).max(1),
                                ((ymax - ymin).ceil() as u32).max(1),
                            ),
                            Rgb([255, 0, 0]),
                        );
                    }
                }

                writer.write_all(&img_buf.into_raw())?;
            }

            if frames_processed.is_multiple_of(100) {
                println!("[Rust] Processed {frames_processed} frames...");
            }
        }
        Ok(())
    })();
    if watchdog.finish() {
        processing_result = Err(anyhow::anyhow!(
            "benchmark exceeded the {}-second wall-clock limit",
            args.max_duration_seconds
        ));
    }
    let benchmark_duration_s = start_time.elapsed().as_secs_f64();

    if processing_result.is_err() {
        running.store(false, Ordering::Release);
        input_process.terminate();
    }
    let reader_result = reader_thread
        .join()
        .map_err(|_| anyhow::anyhow!("frame reader thread panicked"))
        .and_then(|result| result.map_err(anyhow::Error::from));
    let pipeline_ok = processing_result.is_ok() && reader_result.is_ok();
    let input_result = if frame_limit_reached.load(Ordering::Acquire) || !pipeline_ok {
        input_process.terminate();
        Ok(())
    } else {
        let status = input_process.wait(Duration::from_secs(10))?;
        if status.success() {
            Ok(())
        } else {
            Err(anyhow::anyhow!("input ffmpeg exited with {status}"))
        }
    };
    drop(output_stdin.take());
    let output_result = if let Some(mut process) = output_process.take() {
        if pipeline_ok && input_result.is_ok() {
            let status = process.wait(Duration::from_secs(30))?;
            if status.success() {
                Ok(())
            } else {
                Err(anyhow::anyhow!("output ffmpeg exited with {status}"))
            }
        } else {
            process.terminate();
            Ok(())
        }
    } else {
        Ok(())
    };

    processing_result?;
    reader_result?;
    input_result?;
    output_result?;

    let final_video_sha256 = sha256_bounded_open_file(
        &mut video_verification_file,
        video_identity,
        &args.video,
        MAX_VIDEO_BYTES,
    )?;
    if final_video_sha256 != video_sha256 {
        anyhow::bail!("video input changed while the benchmark was running")
    }
    ensure_file_identity(&args.video, video_identity, MAX_VIDEO_BYTES)?;

    let authenticated_output_video = match evidence.staged_video_path() {
        Some(path) => Some(digest_and_sync_regular_file(path, MAX_VIDEO_BYTES)?),
        None => None,
    };
    let output_video_sha256 = authenticated_output_video
        .as_ref()
        .map(|authenticated| authenticated.sha256.as_str());

    // Calculate Stats
    if frames_processed == 0 {
        anyhow::bail!("no frames were processed")
    }
    let processed_fps = frames_processed as f64 / benchmark_duration_s;

    let frames_seen = frames_presented.load(Ordering::Relaxed) as f64;
    let drop_rate = if frames_seen > 0.0 {
        1.0 - (frames_processed as f64 / frames_seen)
    } else {
        0.0
    };

    let avg_lat = if latencies.is_empty() {
        0.0
    } else {
        latencies.iter().sum::<f64>() / latencies.len() as f64
    };
    let avg_frame_read_wait = if frame_read_wait_times.is_empty() {
        0.0
    } else {
        frame_read_wait_times.iter().sum::<f64>() / frame_read_wait_times.len() as f64
    };
    let avg_infer = if total_infer_times.is_empty() {
        0.0
    } else {
        total_infer_times.iter().sum::<f64>() / total_infer_times.len() as f64
    };
    let avg_preprocess_upload_sync = if preprocess_upload_sync_times.is_empty() {
        0.0
    } else {
        preprocess_upload_sync_times.iter().sum::<f64>() / preprocess_upload_sync_times.len() as f64
    };
    let avg_model_forward_sync = if model_forward_sync_times.is_empty() {
        0.0
    } else {
        model_forward_sync_times.iter().sum::<f64>() / model_forward_sync_times.len() as f64
    };

    latencies.sort_by(f64::total_cmp);
    let p99 = percentile(&latencies, 99.0);

    let results = serde_json::json!({
        "model": "rust_candle",
        "model_sha256": model_sha256,
        "video": args.video,
        "video_sha256": video_sha256,
        "input_demuxer": input_demuxer,
        "ffmpeg_path": ffmpeg.path(),
        "ffmpeg_sha256": ffmpeg.sha256(),
        "device": "Metal GPU",
        "target_fps": args.target_fps,
        "run_id": args.run_id,
        "processed_fps": processed_fps,
        "benchmark_duration_s": benchmark_duration_s,
        "drop_rate": drop_rate,
        "raw_pipe_to_forward_avg_latency_ms": avg_lat,
        "raw_pipe_to_forward_p99_latency_ms": p99,
        "processed_frame_read_wait_avg_ms": avg_frame_read_wait,
        "inference_avg_ms": avg_infer,
        "preprocess_upload_sync_avg_ms": avg_preprocess_upload_sync,
        "model_forward_sync_avg_ms": avg_model_forward_sync,
        "frames_presented": frames_presented.load(Ordering::Relaxed),
        "frames_processed": frames_processed,
        "max_frames": args.max_frames,
        "max_duration_seconds": args.max_duration_seconds,
        "frame_limit_reached": frame_limit_reached.load(Ordering::Relaxed),
        "inference_scope": "preprocess/upload through model-forward completion with transfer-free device synchronization after each stage; metadata-only fixed [1,84,8400] COCO output schema validation included, full-output device compaction, CPU readback, and finite-value validation excluded",
        "latency_scope": "selected raw-frame pipe read call start through synchronized model-forward completion; source capture, full-output device compaction, CPU readback, finite-value validation, and NMS/rendering/encoding excluded",
        "processed_fps_scope": "reader-start to final processed frame; includes producer pacing, frame drops, full-output device compaction, CPU readback, finite-value validation, and optional rendering/encoder-pipe writes but excludes encoder finalization",
        "producer_pacing": "reader thread paced at target_fps; 0 means unthrottled",
        "save_video": args.save_video,
        "output_video": evidence.output_path(),
        "output_video_sha256": output_video_sha256,
    });

    evidence.publish(&results, authenticated_output_video.as_ref())?;

    println!(
        "[Rust Candle] FPS: {processed_fps:.2} | Infer: {avg_infer:.1}ms (preprocess/upload+sync: {avg_preprocess_upload_sync:.1}ms, forward+sync: {avg_model_forward_sync:.1}ms) | Pipe->forward: {avg_lat:.1}ms avg | Processed-frame read wait: {avg_frame_read_wait:.1}ms"
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn evidence_test_directory(label: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "manwe-video-evidence-{label}-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ))
    }

    fn blocking_child() -> ChildGuard {
        let child = Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "tests::child_guard_blocking_helper"])
            .env_clear()
            .env("MANWE_BENCHMARK_CHILD_GUARD_TEST", "1")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .unwrap();
        ChildGuard::new(child)
    }

    #[test]
    fn fixed_video_input_has_the_expected_yolov8_prediction_count() {
        assert_eq!(EXPECTED_COCO_PREDICTIONS, 8_400);
    }

    #[test]
    fn fps_parser_rejects_values_that_can_overflow_duration() {
        assert!(bounded_fps("1e-300").is_err());
        assert_eq!(bounded_fps("0").unwrap(), 0.0);
        assert_eq!(bounded_fps("30").unwrap(), 30.0);
    }

    #[test]
    fn frame_limit_parser_bounds_total_video_work() {
        assert!(bounded_frame_count("0").is_err());
        assert_eq!(bounded_frame_count("1000000").unwrap(), MAX_VIDEO_FRAMES);
        assert!(bounded_frame_count("1000001").is_err());
        assert!(bounded_duration_seconds("0").is_err());
        assert!(bounded_duration_seconds("7200").is_ok());
    }

    #[test]
    fn raw_frame_reader_distinguishes_clean_eof_from_truncation() {
        let mut frame = [0_u8; 4];
        assert!(!read_complete_frame(&mut Cursor::new([]), &mut frame).unwrap());
        assert!(read_complete_frame(&mut Cursor::new([1, 2, 3, 4]), &mut frame).unwrap());
        let error = read_complete_frame(&mut Cursor::new([1, 2, 3]), &mut frame).unwrap_err();
        assert_eq!(error.kind(), std::io::ErrorKind::UnexpectedEof);
    }

    #[test]
    fn percentile_uses_the_same_indexing_contract_as_static_benchmarks() {
        let samples = (1..=100).map(f64::from).collect::<Vec<_>>();
        assert_eq!(percentile(&samples, 99.0), 99.0);
    }

    #[test]
    fn video_demuxer_rejects_reference_playlist_formats() {
        assert_eq!(video_demuxer(Path::new("input.mp4")).unwrap(), "mov");
        assert_eq!(video_demuxer(Path::new("input.webm")).unwrap(), "matroska");
        assert!(video_demuxer(Path::new("input.m3u8")).is_err());
        assert!(video_demuxer(Path::new("input.ffconcat")).is_err());
    }

    #[test]
    fn evidence_run_reserves_and_publishes_without_replacement() {
        let directory = evidence_test_directory("publish");
        let _ = fs::remove_dir_all(&directory);
        fs::create_dir(&directory).unwrap();
        let mut run = EvidenceRun::acquire(&directory, "input_mp4", "run_1", false).unwrap();

        assert!(EvidenceRun::acquire(&directory, "input_mp4", "run_1", false).is_err());
        run.publish(&serde_json::json!({"complete": true}), None)
            .unwrap();

        let result = directory.join("res_rust_input_mp4_run_1.json");
        assert!(result.is_file());
        assert!(EvidenceRun::acquire(&directory, "input_mp4", "run_1", false).is_err());
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn evidence_drop_preserves_a_swapped_final_and_the_stage_marker() {
        let directory = evidence_test_directory("swapped-final");
        let _ = fs::remove_dir_all(&directory);
        fs::create_dir(&directory).unwrap();
        let mut run = EvidenceRun::acquire(&directory, "input_mp4", "run_2", false).unwrap();
        let expected =
            write_verified_json_once(&run.stage_result, &serde_json::json!({"original": true}))
                .unwrap();
        fs::hard_link(&run.stage_result, &run.result_path).unwrap();
        verify_json_file(&run.stage_result, &expected).unwrap();
        run.final_link_created = true;
        fs::remove_file(&run.result_path).unwrap();
        fs::write(&run.result_path, b"replacement-evidence").unwrap();
        let stage_dir = run.stage_dir.path().to_path_buf();
        let result_path = run.result_path.clone();

        drop(run);

        assert_eq!(fs::read(result_path).unwrap(), b"replacement-evidence");
        assert!(stage_dir.is_dir());
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn evidence_publication_fails_closed_when_run_directory_is_replaced() {
        let directory = evidence_test_directory("directory-replacement");
        let moved = directory.with_extension("moved");
        let _ = fs::remove_dir_all(&directory);
        let _ = fs::remove_dir_all(&moved);
        fs::create_dir(&directory).unwrap();
        let mut run = EvidenceRun::acquire(&directory, "input_mp4", "run_3", false).unwrap();
        fs::rename(&directory, &moved).unwrap();
        fs::create_dir(&directory).unwrap();
        let replacement = directory.join("res_rust_input_mp4_run_3.json");
        fs::write(&replacement, b"replacement-directory").unwrap();

        assert!(run
            .publish(&serde_json::json!({"original": true}), None)
            .is_err());
        drop(run);

        assert_eq!(fs::read(&replacement).unwrap(), b"replacement-directory");
        assert!(moved
            .join(".manwe-benchmark-input_mp4-run_3.in-progress")
            .is_dir());
        fs::remove_dir_all(directory).unwrap();
        fs::remove_dir_all(moved).unwrap();
    }

    #[test]
    fn video_link_is_preserved_when_result_publication_later_fails() {
        let directory = evidence_test_directory("partial-publication");
        let _ = fs::remove_dir_all(&directory);
        fs::create_dir(&directory).unwrap();
        let mut run = EvidenceRun::acquire(&directory, "input_mp4", "run_4", true).unwrap();
        let staged_video = run.staged_video_path().unwrap().to_path_buf();
        fs::write(&staged_video, b"video-evidence").unwrap();
        let authenticated = digest_and_sync_regular_file(&staged_video, MAX_VIDEO_BYTES).unwrap();
        fs::write(&run.result_path, b"occupied-result").unwrap();
        let video_path = run.output_path().unwrap().to_path_buf();
        let stage_dir = run.stage_dir.path().to_path_buf();

        assert!(run
            .publish(
                &serde_json::json!({"complete": false}),
                Some(&authenticated),
            )
            .is_err());
        drop(run);

        assert_eq!(fs::read(video_path).unwrap(), b"video-evidence");
        assert_eq!(
            fs::read(directory.join("res_rust_input_mp4_run_4.json")).unwrap(),
            b"occupied-result"
        );
        assert!(stage_dir.is_dir());
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn authenticated_staged_video_replacement_is_rejected_before_publication() {
        let directory = evidence_test_directory("staged-video-swap");
        let _ = fs::remove_dir_all(&directory);
        fs::create_dir(&directory).unwrap();
        let mut run = EvidenceRun::acquire(&directory, "input_mp4", "run_5", true).unwrap();
        let staged_video = run.staged_video_path().unwrap().to_path_buf();
        fs::write(&staged_video, b"authenticated-video").unwrap();
        let authenticated = digest_and_sync_regular_file(&staged_video, MAX_VIDEO_BYTES).unwrap();
        fs::remove_file(&staged_video).unwrap();
        fs::write(&staged_video, b"replacement-video").unwrap();
        let final_video = run.output_path().unwrap().to_path_buf();
        let final_result = run.result_path.clone();
        let stage_dir = run.stage_dir.path().to_path_buf();

        assert!(run
            .publish(
                &serde_json::json!({"output_video_sha256": authenticated.sha256.as_str()}),
                Some(&authenticated),
            )
            .is_err());
        drop(run);

        assert!(!final_video.exists());
        assert!(!final_result.exists());
        assert!(!stage_dir.exists());
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn pipeline_deadline_terminates_a_blocked_child() {
        let child = blocking_child();
        let running = Arc::new(AtomicBool::new(true));
        let watchdog = PipelineDeadlineWatchdog::spawn(
            Instant::now(),
            Arc::clone(&running),
            child.terminator(),
            None,
        );

        assert!(watchdog.finish());
        assert!(!running.load(Ordering::Acquire));
        assert!(child.child.lock().unwrap().is_none());
        drop(child);
    }

    #[test]
    fn pipeline_completion_before_deadline_cancels_the_watchdog() {
        let child = blocking_child();
        let running = Arc::new(AtomicBool::new(true));
        let watchdog = PipelineDeadlineWatchdog::spawn(
            Instant::now() + Duration::from_secs(MAX_BENCHMARK_DURATION_SECONDS),
            Arc::clone(&running),
            child.terminator(),
            None,
        );

        assert!(!watchdog.finish());
        assert!(running.load(Ordering::Acquire));
        assert!(child.child.lock().unwrap().is_some());
        drop(child);
    }

    #[test]
    fn pipeline_cancellation_wins_after_worker_observes_pending() {
        let state = PipelineDeadlineState::new();

        // Force the former race ordering without depending on thread scheduling:
        // worker observes pending, finish cancels, then worker claims timeout.
        let worker_observed_pending = state.is_pending();
        let cancellation_claimed = state.cancel();
        let timeout_claimed = state.claim_timeout();

        assert_eq!(
            (
                worker_observed_pending,
                cancellation_claimed,
                timeout_claimed,
                state.timed_out(),
            ),
            (true, true, false, false)
        );
    }

    #[test]
    fn child_guard_blocking_helper() {
        if std::env::var_os("MANWE_BENCHMARK_CHILD_GUARD_TEST").is_some() {
            loop {
                thread::sleep(Duration::from_secs(60));
            }
        }
    }
}
