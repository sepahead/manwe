//! YOLOv8s Performance Benchmark - Rust/Candle with Metal GPU
//!
//! This benchmark measures inference performance of YOLOv8s object detection
//! using the Candle framework with Metal GPU acceleration on Apple Silicon.
//!
//! # Precision
//! All computations use FP32 (single precision) to ensure fair comparison
//! with PyTorch MPS and CoreML implementations.
//!
//! # Metrics
//! - FPS (frames per second)
//! - Latency (milliseconds per frame)
//! - P1 latency (1st percentile - fastest frames)
//! - P99 latency (99th percentile - slowest frames / potential dropped frames)

use std::env;
use std::fs::{self, OpenOptions};
use std::io::{Cursor, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::Context;
use candle::{DType, Device, IndexOp, Tensor};
use candle_nn::{Module, VarBuilder};
use clap::Parser;
use manwe::model::{Multiples, YoloV8};
use manwe::prepare_image;
use manwe::secure_io::{
    open_bounded_regular_file, read_bounded_open_file, read_bounded_regular_file,
    read_bounded_regular_file_with_identity, BoundDirectory, FileIdentity, MAX_ENCODED_IMAGE_BYTES,
    MAX_MODEL_BYTES,
};
use serde_json::json;
use sha2::{Digest, Sha256};

/// Benchmark configuration.
const WARMUP_ITERATIONS: usize = 10;
const NUM_COCO_CLASSES: usize = 80;
const MAX_DIRECTORY_ENTRIES: usize = 100_000;
const MAX_RESULT_BYTES: usize = 64 * 1024 * 1024;

fn positive_count(value: &str) -> std::result::Result<usize, String> {
    let parsed = value
        .parse::<usize>()
        .map_err(|_| format!("{value:?} is not a positive integer"))?;
    if !(1..=10_000).contains(&parsed) {
        Err("value must be between 1 and 10000".to_string())
    } else {
        Ok(parsed)
    }
}

fn safe_run_id(value: &str) -> std::result::Result<String, String> {
    if value.len() <= 64
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

#[derive(Parser, Debug)]
#[command(name = "YOLOv8s Benchmark")]
#[command(about = "Benchmark YOLOv8s inference on Metal GPU using Candle")]
struct Args {
    /// Number of images to process
    #[arg(short, long, default_value_t = 500, value_parser = positive_count)]
    num_images: usize,

    /// Run identifier for result files
    #[arg(short, long, default_value = "", value_parser = safe_run_id)]
    run_id: String,

    /// Directory containing benchmark images.
    #[arg(long, default_value = "assets/kaggle1/train/images")]
    image_dir: PathBuf,

    /// Local, non-symlinked YOLOv8s safetensors weights.
    #[arg(long)]
    model: PathBuf,

    /// Expected SHA-256 for the exact model artifact.
    #[arg(long, value_parser = sha256_digest)]
    model_sha256: String,
}

#[derive(Debug)]
struct SelectedImage {
    path: PathBuf,
    sha256: String,
    identity: FileIdentity,
}

fn load_selected_image(selected: &SelectedImage) -> anyhow::Result<Tensor> {
    let (bytes, identity) =
        read_bounded_regular_file_with_identity(&selected.path, MAX_ENCODED_IMAGE_BYTES)?;
    if identity != selected.identity {
        anyhow::bail!(
            "benchmark input identity changed during the run: {}",
            selected.path.display()
        )
    }
    let digest = format!("{:x}", Sha256::digest(&bytes));
    if digest != selected.sha256 {
        anyhow::bail!(
            "benchmark input changed during the run: {}",
            selected.path.display()
        )
    }
    let original = decode_image_bytes(&bytes)?;
    let (tensor, _) = prepare_image(&original, 640, 32, &Device::Cpu)?;
    Ok(tensor)
}

fn decode_image_bytes(bytes: &[u8]) -> anyhow::Result<image::DynamicImage> {
    let mut reader = image::ImageReader::new(Cursor::new(bytes)).with_guessed_format()?;
    let mut limits = image::Limits::default();
    limits.max_image_width = Some(32_768);
    limits.max_image_height = Some(32_768);
    limits.max_alloc = Some(256 * 1024 * 1024);
    reader.limits(limits);
    Ok(reader.decode()?)
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

fn path_occupied(path: &Path) -> anyhow::Result<bool> {
    match fs::symlink_metadata(path) {
        Ok(_) => Ok(true),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(error.into()),
    }
}

fn verify_json_file(path: &Path, expected: &[u8]) -> anyhow::Result<FileIdentity> {
    if expected.is_empty() || expected.len() > MAX_RESULT_BYTES {
        anyhow::bail!("result JSON must contain between 1 and {MAX_RESULT_BYTES} bytes")
    }
    let max_bytes = u64::try_from(MAX_RESULT_BYTES).context("result size limit overflowed")?;
    let (mut file, identity) = open_bounded_regular_file(path, max_bytes)?;
    let actual = read_bounded_open_file(&mut file, identity, path, max_bytes)?;
    if actual != expected {
        anyhow::bail!("result JSON verification failed: {}", path.display())
    }
    serde_json::from_slice::<serde_json::Value>(&actual)
        .with_context(|| format!("staged result is not valid JSON: {}", path.display()))?;
    Ok(identity)
}

fn write_verified_json_once(path: &Path, value: &serde_json::Value) -> anyhow::Result<Vec<u8>> {
    let bytes = serde_json::to_vec_pretty(value)?;
    if bytes.len() > MAX_RESULT_BYTES {
        anyhow::bail!("result JSON exceeds the {MAX_RESULT_BYTES}-byte limit")
    }
    let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
    file.write_all(&bytes)?;
    file.sync_all()?;
    drop(file);
    verify_json_file(path, &bytes)?;
    Ok(bytes)
}

struct EvidenceRun {
    run_dir: BoundDirectory,
    stage_dir: BoundDirectory,
    stage_result: PathBuf,
    result_path: PathBuf,
    final_link_created: bool,
    committed: bool,
}

impl EvidenceRun {
    fn acquire(run_dir: &Path, run_id: &str) -> anyhow::Result<Self> {
        let run_dir = BoundDirectory::open(run_dir)?;
        let result_path = run_dir.path().join(format!("results_rust_{run_id}.json"));
        let stage_path = run_dir
            .path()
            .join(format!(".manwe-static-benchmark-{run_id}.in-progress"));
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
            run_dir,
            stage_dir,
            result_path,
            final_link_created: false,
            committed: false,
        };
        run.run_dir.sync()?;
        run.stage_dir.verify()?;
        if path_occupied(&run.result_path)? {
            anyhow::bail!("run id would overwrite existing benchmark evidence")
        }
        Ok(run)
    }

    fn result_path(&self) -> &Path {
        &self.result_path
    }

    fn publish(&mut self, results: &serde_json::Value) -> anyhow::Result<()> {
        let expected = write_verified_json_once(&self.stage_result, results)?;
        self.stage_dir.sync()?;
        self.run_dir.verify()?;
        fs::hard_link(&self.stage_result, &self.result_path)?;
        self.final_link_created = true;
        let stage_identity = verify_json_file(&self.stage_result, &expected)?;
        self.run_dir.verify()?;
        self.stage_dir.verify()?;
        let result_identity = verify_json_file(&self.result_path, &expected)?;
        if result_identity != stage_identity {
            anyhow::bail!("published result identity does not match the staged result")
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
                    eprintln!("result was published but staging cleanup sync failed: {error}");
                }
            }
            Err(error) => {
                eprintln!("result was published but staging cleanup failed: {error}");
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

/// Calculate percentile from sorted latency array.
fn percentile(sorted_latencies: &[f64], p: f64) -> f64 {
    if sorted_latencies.is_empty() {
        return 0.0;
    }
    let idx = ((p / 100.0) * (sorted_latencies.len() - 1) as f64).round() as usize;
    sorted_latencies[idx.min(sorted_latencies.len() - 1)]
}

fn timing_summary(latencies_ms: &[f64]) -> anyhow::Result<(f64, f64, f64)> {
    if latencies_ms.is_empty()
        || latencies_ms
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
    {
        anyhow::bail!("latency samples must be finite and positive")
    }
    let total_ms = latencies_ms.iter().sum::<f64>();
    let average_ms = total_ms / latencies_ms.len() as f64;
    Ok((total_ms / 1000.0, 1000.0 / average_ms, average_ms))
}

/// Force GPU synchronization by reading a value back to CPU.
fn sync_metal(tensor: &Tensor) -> anyhow::Result<()> {
    // Pull a single scalar instead of the whole tensor to avoid large GPU->CPU copies.
    let _ = tensor.flatten_all()?.i(0)?.to_scalar::<f32>()?;
    Ok(())
}

fn run_dir() -> anyhow::Result<PathBuf> {
    let dir = env::var("RUN_DIR").unwrap_or_else(|_| ".".to_string());
    let path = PathBuf::from(dir);
    fs::create_dir_all(&path)?;
    Ok(path)
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let run_id = if args.run_id.is_empty() {
        format!("{}", std::process::id())
    } else {
        args.run_id
    };
    let base_dir = run_dir()?;
    let mut evidence = EvidenceRun::acquire(&base_dir, &run_id)?;

    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║       YOLOv8s Benchmark - Rust/Candle (Metal GPU)        ║");
    println!("╠══════════════════════════════════════════════════════════╣");
    println!("║ Precision: FP32                                          ║");
    println!("║ Model: YOLOv8s (~11M params, 21MB weights)               ║");
    println!("╚══════════════════════════════════════════════════════════╝");
    println!();
    println!("Configuration:");
    println!("  Images: {}", args.num_images);
    println!("  Run ID: {}", run_id);
    println!("  Warmup: {} iterations", WARMUP_ITERATIONS);
    println!();

    // This crate produces Metal evidence; changing backend invalidates the run.
    let device = Device::new_metal(0)?;
    println!("Device: Metal GPU (Apple Silicon)");

    // Load YOLOv8s model.
    println!("Loading YOLOv8s model...");
    let model_path = args.model;
    if model_path.extension().and_then(|ext| ext.to_str()) != Some("safetensors") {
        anyhow::bail!("model must have a .safetensors extension")
    }
    let model_bytes = read_bounded_regular_file(&model_path, MAX_MODEL_BYTES)?;
    let model_sha256 = format!("{:x}", Sha256::digest(&model_bytes));
    if model_sha256 != args.model_sha256 {
        anyhow::bail!("model SHA-256 does not match the expected digest")
    }
    let vb = VarBuilder::from_buffered_safetensors(model_bytes, DType::F32, &device)?;
    let model = YoloV8::load(vb, Multiples::s(), NUM_COCO_CLASSES)?;
    println!("Model loaded successfully.");
    println!();

    // Load and preprocess images
    let image_dir = args.image_dir;
    println!("Loading images from {:?}...", image_dir);

    let mut images: Vec<SelectedImage> = Vec::new();
    let all_entries = fs::read_dir(&image_dir)?
        .take(MAX_DIRECTORY_ENTRIES + 1)
        .collect::<std::io::Result<Vec<_>>>()?;
    if all_entries.len() > MAX_DIRECTORY_ENTRIES {
        anyhow::bail!("image directory contains more than {MAX_DIRECTORY_ENTRIES} entries")
    }
    let mut entries: Vec<_> = all_entries
        .into_iter()
        .filter(|e| {
            e.path()
                .extension()
                .map(|ext| {
                    let ext = ext.to_string_lossy().to_lowercase();
                    ext == "jpg" || ext == "jpeg" || ext == "png"
                })
                .unwrap_or(false)
        })
        .collect();
    entries.sort_by_key(std::fs::DirEntry::path);
    for entry in entries {
        if images.len() == args.num_images {
            break;
        }
        let path = entry.path();
        let (bytes, identity) =
            match read_bounded_regular_file_with_identity(&path, MAX_ENCODED_IMAGE_BYTES) {
                Ok(value) => value,
                Err(error) => {
                    eprintln!("skipping {}: {error}", path.display());
                    continue;
                }
            };
        if let Err(error) = decode_image_bytes(&bytes) {
            eprintln!("skipping {}: {error}", path.display());
            continue;
        }
        images.push(SelectedImage {
            path,
            sha256: format!("{:x}", Sha256::digest(&bytes)),
            identity,
        });
    }

    println!("Loaded {} images.", images.len());

    if images.len() != args.num_images {
        anyhow::bail!(
            "requested {} valid images but found {} in {}",
            args.num_images,
            images.len(),
            image_dir.display()
        )
    }
    println!();

    // Warmup phase - compile Metal shaders and warm caches
    println!("Warming up ({} iterations)...", WARMUP_ITERATIONS);
    let first_img = load_selected_image(&images[0])?;
    for _ in 0..WARMUP_ITERATIONS {
        let img_dev = first_img.to_device(&device)?;
        let pred = model.forward(&img_dev)?;
        sync_metal(&pred)?;
    }
    println!("Warmup complete.");
    println!();

    // Benchmark phase - measure per-frame latency
    println!("Running benchmark...");
    let mut latencies_ms: Vec<f64> = Vec::with_capacity(images.len());

    for selected in &images {
        let img = load_selected_image(selected)?;
        let frame_start = Instant::now();

        let img_dev = img.to_device(&device)?;
        let pred = model.forward(&img_dev)?;
        sync_metal(&pred)?;

        let frame_time = frame_start.elapsed();
        latencies_ms.push(frame_time.as_secs_f64() * 1000.0);
    }

    // Calculate statistics
    let num_images = images.len();
    let (total_time_s, fps, avg_latency) = timing_summary(&latencies_ms)?;

    // Sort for percentile calculations
    latencies_ms.sort_by(f64::total_cmp);
    let p1_latency = percentile(&latencies_ms, 1.0);
    let p50_latency = percentile(&latencies_ms, 50.0);
    let p99_latency = percentile(&latencies_ms, 99.0);
    let min_latency = latencies_ms.first().copied().unwrap_or(0.0);
    let max_latency = latencies_ms.last().copied().unwrap_or(0.0);

    // Print results
    println!();
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║                        RESULTS                           ║");
    println!("╠══════════════════════════════════════════════════════════╣");
    println!(
        "║ Images processed: {:>6}                                 ║",
        num_images
    );
    println!(
        "║ Total time:       {:>6.2} s                               ║",
        total_time_s
    );
    println!(
        "║ Average FPS:      {:>6.2}                                 ║",
        fps
    );
    println!(
        "║ Avg latency:      {:>6.2} ms                              ║",
        avg_latency
    );
    println!("╠══════════════════════════════════════════════════════════╣");
    println!("║ Latency Percentiles:                                     ║");
    println!(
        "║   Min (best):     {:>6.2} ms                              ║",
        min_latency
    );
    println!(
        "║   P1:             {:>6.2} ms                              ║",
        p1_latency
    );
    println!(
        "║   P50 (median):   {:>6.2} ms                              ║",
        p50_latency
    );
    println!(
        "║   P99:            {:>6.2} ms                              ║",
        p99_latency
    );
    println!(
        "║   Max (worst):    {:>6.2} ms                              ║",
        max_latency
    );
    println!("╚══════════════════════════════════════════════════════════╝");

    // Save results to JSON
    let input_manifest = images
        .iter()
        .map(|image| {
            json!({
                "path": image.path.to_string_lossy(),
                "sha256": image.sha256,
            })
        })
        .collect::<Vec<_>>();

    let results = json!({
        "model": "YOLOv8s",
        "implementation": "Rust (Candle 0.9.2)",
        "device": "Metal GPU",
        "precision": "FP32",
        "model_sha256": model_sha256,
        "timing_scope": "host-to-device upload + model forward + synchronization",
        "excluded_from_timing": ["image decode", "resize/letterbox", "normalization", "NMS/postprocess", "rendering"],
        "input_selection": "lexicographically sorted file names, first N valid images",
        "input_manifest": input_manifest,
        "image_dir": image_dir,
        "images": num_images,
        "total_time_s": total_time_s,
        "fps": fps,
        "latency_ms": {
            "average": avg_latency,
            "min": min_latency,
            "p1": p1_latency,
            "p50": p50_latency,
            "p99": p99_latency,
            "max": max_latency
        },
        "run_id": run_id,
        "config": {
            "warmup_iterations": WARMUP_ITERATIONS
        }
    });

    evidence.publish(&results)?;
    println!("\nResults saved to: {}", evidence.result_path().display());

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::process::{Command, Stdio};

    const HARD_EXIT_DIRECTORY: &str = "MANWE_STATIC_EVIDENCE_HARD_EXIT_DIRECTORY";

    fn evidence_test_directory(label: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "manwe-static-evidence-{label}-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ))
    }

    #[test]
    fn benchmark_parsers_bound_work_and_artifact_identity() {
        assert!(positive_count("0").is_err());
        assert!(positive_count("10001").is_err());
        assert!(safe_run_id("../escape").is_err());
        assert!(sha256_digest(&"f".repeat(64)).is_ok());
        assert!(sha256_digest(&"f".repeat(65)).is_err());
    }

    #[test]
    fn timing_summary_uses_the_same_samples_for_mean_and_fps() {
        let (total_seconds, fps, average_ms) = timing_summary(&[2.0, 4.0]).unwrap();

        assert!((total_seconds - 0.006).abs() < f64::EPSILON);
        assert!((average_ms - 3.0).abs() < f64::EPSILON);
        assert!((fps - (1000.0 / 3.0)).abs() < 1e-10);
    }

    #[test]
    fn evidence_run_publishes_verified_json_and_removes_staging() {
        let directory = evidence_test_directory("publish");
        let _ = fs::remove_dir_all(&directory);
        fs::create_dir(&directory).unwrap();
        let mut run = EvidenceRun::acquire(&directory, "run_1").unwrap();
        let stage_dir = run.stage_dir.path().to_path_buf();

        run.publish(&serde_json::json!({"complete": true})).unwrap();

        let result: serde_json::Value =
            serde_json::from_slice(&fs::read(run.result_path()).unwrap()).unwrap();
        assert_eq!(result, serde_json::json!({"complete": true}));
        assert!(!stage_dir.exists());
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn evidence_run_does_not_replace_output_created_after_reservation() {
        let directory = evidence_test_directory("occupied");
        let _ = fs::remove_dir_all(&directory);
        fs::create_dir(&directory).unwrap();
        let mut run = EvidenceRun::acquire(&directory, "run_2").unwrap();
        let result_path = run.result_path().to_path_buf();
        fs::write(&result_path, b"existing-evidence").unwrap();

        assert!(run
            .publish(&serde_json::json!({"replacement": true}))
            .is_err());
        drop(run);

        assert_eq!(fs::read(&result_path).unwrap(), b"existing-evidence");
        assert!(!directory
            .join(".manwe-static-benchmark-run_2.in-progress")
            .exists());
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn evidence_run_cleans_staging_on_ordinary_failure() {
        let directory = evidence_test_directory("ordinary-failure");
        let _ = fs::remove_dir_all(&directory);
        fs::create_dir(&directory).unwrap();
        let run = EvidenceRun::acquire(&directory, "run_3").unwrap();
        let stage_dir = run.stage_dir.path().to_path_buf();

        drop(run);

        assert!(!stage_dir.exists());
        assert!(!directory.join("results_rust_run_3.json").exists());
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn evidence_cleanup_preserves_a_replacement_at_the_final_path() {
        let directory = evidence_test_directory("result-replacement");
        let _ = fs::remove_dir_all(&directory);
        fs::create_dir(&directory).unwrap();
        let mut run = EvidenceRun::acquire(&directory, "run_4").unwrap();
        let expected =
            write_verified_json_once(&run.stage_result, &serde_json::json!({"original": true}))
                .unwrap();
        fs::hard_link(&run.stage_result, &run.result_path).unwrap();
        verify_json_file(&run.stage_result, &expected).unwrap();
        run.final_link_created = true;
        fs::remove_file(&run.result_path).unwrap();
        fs::write(&run.result_path, b"replacement-evidence").unwrap();

        drop(run);

        assert_eq!(
            fs::read(directory.join("results_rust_run_4.json")).unwrap(),
            b"replacement-evidence"
        );
        assert!(directory
            .join(".manwe-static-benchmark-run_4.in-progress")
            .is_dir());
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn evidence_failure_preserves_its_exact_uncommitted_link_and_marker() {
        let directory = evidence_test_directory("exact-cleanup");
        let _ = fs::remove_dir_all(&directory);
        fs::create_dir(&directory).unwrap();
        let mut run = EvidenceRun::acquire(&directory, "run_exact").unwrap();
        let expected =
            write_verified_json_once(&run.stage_result, &serde_json::json!({"exact": true}))
                .unwrap();
        fs::hard_link(&run.stage_result, &run.result_path).unwrap();
        verify_json_file(&run.stage_result, &expected).unwrap();
        run.final_link_created = true;
        let stage_dir = run.stage_dir.path().to_path_buf();
        let result_path = run.result_path.clone();

        drop(run);

        assert!(result_path.is_file());
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
        let mut run = EvidenceRun::acquire(&directory, "run_5").unwrap();
        fs::rename(&directory, &moved).unwrap();
        fs::create_dir(&directory).unwrap();
        let replacement = directory.join("results_rust_run_5.json");
        fs::write(&replacement, b"replacement-directory").unwrap();

        assert!(run.publish(&serde_json::json!({"original": true})).is_err());
        drop(run);

        assert_eq!(fs::read(&replacement).unwrap(), b"replacement-directory");
        fs::remove_dir_all(directory).unwrap();
        fs::remove_dir_all(moved).unwrap();
    }

    #[test]
    fn evidence_run_leaves_an_in_progress_marker_after_hard_exit() {
        let directory = evidence_test_directory("hard-exit");
        let _ = fs::remove_dir_all(&directory);
        fs::create_dir(&directory).unwrap();
        let status = Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "tests::evidence_run_hard_exit_helper"])
            .env_clear()
            .env(HARD_EXIT_DIRECTORY, &directory)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .unwrap();

        assert_eq!(status.code(), Some(73));
        assert!(directory
            .join(".manwe-static-benchmark-hard_exit.in-progress")
            .is_dir());
        assert!(!directory.join("results_rust_hard_exit.json").exists());
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn evidence_run_hard_exit_helper() {
        let Some(directory) = std::env::var_os(HARD_EXIT_DIRECTORY) else {
            return;
        };
        let directory = PathBuf::from(directory);
        let _run = EvidenceRun::acquire(&directory, "hard_exit").unwrap();
        std::process::exit(73);
    }
}
