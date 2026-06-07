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

mod model;

use candle::{DType, Device, Tensor, IndexOp};
use candle_nn::{Module, VarBuilder};
use clap::Parser;
use model::{Multiples, YoloV8};
use serde_json::json;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;
use std::env;
use image::{DynamicImage, RgbImage};

/// Benchmark configuration.
const CONFIDENCE_THRESHOLD: f32 = 0.25;
const NMS_THRESHOLD: f32 = 0.45;
const WARMUP_ITERATIONS: usize = 10;
const NUM_COCO_CLASSES: usize = 80;

#[derive(Parser, Debug)]
#[command(name = "YOLOv8s Benchmark")]
#[command(about = "Benchmark YOLOv8s inference on Metal GPU using Candle")]
struct Args {
    /// Number of images to process
    #[arg(short, long, default_value_t = 500)]
    num_images: usize,

    /// Run identifier for result files
    #[arg(short, long, default_value = "")]
    run_id: String,
}

/// Calculate percentile from sorted latency array.
fn percentile(sorted_latencies: &[f64], p: f64) -> f64 {
    if sorted_latencies.is_empty() {
        return 0.0;
    }
    let idx = ((p / 100.0) * (sorted_latencies.len() - 1) as f64).round() as usize;
    sorted_latencies[idx.min(sorted_latencies.len() - 1)]
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

fn letterbox_to_square(img: &DynamicImage, size: u32) -> RgbImage {
    let (w, h) = (img.width(), img.height());
    let (new_w, new_h) = if w > h {
        (size, (h * size / w) / 32 * 32)
    } else {
        ((w * size / h) / 32 * 32, size)
    };
    let resized = img.resize_exact(new_w, new_h, image::imageops::FilterType::CatmullRom);
    let mut canvas = RgbImage::new(size, size);
    let x_off = ((size - new_w) / 2) as i64;
    let y_off = ((size - new_h) / 2) as i64;
    image::imageops::overlay(&mut canvas, &resized.to_rgb8(), x_off, y_off);
    canvas
}

fn main() -> anyhow::Result<()> {
    // Keep panics visible for debugging.
    std::panic::set_hook(Box::new(|info| eprintln!("panic: {info}")));

    let args = Args::parse();
    let run_id = if args.run_id.is_empty() {
        format!("{}", std::process::id())
    } else {
        args.run_id.clone()
    };

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

    // Initialize Device (Metal -> CUDA -> CPU)
    let device = if candle::utils::metal_is_available() {
        println!("Device: Metal GPU (Apple Silicon)");
        Device::new_metal(0)?
    } else if candle::utils::cuda_is_available() {
        println!("Device: CUDA GPU");
        Device::new_cuda(0)?
    } else {
        println!("Device: CPU (Fallback)");
        Device::Cpu
    };

    // Create output directory
    fs::create_dir_all("output")?;

    // Load YOLOv8s model from HuggingFace
    println!("Loading YOLOv8s model...");
    let api = hf_hub::api::sync::Api::new()?;
    let api_repo = api.model("lmz/candle-yolo-v8".to_string());
    let model_path = api_repo.get("yolov8s.safetensors")?;

    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[&model_path], DType::F32, &device)? };
    let model = YoloV8::load(vb, Multiples::s(), NUM_COCO_CLASSES)?;
    println!("Model loaded successfully.");
    println!();

    // Load and preprocess images
    let image_dir = PathBuf::from("assets/kaggle1/train/images");
    println!("Loading images from {:?}...", image_dir);

    let mut images: Vec<Tensor> = Vec::new();
    let entries: Vec<_> = fs::read_dir(&image_dir)?
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path()
                .extension()
                .map(|ext| {
                    let ext = ext.to_string_lossy().to_lowercase();
                    ext == "jpg" || ext == "jpeg" || ext == "png"
                })
                .unwrap_or(false)
        })
        .take(args.num_images)
        .collect();

    for entry in entries {
        let path = entry.path();

        // Use catch_unwind to handle corrupted images gracefully
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            image::io::Reader::open(&path)?.decode()
        }));

        let original_image = match result {
            Ok(Ok(img)) => img,
            _ => continue,
        };

        // Letterbox to fixed 640x640 canvas for consistent shape and buffer reuse.
        let boxed = letterbox_to_square(&original_image, 640);
        let data = boxed.into_raw();
        let tensor = Tensor::from_vec(data, (640, 640, 3), &Device::Cpu)?
            .permute((2, 0, 1))?
            .unsqueeze(0)?
            .to_dtype(DType::F32)?;
        let tensor = (tensor * (1.0 / 255.0))?;

        images.push(tensor);
    }

    println!("Loaded {} images.", images.len());

    if images.is_empty() {
        println!("ERROR: No valid images found!");
        return Ok(());
    }
    println!();

    // Warmup phase - compile Metal shaders and warm caches
    println!("Warming up ({} iterations)...", WARMUP_ITERATIONS);
    let first_img = &images[0];
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

    let total_start = Instant::now();

    for img in images.iter() {
        let frame_start = Instant::now();

        let img_dev = img.to_device(&device)?;
        let pred = model.forward(&img_dev)?;
        sync_metal(&pred)?;

        let frame_time = frame_start.elapsed();
        latencies_ms.push(frame_time.as_secs_f64() * 1000.0);
    }

    let total_duration = total_start.elapsed();

    // Calculate statistics
    let num_images = images.len();
    let total_time_s = total_duration.as_secs_f64();
    let fps = num_images as f64 / total_time_s;
    let avg_latency = total_time_s * 1000.0 / num_images as f64;

    // Sort for percentile calculations
    latencies_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
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
    let device_name = if candle::utils::metal_is_available() {
        "Metal GPU"
    } else if candle::utils::cuda_is_available() {
        "CUDA GPU"
    } else {
        "CPU"
    };

    let results = json!({
        "model": "YOLOv8s",
        "implementation": "Rust (Candle 0.9.1)",
        "device": device_name,
        "precision": "FP32",
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
            "warmup_iterations": WARMUP_ITERATIONS,
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "nms_threshold": NMS_THRESHOLD
        }
    });

    let base_dir = run_dir()?;
    let filename = base_dir.join(format!("results_rust_{}.json", run_id));
    let mut file = fs::File::create(&filename)?;
    file.write_all(serde_json::to_string_pretty(&results)?.as_bytes())?;
    println!("\nResults saved to: {}", filename.display());

    Ok(())
}
