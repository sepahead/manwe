use clap::Parser;
use std::io::{Read, Write};
use std::process::{Command, Stdio};
use std::sync::{
    atomic::{AtomicBool, AtomicU64, Ordering},
    Arc, Mutex,
};
use std::thread;
use std::time::{Duration, Instant};
use candle::{Device, DType, IndexOp, Tensor};
use candle_nn::{Module, VarBuilder};
use image::{ImageBuffer, Rgb};

// Import the model definition from the existing file
#[path = "model.rs"]
mod model;
use model::{Multiples, YoloV8};
#[path = "coco_classes.rs"]
mod coco_classes;

const NUM_CLASSES: usize = 80;
const INPUT_W: usize = 640;
const INPUT_H: usize = 640;

#[derive(Clone)]
struct FramePacket {
    data: Vec<u8>,
    arrival: Instant,
    decode_ms: f64,
    idx: u64,
}

#[derive(Parser, Clone)]
struct Args {
    #[arg(long)]
    video: String,
    #[arg(long)]
    target_fps: f64,
    #[arg(long)]
    run_id: String,
    #[arg(long)]
    save_video: bool,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let run_dir = std::env::var("RUN_DIR").unwrap_or_else(|_| "video_results".to_string());
    let run_dir = std::path::PathBuf::from(run_dir);
    std::fs::create_dir_all(&run_dir)?;
    // Explicitly clear any stray per-run outputs when RUN_DIR points to the root to avoid mixing runs.
    if run_dir == std::path::PathBuf::from("video_results") {
        let _ = std::fs::remove_file(run_dir.join("res_rust.tmp")); // no-op marker
    }
    
    // 1. Load Model
    let device = Device::new_metal(0).unwrap_or(Device::Cpu);
    println!("Loading Model on {:?}...", device);
    
    let api = hf_hub::api::sync::Api::new()?;
    let api_repo = api.model("lmz/candle-yolo-v8".to_string());
    let model_path = api_repo.get("yolov8s.safetensors")?;
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[&model_path], DType::F32, &device)? };
    let model = YoloV8::load(vb, Multiples::s(), NUM_CLASSES)?;
    
    // 2. Setup Input Pipe (FFmpeg -> Rust)
    // We force resize to 640x640 in FFmpeg to simulate hardware scaler/efficient decode
    let mut input_cmd = Command::new("ffmpeg")
        .args(&[
            "-i", &args.video,
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "-vcodec", "rawvideo",
            "-s", &format!("{}x{}", INPUT_W, INPUT_H),
            "-",
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::null()) // Silence ffmpeg logs
        .spawn()?;
        
    let mut input_stdout = input_cmd.stdout.take().unwrap();
    
    // 3. Setup Output Pipe (Rust -> FFmpeg) if needed
    let mut output_stdin = if args.save_video {
        let safe_name = std::path::Path::new(&args.video)
            .file_name().unwrap().to_str().unwrap().replace(".", "_");
        let out_path = run_dir.join(format!("video_rust_{}_{}.mp4", safe_name, args.run_id));
        std::fs::create_dir_all(&run_dir)?;
        
        let mut cmd = Command::new("ffmpeg")
            .args(&[
                "-f", "rawvideo",
                "-pix_fmt", "rgb24",
                "-s", &format!("{}x{}", INPUT_W, INPUT_H),
                "-r", &format!("{}", args.target_fps),
                "-i", "-",
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-y", // overwrite
                out_path.to_str().unwrap(),
            ])
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()?;
        
        Some(cmd.stdin.take().unwrap())
    } else {
        None
    };

    // 4. Reader thread: keep only the latest frame (drop older ones) to mirror RTSP simulation semantics.
    let running = Arc::new(AtomicBool::new(true));
    let frames_presented = Arc::new(AtomicU64::new(0));
    let latest_frame: Arc<Mutex<Option<FramePacket>>> = Arc::new(Mutex::new(None));

    let reader_running = running.clone();
    let reader_presented = frames_presented.clone();
    let reader_latest = latest_frame.clone();

    let reader_thread = thread::spawn(move || {
        let frame_size = INPUT_W * INPUT_H * 3;
        let mut buffer = vec![0u8; frame_size];
        loop {
            let t0 = Instant::now();
            match input_stdout.read_exact(&mut buffer) {
                Ok(_) => {
                    let decode_ms = t0.elapsed().as_secs_f64() * 1000.0;
                    let idx = reader_presented.fetch_add(1, Ordering::Relaxed) + 1;
                    let packet = FramePacket {
                        data: buffer.clone(),
                        arrival: Instant::now(),
                        decode_ms,
                        idx,
                    };
                    if let Ok(mut slot) = reader_latest.lock() {
                        *slot = Some(packet);
                    }
                }
                Err(_) => {
                    reader_running.store(false, Ordering::Relaxed);
                    break; // EOF
                }
            }
        }
    });

    // 5. Main Loop (Inference)
    let mut frames_processed: u64 = 0;
    let mut decode_times = Vec::new(); // Actual FFmpeg read time
    let mut upload_times = Vec::new();
    let mut compute_times = Vec::new();
    let mut total_infer_times = Vec::new();
    let mut latencies = Vec::new();
    
    let start_time = Instant::now();
    let frame_interval = if args.target_fps > 0.0 {
        Duration::from_secs_f64(1.0 / args.target_fps)
    } else {
        Duration::ZERO
    };

    // Warmup
    {
        let dummy = Tensor::zeros((1, 3, 640, 640), DType::F32, &device)?;
        let _ = model.forward(&dummy)?;
    }

    println!("Starting Rust Benchmark Loop...");

    let mut last_seen_idx: u64 = 0;

    while running.load(Ordering::Relaxed) || last_seen_idx < frames_presented.load(Ordering::Relaxed) {
        let packet = {
            let slot = latest_frame.lock().unwrap();
            slot.as_ref()
                .filter(|p| p.idx > last_seen_idx)
                .cloned()
        };

        let Some(packet) = packet else {
            thread::sleep(Duration::from_millis(1));
            continue;
        };

        last_seen_idx = packet.idx;

        // Record decode time and increment processed count
        decode_times.push(packet.decode_ms);
        frames_processed += 1;

        let loop_start = Instant::now();

        // 1. Upload / Preprocess (CPU -> GPU)
        let t_upload_start = Instant::now();
        // 1. Upload / Preprocess (CPU -> GPU)
        let t_upload_start = Instant::now();
        let tensor = Tensor::from_vec(packet.data.clone(), (INPUT_H, INPUT_W, 3), &device)?
            .permute((2, 0, 1))?
            .unsqueeze(0)?
            .to_dtype(DType::F32)?;
        let tensor = (tensor * (1.0 / 255.0))?;
        let t_upload_end = Instant::now();
        let upload_dur = t_upload_end.duration_since(t_upload_start).as_secs_f64() * 1000.0;
        upload_times.push(upload_dur);

        // 2. Compute (Inference)
        let t_compute_start = Instant::now();
        let preds = model.forward(&tensor)?;
        // Lightweight sync for timing without copying full tensor back
        let _ = preds.i((0, 0, 0))?.to_scalar::<f32>()?;
        let t_compute_end = Instant::now();
        let compute_dur = t_compute_end.duration_since(t_compute_start).as_secs_f64() * 1000.0;
        compute_times.push(compute_dur);
        
        // Total Inference (Upload + Compute)
        let total_infer_dur = t_compute_end.duration_since(t_upload_start).as_secs_f64() * 1000.0;
        total_infer_times.push(total_infer_dur);
        
        // Total System Latency (Camera/Read -> Result)
        let sys_latency = t_compute_end.duration_since(packet.arrival).as_secs_f64() * 1000.0;
        latencies.push(sys_latency);
        
        // Video Writing (Optional)
        if let Some(ref mut writer) = output_stdin {
            use candle_transformers::object_detection::{non_maximum_suppression, Bbox};
             
            let pred_host = preds.to_device(&Device::Cpu)?;
            let pred_host = pred_host.i(0)?;
            let (pred_size, npreds) = pred_host.dims2()?;
            let nclasses = pred_size - 4;
            let mut bboxes: Vec<Vec<Bbox<Vec<()>>>> = (0..nclasses).map(|_| vec![]).collect();
             
            for index in 0..npreds {
                let p_vec = Vec::<f32>::try_from(pred_host.i((.., index))?)?;
                let confidence = *p_vec[4..].iter().max_by(|x, y| x.total_cmp(y)).unwrap();
                if confidence > 0.25 {
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
                }
            }
            non_maximum_suppression(&mut bboxes, 0.45);
             
            let mut img_buf = ImageBuffer::<Rgb<u8>, _>::from_raw(INPUT_W as u32, INPUT_H as u32, packet.data.clone()).unwrap();
             
            for class_boxes in bboxes.iter() {
                for b in class_boxes {
                    let x = b.xmin.max(0.) as i32;
                    let y = b.ymin.max(0.) as i32;
                    let w = (b.xmax - b.xmin).max(1.) as u32;
                    let h = (b.ymax - b.ymin).max(1.) as u32;
                     
                    imageproc::drawing::draw_hollow_rect_mut(
                        &mut img_buf,
                        imageproc::rect::Rect::at(x, y).of_size(w, h),
                        Rgb([255, 0, 0])
                    );
                }
            }
             
            writer.write_all(&img_buf.into_raw())?;
        }

        if frames_processed % 100 == 0 {
            println!("[Rust] Processed {} frames...", frames_processed);
        }

        // Throttle consumer a tiny bit if target fps is set to reduce busy-spin when faster than source
        if args.target_fps > 0.0 {
            let elapsed = loop_start.elapsed();
            if frame_interval > elapsed {
                thread::sleep(frame_interval - elapsed);
            }
        }
    }
    
    reader_thread.join().unwrap();
    
    // Calculate Stats
    let total_dur = start_time.elapsed().as_secs_f64();
    let processed_fps = frames_processed as f64 / total_dur;
    
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
    let avg_decode = if decode_times.is_empty() {
        0.0
    } else {
        decode_times.iter().sum::<f64>() / decode_times.len() as f64
    };
    let avg_infer = if total_infer_times.is_empty() {
        0.0
    } else {
        total_infer_times.iter().sum::<f64>() / total_infer_times.len() as f64
    };
    let avg_upload = if upload_times.is_empty() {
        0.0
    } else {
        upload_times.iter().sum::<f64>() / upload_times.len() as f64
    };
    let avg_compute = if compute_times.is_empty() {
        0.0
    } else {
        compute_times.iter().sum::<f64>() / compute_times.len() as f64
    };
    
    latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p99 = if latencies.is_empty() {
        0.0
    } else {
        let idx = ((latencies.len() as f64 * 0.99).floor() as usize).min(latencies.len() - 1);
        latencies[idx]
    };

    let results = serde_json::json!({
        "model": "rust_candle",
        "video": args.video,
        "target_fps": args.target_fps,
        "run_id": args.run_id,
        "processed_fps": processed_fps,
        "drop_rate": drop_rate,
        "avg_latency_ms": avg_lat,
        "p99_latency_ms": p99,
        "decode_avg_ms": avg_decode,
        "inference_avg_ms": avg_infer,
        "upload_avg_ms": avg_upload,
        "compute_avg_ms": avg_compute
    });
    
    let safe_name = std::path::Path::new(&args.video)
        .file_name().unwrap().to_str().unwrap().replace(".", "_");
    let filename = run_dir.join(format!("res_rust_{}_{}.json", safe_name, args.run_id));
    
    let mut file = std::fs::File::create(&filename)?;
    file.write_all(serde_json::to_string_pretty(&results)?.as_bytes())?;
    
    println!("[Rust Candle] FPS: {:.2} | Infer: {:.1}ms (Up: {:.1}ms, Comp: {:.1}ms) | Decode: {:.1}ms", 
        processed_fps, avg_infer, avg_upload, avg_compute, avg_decode);

    Ok(())
}
