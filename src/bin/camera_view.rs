use bevy::prelude::*;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};
use bevy::render::render_asset::RenderAssetUsages;
use candle::{DType, Device, Tensor};
use candle_nn::{VarBuilder, Module};
use clap::Parser;
use manwe::model::{Multiples, YoloV8};
use manwe::report_detect;
use std::sync::{Arc, Mutex};
use std::thread;
use std::process::{Command, Stdio};
use std::io::Read;
use image::{DynamicImage, ImageBuffer, Rgb};

// Absolute path to ffmpeg found on system
const FFMPEG_PATH: &str = "/Users/torusprime/pinokio/bin/miniconda/bin/ffmpeg";

#[derive(Parser, Debug, Resource, Clone)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// RTSP URL (Optional, will use defaults if not provided or for multiple streams)
    #[arg(long)]
    url: Option<String>,

    /// CPU
    #[arg(long)]
    cpu: bool,
}

#[derive(Resource, Clone)]
struct FrameBuffer {
    // Index -> Image
    frames: Arc<Mutex<Vec<Option<DynamicImage>>>>,
}

#[derive(Component)]
struct CameraView {
    index: usize,
}

fn main() {
    let args = Args::parse();

    App::new()
        .add_plugins(DefaultPlugins)
        .insert_resource(args.clone())
        // Initialize buffer for 3 streams
        .insert_resource(FrameBuffer { 
            frames: Arc::new(Mutex::new(vec![None, None, None])) 
        })
        .add_systems(Startup, setup)
        .add_systems(Update, update_frame)
        .run();
}

fn setup(mut commands: Commands, args: Res<Args>, frame_buffer: Res<FrameBuffer>) {
    commands.spawn(Camera2dBundle::default());

    let urls = vec![
        "rtsp://admin:sauronsauron1@192.168.10.172:554/h264".to_string(),
        "rtsp://admin:sauronsauron1@192.168.10.145:554/h264".to_string(),
    ];

    // Load Model ONCE (or per thread? Candle models are usually thread safe if immutable)
    // But to be safe and simple, let's load it inside each thread or clone it.
    // Candle models contain Tensors which are Arc'd, so cloning is cheap.
    // However, we need to load it first.
    
    // Spawn 3 Threads
    for (i, url) in urls.into_iter().enumerate() {
        let buffer = frame_buffer.frames.clone();
        let cpu = args.cpu;
        
        thread::spawn(move || {
            println!("Loading model for stream {}...", i);
            let device = if cpu { Device::Cpu } else { Device::new_metal(0).unwrap_or(Device::Cpu) };
            let api = hf_hub::api::sync::Api::new().expect("Failed to create API");
            let api = api.model("lmz/candle-yolo-v8".to_string());
            let model_path = api.get("yolov8n.safetensors").expect("Failed to download model");
            let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[model_path], DType::F32, &device).expect("Failed to load weights") };
            let model = YoloV8::load(vb, Multiples::n(), 80).expect("Failed to load model");
            println!("Model loaded for stream {}", i);

            println!("Starting stream {} for {}", i, url);
            let width = 1280;
            let height = 720;
            let frame_size = width * height * 3;

            let child = Command::new(FFMPEG_PATH)
                .args([
                    "-i", &url,
                    "-f", "image2pipe",
                    "-pix_fmt", "rgb24",
                    "-vcodec", "rawvideo",
                    "-s", &format!("{}x{}", width, height),
                    "-",
                ])
                .stdout(Stdio::piped())
                .stderr(Stdio::null())
                .spawn();

            match child {
                Ok(mut child) => {
                    let mut stdout = child.stdout.take().expect("Failed to open stdout");
                    let mut data = vec![0u8; frame_size];

                    let mut frame_count = 0;
                    let mut last_predictions: Option<Tensor> = None;
                    
                    loop {
                        if let Ok(()) = stdout.read_exact(&mut data) {
                            let img_buf = ImageBuffer::<Rgb<u8>, _>::from_raw(width as u32, height as u32, data.clone());
                            if let Some(original_image) = img_buf {
                                let original_image = DynamicImage::ImageRgb8(original_image);
                                frame_count += 1;
                                
                                // --- YOLO INFERENCE HERE (Async Process) ---
                                // Run inference every 3rd frame for speed
                                if frame_count % 3 == 0 {
                                    // Resize for model
                                    let image_t = {
                                        let img = original_image.resize_exact(
                                            640,
                                            640,
                                            image::imageops::FilterType::CatmullRom,
                                        );
                                        let data = img.to_rgb8().into_raw();
                                        Tensor::from_vec(
                                            data,
                                            (640, 640, 3),
                                            &device,
                                        ).unwrap()
                                        .permute((2, 0, 1)).unwrap()
                                    };
                                    let image_t = (image_t.unsqueeze(0).unwrap().to_dtype(DType::F32).unwrap() * (1. / 255.)).unwrap();
                                    
                                    if let Ok(preds) = model.forward(&image_t) {
                                        if let Ok(preds) = preds.squeeze(0) {
                                            last_predictions = Some(preds);
                                        }
                                    }
                                }

                                // Always draw last known predictions if available
                                let display_image = if let Some(ref preds) = last_predictions {
                                    let (w, h) = (original_image.width() as usize, original_image.height() as usize);
                                    report_detect(
                                        preds,
                                        original_image,
                                        w,
                                        h,
                                        0.25,
                                        0.45,
                                        14,
                                    ).unwrap_or_else(|_| {
                                        // If drawing fails, return original
                                        // We need to reconstruct original_image because report_detect consumes it
                                        // But wait, report_detect takes ownership.
                                        // We can't easily recover it if it fails inside.
                                        // But report_detect failure is rare (image ops).
                                        // Actually, if we pass original_image to report_detect, we can't use it in the else block.
                                        // So we need to handle the flow carefully.
                                        // Let's just unwrap for now as before.
                                        panic!("Report detect failed");
                                    })
                                } else {
                                    original_image
                                };
                                // -------------------------------------------

                                let mut lock = buffer.lock().unwrap();
                                lock[i] = Some(display_image);
                            }
                        } else {
                            eprintln!("Stream {} ended", i);
                            break;
                        }
                    }
                }
                Err(e) => {
                    eprintln!("Failed to spawn ffmpeg for stream {}: {}", i, e);
                }
            }
        });
    }

    // Spawn Sprites for Display
    // Grid layout for 2 cameras: Side by side
    // Images are 1280x720. Scale 0.5 -> 640x360.
    let scale = 0.5;
    let positions = vec![
        Vec3::new(-330.0, 0.0, 0.0), // Left
        Vec3::new(330.0, 0.0, 0.0),  // Right
    ];

    for i in 0..2 {
        let pos = if i < positions.len() { positions[i] } else { Vec3::ZERO };
        commands.spawn((
            SpriteBundle {
                transform: Transform::from_translation(pos).with_scale(Vec3::splat(scale)),
                ..default()
            },
            CameraView { index: i },
        ));
    }
}

fn update_frame(
    frame_buffer: Res<FrameBuffer>,
    mut images: ResMut<Assets<Image>>,
    mut query: Query<(&CameraView, &mut Handle<Image>)>,
) {
    let frames = {
        let lock = frame_buffer.frames.lock().unwrap();
        lock.clone() 
    };

    for (view, mut handle) in query.iter_mut() {
        if let Some(Some(img)) = frames.get(view.index) {
             let rgba_image = img.to_rgba8();
            let size = Extent3d {
                width: rgba_image.width(),
                height: rgba_image.height(),
                depth_or_array_layers: 1,
            };
            let image = Image::new(
                size,
                TextureDimension::D2,
                rgba_image.into_raw(),
                TextureFormat::Rgba8UnormSrgb,
                RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
            );
            *handle = images.add(image);
        }
    }
}
