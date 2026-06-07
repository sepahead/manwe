# Manwe

> Computer vision toolkit for real-time object detection and pose estimation on Apple Silicon.

Manwe provides production-ready YOLOv8 inference across three backends — **Rust/Candle/Metal**, **PyTorch/MPS**, and **CoreML/ANE** — with an integrated benchmarking suite to help you choose the right engine for your workload. Designed for civilian aerial vehicle detection in applications such as disaster relief coordination, urban delivery logistics, and infrastructure inspection.

## Features

- **Multi-backend inference** — Rust (Candle + Metal GPU), Python (PyTorch + MPS), and CoreML (Apple Neural Engine)
- **Object detection** — 80 COCO classes with bounding boxes, labels, and confidence scores
- **Pose estimation** — 17 keypoint skeleton detection
- **Real-time camera streaming** — Multi-camera RTSP viewer with live YOLOv8 overlay
- **Comprehensive benchmarks** — Single and concurrent performance comparisons across all backends
- **Multiple model sizes** — Nano, Small, Medium, Large, and X-Large variants
- **Built for Apple Silicon** — Optimized for M-series chips with Metal GPU and ANE acceleration

## Quick Start

### Prerequisites

- macOS 15+ (Apple Silicon)
- Rust 1.82+ with `aarch64-apple-darwin` target
- Python 3.12+

### Installation

```bash
# Clone the repository
git clone https://github.com/sepehrmn/manwe.git
cd manwe

# Build the Rust CLI
cargo build --release

# Install Python dependencies for the camera viewer
pip install ultralytics torch opencv-python

# Additional dependencies for benchmark suite (optional)
pip install coremltools  # only needed for CoreML export/benchmarks
```

### Object Detection

```bash
# Run detection on an image (first run downloads the model from HuggingFace)
./target/release/manwe --which s path/to/image.jpg

# Use a larger model for higher accuracy
./target/release/manwe --which m path/to/image.jpg

# Custom confidence and NMS thresholds
./target/release/manwe --which s --confidence-threshold 0.5 --nms-threshold 0.4 image.jpg
```

Outputs an annotated `image.pp.jpg` with bounding boxes, class labels, and confidence percentages.

### Pose Estimation

```bash
./target/release/manwe --which s --task pose path/to/image.jpg
```

Draws keypoints and skeleton connections on detected persons.

### Real-Time Camera Viewer

```bash
# Configure your RTSP camera URLs in camera_view.py, then:
python camera_view.py
```

Displays live YOLOv8 detection across multiple RTSP streams side by side. Press `q` to quit.

## Benchmarks

The `metal-yolo-tests/` directory contains a full benchmarking suite comparing YOLOv8s inference across three Apple Silicon backends.

### Single Instance (M4 Max, 500 images)

| Backend | FPS | Latency (ms) | Hardware |
|---------|-----|-------------|----------|
| **CoreML** | **80.33** | **12.45** | ANE / GPU / CPU |
| Rust (Candle) | 29.94 | 33.40 | Metal GPU |
| PyTorch MPS | 27.99 | 35.73 | Metal GPU |

### Concurrent (3 instances, 500 images each)

| Backend | Avg FPS/Instance | Total FPS | Avg Latency (ms) |
|---------|------------------|-----------|------------------|
| **CoreML** | **75.48** | **226.45** | **13.25** |
| PyTorch MPS | 26.11 | 78.32 | 38.30 |
| Rust (Candle) | 11.79 | 35.38 | 84.81 |

**Key takeaway:** CoreML achieves 2.7× the throughput of the Metal GPU backends by leveraging the Apple Neural Engine (ANE), and scales to concurrent workloads with only ~6% performance degradation.

### Running Benchmarks

```bash
cd metal-yolo-tests

# Run the full benchmark suite
./run_benchmarks.sh

# Run individual backend benchmarks
cargo run --release --bin performance_test -- --num-images 500
python performance_test_mps.py --model pytorch --num-images 500
python performance_test_mps.py --model coreml --num-images 500

# Generate plots from results
python plot_benchmark_results.py
```

## Project Structure

```
manwe/
├── src/
│   ├── main.rs              # Rust YOLOv8 CLI (detection + pose)
│   ├── lib.rs               # Shared utilities (image I/O, annotation, NMS)
│   ├── model.rs             # YOLOv8 model architecture (Candle)
│   ├── coco_classes.rs      # 80-class COCO label names
│   └── bin/
│       ├── camera_view.rs   # Rust camera viewer binary
│       └── launcher.rs      # Application launcher
├── camera_view.py           # Python multi-camera RTSP viewer
├── metal-yolo-tests/        # Cross-backend benchmarking suite
│   ├── src/                 # Rust benchmark sources
│   ├── *.py                 # Python benchmark scripts
│   ├── *.sh                 # Shell benchmark runners
│   └── assets/              # Dataset configurations
├── Cargo.toml               # Rust dependencies
└── LICENSE                  # MIT
```

## Technology Stack

| Component | Technology |
|-----------|-----------|
| Rust ML framework | [Candle](https://github.com/huggingface/candle) 0.8.4 |
| Python ML framework | [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) |
| Apple GPU acceleration | Metal (Rust) / MPS (Python) |
| Apple Neural Engine | CoreML |
| Image processing | `image` + `imageproc` (Rust), OpenCV (Python) |
| Model format | safetensors |

## Use Cases

- **Disaster relief** — Real-time aerial vehicle detection for search-and-rescue coordination and supply drop planning
- **Urban delivery logistics** — Object detection and tracking for autonomous last-mile delivery systems
- **Infrastructure inspection** — Automated visual inspection of bridges, power lines, and pipelines
- **Traffic monitoring** — Vehicle and pedestrian detection for smart city applications
- **Wildlife conservation** — Animal detection and population monitoring from aerial imagery

## Contributing

Contributions are welcome! Please open an issue to discuss proposed changes before submitting a pull request. Areas of interest include:

- Additional model architectures and backends
- Performance optimizations for Apple Silicon
- New benchmark scenarios and visualizations
- Expanded civilian use case examples

## License

MIT — see [LICENSE](LICENSE) for details.
