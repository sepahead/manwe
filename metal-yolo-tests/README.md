# YOLOv8s Benchmark: Rust/Candle vs PyTorch MPS vs CoreML

Performance comparison of YOLOv8s object detection on **Apple Silicon M4 Max**, testing three different implementations:

- **Rust (Candle 0.9.1)** - Metal GPU backend
- **PyTorch MPS** - Metal Performance Shaders
- **CoreML** - Apple Neural Engine / GPU / CPU

## Hardware & Software

| Component | Version |
|-----------|---------|
| CPU | Apple M4 Max |
| GPU | M4 Max integrated GPU |
| Neural Engine | M4 Max ANE |
| macOS | 15.x (Sequoia) |
| Rust | 1.x + Candle 0.9.1 |
| Python | 3.12 |
| PyTorch | 2.9.1 |
| Ultralytics | 8.3.x |
| CoreML Tools | 9.0 |

## Model

- **YOLOv8s** (small variant)
- Downloaded automatically from HuggingFace on first run
- 80 COCO classes
- F32 precision

> **Note:** Model weights (`.pt`, `.safetensors`, `.mlpackage`) are not included in this repository. They are downloaded at runtime or can be exported using the provided scripts.

## Benchmark Results

### Single Instance Performance (500 images)

| Implementation | FPS | Latency (ms) | Device |
|----------------|-----|--------------|--------|
| **CoreML** | **80.33** | **12.45** | ANE/GPU/CPU |
| Rust (Candle) | 29.94 | 33.40 | Metal GPU |
| PyTorch MPS | 27.99 | 35.73 | Metal GPU |

**Key Finding**: CoreML is **2.7x faster** than Rust/Candle and **2.9x faster** than PyTorch MPS in single-instance scenarios. This is because CoreML leverages the Apple Neural Engine (ANE), while both Rust and PyTorch only use the Metal GPU.

### Concurrent Performance (3 instances, 500 images each)

| Implementation | Avg FPS/Instance | Total FPS (3x) | Avg Latency (ms) |
|----------------|------------------|----------------|------------------|
| **CoreML** | **75.48** | **226.45** | **13.25** |
| PyTorch MPS | 26.11 | 78.32 | 38.30 |
| Rust (Candle) | 11.79 | 35.38 | 84.81 |

**Key Finding**: CoreML maintains excellent performance under load with only **6% degradation** vs single instance. PyTorch MPS scales reasonably well (Total 78 FPS), while Rust/Candle suffers significant degradation (Total 35 FPS) likely due to Metal kernel contention or locking overhead.

## Analysis

### Why is CoreML so much faster?

1.  **Apple Neural Engine (ANE)**: CoreML automatically routes inference to the 16-core Neural Engine on M4 Max.
2.  **Fused Operations**: CoreML compiles the model with fused operations (Conv+BN+ReLU), minimizing memory bandwidth usage.
3.  **Concurrent Scaling**: ANE handles multiple instances efficiently without the heavy contention seen on the GPU.

### Rust/Candle vs PyTorch MPS

- **Single Instance**: Rust (29.9 FPS) is slightly faster than PyTorch (28.0 FPS).
- **Concurrent**: PyTorch significantly outperforms Rust (78 Total FPS vs 35 Total FPS). This suggests PyTorch's Metal backend handles context switching and concurrent command encoding much better than the current Candle Metal implementation.

## Running the Benchmarks

### Prerequisites

```bash
# Install Rust dependencies
cargo build --release

# Install Python dependencies
pip install ultralytics coremltools torch

# Download CoreML model
python -c "from huggingface_hub import hf_hub_download; ..."
```

### Run All Benchmarks

```bash
./run_benchmarks.sh
```

### Run Individual Benchmarks

```bash
# Rust/Candle (500 images)
./target/release/performance_test --num-images 500 --run-id test1

# PyTorch MPS (500 images)
python performance_test_mps.py --model pytorch --num-images 500 --run-id test1

# CoreML (500 images)
python performance_test_mps.py --model coreml --num-images 500 --run-id test1
```

## Files

| File | Description |
|------|-------------|
| `src/performance_test.rs` | Rust/Candle benchmark |
| `performance_test_mps.py` | PyTorch MPS & CoreML benchmark |
| `run_benchmarks.sh` | Full benchmark suite runner |
| `plot_benchmark_results.py` | Generate plots from JSON results |
| `export_coreml_fp32.py` | Export PyTorch model to CoreML format |

## Conclusions

1. **For maximum single-instance performance**: Use **CoreML** (80.33 FPS)
2. **For Rust-based inference**: Use **Candle with Metal** (29.94 FPS)
3. **For Python-based GPU inference**: Use **PyTorch MPS** (27.99 FPS)
4. **For multi-instance workloads**: **CoreML** scales best with minimal degradation

CoreML's advantage comes from utilizing the Apple Neural Engine, making it the clear choice for production deployments on Apple Silicon when maximum performance is required.

## License

MIT
