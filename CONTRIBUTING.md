# Contributing to Manwe

Thank you for your interest in contributing! Manwe is a computer vision toolkit for real-time object detection and pose estimation on Apple Silicon, focused on civilian aerial vehicle detection for applications like disaster relief and urban delivery logistics.

## Getting Started

### Prerequisites

- macOS 15+ (Apple Silicon)
- Rust 1.82+ with `aarch64-apple-darwin` target
- Python 3.12+

### Development Setup

```bash
# Clone and build
git clone https://github.com/sepehrmn/manwe.git
cd manwe
cargo build --release

# Verify everything compiles cleanly
cargo build --release 2>&1 | grep warning  # should output nothing
```

### Project Structure

```
manwe/
├── src/                   # Rust YOLOv8 CLI (object detection + pose estimation)
│   ├── main.rs            # CLI entry point
│   ├── lib.rs             # Shared utilities (image I/O, annotation, NMS)
│   ├── model.rs           # YOLOv8 model architecture (Candle)
│   ├── coco_classes.rs    # 80-class COCO label names
│   └── bin/               # Additional binaries
├── camera_view.py         # Python multi-camera RTSP viewer
├── metal-yolo-tests/      # Cross-backend benchmarking suite
│   ├── src/               # Rust benchmark sources
│   ├── *.py               # Python benchmark scripts
│   └── *.sh               # Shell benchmark runners
├── Cargo.toml
└── LICENSE
```

## How to Contribute

### Reporting Issues

- Search [existing issues](https://github.com/sepehrmn/manwe/issues) before opening a new one
- Include your macOS version, chip model (M1/M2/M3/M4), and any relevant logs
- For performance issues, include benchmark results if available

### Pull Requests

1. **Open an issue first** to discuss the proposed change before writing code
2. **Fork the repository** and create a feature branch
3. **Keep changes focused** — one feature or fix per PR
4. **Ensure `cargo build --release` succeeds with zero warnings**
5. **Write clear commit messages** describing what changed and why
6. **Update documentation** if your change affects usage or APIs

### Code Style

- Follow existing conventions in the codebase
- Use `cargo fmt` for Rust formatting
- Use descriptive variable names
- Add comments for non-obvious logic
- Run `cargo clippy` before submitting (if installed)

### Areas of Interest

We're particularly interested in contributions in these areas:

- **Additional model architectures** — support for YOLOv9, YOLOv10, RT-DETR, etc.
- **Performance optimizations** — Metal GPU kernel improvements, CoreML integration
- **New benchmark scenarios** — multi-stream testing, different hardware comparisons
- **Expanded civilian use cases** — infrastructure inspection, wildlife monitoring, traffic analysis
- **Documentation** — tutorials, usage examples, deployment guides

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Questions?

Open an issue or start a discussion — we're happy to help!
