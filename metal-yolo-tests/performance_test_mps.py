#!/usr/bin/env python3
"""
YOLOv8s Performance Benchmark - PyTorch MPS and CoreML

This benchmark measures inference performance of YOLOv8s object detection
using PyTorch with Metal Performance Shaders (MPS) and CoreML on Apple Silicon.

Precision: FP32 (single precision) for fair comparison across implementations.

Metrics:
- FPS (frames per second)
- Latency (milliseconds per frame)
- P1 latency (1st percentile - fastest frames)
- P99 latency (99th percentile - slowest frames / potential dropped frames)
"""

import torch
import time
import sys
import os
import json
import argparse
import numpy as np
from ultralytics import YOLO
from PIL import Image
from typing import List, Dict, Any
import os

# Benchmark configuration
WARMUP_ITERATIONS = 10
CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45


def calculate_percentile(latencies: List[float], p: float) -> float:
    """Calculate the p-th percentile of latencies."""
    if not latencies:
        return 0.0
    sorted_lat = sorted(latencies)
    idx = int((p / 100.0) * (len(sorted_lat) - 1))
    return sorted_lat[min(idx, len(sorted_lat) - 1)]


def load_images(image_dir: str, num_images: int) -> List[Image.Image]:
    """Load and preprocess images from directory."""
    images = []
    
    if not os.path.exists(image_dir):
        print(f"ERROR: Image directory not found: {image_dir}")
        return images
    
    all_files = sorted([
        f for f in os.listdir(image_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])
    
    for filename in all_files[:num_images]:
        try:
            img_path = os.path.join(image_dir, filename)
            img = Image.open(img_path)
            img.load()
            # Convert to RGB to handle palette/transparency images
            if img.mode != 'RGB':
                img = img.convert('RGB')
            images.append(img)
        except Exception:
            continue
    
    return images


def run_pytorch_benchmark(
    images: List[Image.Image],
    run_id: str,
    device: torch.device
) -> Dict[str, Any]:
    """
    Run PyTorch MPS benchmark.
    
    Uses Metal Performance Shaders (MPS) for GPU acceleration.
    All computations in FP32 precision.
    """
    print("╔══════════════════════════════════════════════════════════╗")
    print("║      YOLOv8s Benchmark - PyTorch MPS (Metal GPU)         ║")
    print("╠══════════════════════════════════════════════════════════╣")
    print("║ Precision: FP32                                          ║")
    print("║ Model: YOLOv8s (~11M params, 21MB weights)               ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()
    print(f"Configuration:")
    print(f"  Images: {len(images)}")
    print(f"  Run ID: {run_id}")
    print(f"  Warmup: {WARMUP_ITERATIONS} iterations")
    print(f"  Device: Metal GPU (MPS)")
    print()

    # Load model
    print("Loading YOLOv8s model...")
    model = YOLO("yolov8s.pt")
    model.to(device)
    print("Model loaded successfully.")
    print()

    # Warmup phase
    print(f"Warming up ({WARMUP_ITERATIONS} iterations)...")
    with torch.inference_mode():
        for _ in range(WARMUP_ITERATIONS):
            model(images[0], device=device, verbose=False, half=False)
        torch.mps.synchronize()
    print("Warmup complete.")
    print()

    # Benchmark phase
    print("Running benchmark...")
    latencies_ms = []
    
    torch.mps.synchronize()
    total_start = time.perf_counter()
    
    with torch.inference_mode():
        for img in images:
            frame_start = time.perf_counter()
            model(img, device=device, verbose=False, 
                  conf=CONFIDENCE_THRESHOLD, iou=IOU_THRESHOLD, half=False)
            torch.mps.synchronize()
            frame_time = (time.perf_counter() - frame_start) * 1000.0
            latencies_ms.append(frame_time)
    
    total_time = time.perf_counter() - total_start

    # Calculate statistics
    num_images = len(images)
    fps = num_images / total_time
    avg_latency = total_time * 1000.0 / num_images
    
    sorted_latencies = sorted(latencies_ms)
    min_latency = sorted_latencies[0]
    max_latency = sorted_latencies[-1]
    p1_latency = calculate_percentile(latencies_ms, 1.0)
    p50_latency = calculate_percentile(latencies_ms, 50.0)
    p99_latency = calculate_percentile(latencies_ms, 99.0)

    # Print results
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║                        RESULTS                           ║")
    print("╠══════════════════════════════════════════════════════════╣")
    print(f"║ Images processed: {num_images:>6}                                 ║")
    print(f"║ Total time:       {total_time:>6.2f} s                               ║")
    print(f"║ Average FPS:      {fps:>6.2f}                                 ║")
    print(f"║ Avg latency:      {avg_latency:>6.2f} ms                              ║")
    print("╠══════════════════════════════════════════════════════════╣")
    print("║ Latency Percentiles:                                     ║")
    print(f"║   Min (best):     {min_latency:>6.2f} ms                              ║")
    print(f"║   P1:             {p1_latency:>6.2f} ms                              ║")
    print(f"║   P50 (median):   {p50_latency:>6.2f} ms                              ║")
    print(f"║   P99:            {p99_latency:>6.2f} ms                              ║")
    print(f"║   Max (worst):    {max_latency:>6.2f} ms                              ║")
    print("╚══════════════════════════════════════════════════════════╝")

    return {
        "model": "YOLOv8s",
        "implementation": "PyTorch MPS",
        "device": "Metal GPU",
        "precision": "FP32",
        "images": num_images,
        "total_time_s": total_time,
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
            "iou_threshold": IOU_THRESHOLD
        }
    }


def run_coreml_benchmark(
    images: List[Image.Image],
    run_id: str
) -> Dict[str, Any]:
    """
    Run CoreML benchmark.
    
    CoreML automatically selects optimal compute units (ANE/GPU/CPU).
    Model uses FP32 precision for fair comparison.
    """
    print("╔══════════════════════════════════════════════════════════╗")
    print("║        YOLOv8s Benchmark - CoreML (ANE/GPU/CPU)          ║")
    print("╠══════════════════════════════════════════════════════════╣")
    print("║ Precision: FP32                                          ║")
    print("║ Model: YOLOv8s (~11M params, 21MB weights)               ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()
    print(f"Configuration:")
    print(f"  Images: {len(images)}")
    print(f"  Run ID: {run_id}")
    print(f"  Warmup: {WARMUP_ITERATIONS} iterations")
    print(f"  Device: ANE/GPU/CPU (auto-selected)")
    print()

    # Load CoreML model
    model_path = "yolov8s.mlpackage"
    if not os.path.exists(model_path):
        print(f"ERROR: CoreML model not found: {model_path}")
        return {}
    
    print("Loading YOLOv8s CoreML model...")
    model = YOLO(model_path, task='detect')
    print("Model loaded successfully.")
    print()

    # Warmup phase
    print(f"Warming up ({WARMUP_ITERATIONS} iterations)...")
    for _ in range(WARMUP_ITERATIONS):
        model(images[0], verbose=False)
    print("Warmup complete.")
    print()

    # Benchmark phase
    print("Running benchmark...")
    latencies_ms = []
    
    total_start = time.perf_counter()
    
    for img in images:
        frame_start = time.perf_counter()
        model(img, verbose=False, conf=CONFIDENCE_THRESHOLD, iou=IOU_THRESHOLD)
        frame_time = (time.perf_counter() - frame_start) * 1000.0
        latencies_ms.append(frame_time)
    
    total_time = time.perf_counter() - total_start

    # Calculate statistics
    num_images = len(images)
    fps = num_images / total_time
    avg_latency = total_time * 1000.0 / num_images
    
    sorted_latencies = sorted(latencies_ms)
    min_latency = sorted_latencies[0]
    max_latency = sorted_latencies[-1]
    p1_latency = calculate_percentile(latencies_ms, 1.0)
    p50_latency = calculate_percentile(latencies_ms, 50.0)
    p99_latency = calculate_percentile(latencies_ms, 99.0)

    # Print results
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║                        RESULTS                           ║")
    print("╠══════════════════════════════════════════════════════════╣")
    print(f"║ Images processed: {num_images:>6}                                 ║")
    print(f"║ Total time:       {total_time:>6.2f} s                               ║")
    print(f"║ Average FPS:      {fps:>6.2f}                                 ║")
    print(f"║ Avg latency:      {avg_latency:>6.2f} ms                              ║")
    print("╠══════════════════════════════════════════════════════════╣")
    print("║ Latency Percentiles:                                     ║")
    print(f"║   Min (best):     {min_latency:>6.2f} ms                              ║")
    print(f"║   P1:             {p1_latency:>6.2f} ms                              ║")
    print(f"║   P50 (median):   {p50_latency:>6.2f} ms                              ║")
    print(f"║   P99:            {p99_latency:>6.2f} ms                              ║")
    print(f"║   Max (worst):    {max_latency:>6.2f} ms                              ║")
    print("╚══════════════════════════════════════════════════════════╝")

    return {
        "model": "YOLOv8s",
        "implementation": "CoreML",
        "device": "ANE/GPU/CPU",
        "precision": "FP32",
        "images": num_images,
        "total_time_s": total_time,
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
            "iou_threshold": IOU_THRESHOLD
        }
    }


def main():
    parser = argparse.ArgumentParser(
        description='YOLOv8s Benchmark - PyTorch MPS / CoreML'
    )
    parser.add_argument(
        "--model", type=str, required=True,
        choices=["pytorch", "coreml"],
        help="Implementation to benchmark"
    )
    parser.add_argument(
        "--num-images", type=int, default=500,
        help="Number of images to process"
    )
    parser.add_argument(
        "--run-id", type=str, default="",
        help="Run identifier for result files"
    )
    args = parser.parse_args()

    run_id = args.run_id if args.run_id else str(os.getpid())

    # Verify MPS availability
    if not torch.backends.mps.is_available():
        print("ERROR: MPS not available. This benchmark requires Apple Silicon.")
        sys.exit(1)
    
    device = torch.device("mps")

    # Create output directory
    os.makedirs("output", exist_ok=True)

    # Load images
    image_dir = "assets/kaggle1/train/images"
    print(f"Loading images from {image_dir}...")
    images = load_images(image_dir, args.num_images)
    
    if not images:
        print("ERROR: No valid images found!")
        sys.exit(1)
    
    print(f"Loaded {len(images)} images.")
    print()

    # Run benchmark
    if args.model == "pytorch":
        results = run_pytorch_benchmark(images, run_id, device)
    else:
        results = run_coreml_benchmark(images, run_id)

    if not results:
        sys.exit(1)

    # Save results
    run_dir = os.environ.get("RUN_DIR", ".")
    os.makedirs(run_dir, exist_ok=True)
    filename = os.path.join(run_dir, f"results_{args.model}_{run_id}.json")
    with open(filename, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {filename}")


if __name__ == "__main__":
    main()
