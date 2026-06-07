#!/usr/bin/env python3
"""
YOLOv8s RTSP Simulation Benchmark - PyTorch MPS and CoreML

Simulates a live video feed (RTSP style) where frames arrive at a fixed FPS.
If inference is too slow, frames are dropped (we always process the *latest* available frame).

Metrics:
- Processed FPS (throughput)
- Drop Rate (percentage of frames missed)
- System Latency (Time from Frame Capture -> Inference Complete)
"""

import time
import sys
import os
import json
import argparse
import threading
import queue
import cv2
import numpy as np
from ultralytics import YOLO

# Configure for concurrent runs
# For concurrent runs, we don't want 3 windows opening or 3 separate print blocks interfering too much
# But we will write to JSON.

def run_rtsp_simulation(model_type, video_path, run_id, save_video=False, target_fps=0):
    run_dir = os.environ.get("RUN_DIR", "video_results")
    os.makedirs(run_dir, exist_ok=True)
    output_dir = os.path.join(run_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    # 1. Setup Model
    if model_type == "pytorch":
        import torch
        if not torch.backends.mps.is_available():
            print("Error: MPS not available")
            return
        device = torch.device("mps")
        # Load FP32 model
        model = YOLO("yolov8s.pt") 
        model.to(device)
        # Force FP32 for fairness as per user request (though previously we discussed it being slower)
        # User asked: "are you sure that everything is FP32 now?" -> Yes, we use standard .pt and .mlpackage (FP32)
    elif model_type == "coreml":
        # Load FP32 CoreML model
        model = YOLO("yolov8s.mlpackage", task="detect")
    else:
        print(f"Unknown model: {model_type}")
        return

    # 2. Setup Video Source (The "RTSP" Stream)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video: {video_path}")
        return

    file_fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Logic: If target_fps > 0, use it. If target_fps <= 0, use 0 (Unlimited).
    # Do NOT fallback to file_fps, because we want 'Max Throughput' to be truly unthrottled.
    if target_fps > 0:
        sim_fps = target_fps
        frame_interval = 1.0 / sim_fps
    else:
        sim_fps = 0.0
        frame_interval = 0.0
        
    total_frames_in_file = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Shared variables
    latest_frame = None
    latest_frame_timestamp = 0
    frame_lock = threading.Lock()
    running = True
    
    # Metrics
    frames_presented = 0
    frames_processed = 0
    latencies = []     # System Latency (Arrival -> Finish)
    decode_times = []  # Time to read/decode frame
    infer_times = []   # Time for model.predict
    
    # Video Writer
    video_writer = None
    if save_video:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        safe_name = os.path.basename(video_path).replace(".", "_")
        output_path = os.path.join(output_dir, f"video_{model_type}_{safe_name}_{run_id}.mp4")
        # Use mp4v codec for compatibility
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # If sim_fps is 0, use file_fps for video writing metadata
        writer_fps = sim_fps if sim_fps > 0 else file_fps
        video_writer = cv2.VideoWriter(output_path, fourcc, writer_fps, (width, height))
        print(f"Recording video to {output_path}")

    # 3. Reader Thread (Simulates the Camera/Network)
    def reader_loop():
        nonlocal latest_frame, latest_frame_timestamp, frames_presented, running
        
        while running:
            loop_start = time.time()
            
            t_read_start = time.time()
            ret, frame = cap.read()
            if ret:
                # Match Rust: force 640x640 resize to keep CPU decode/resize comparable.
                frame = cv2.resize(frame, (640, 640), interpolation=cv2.INTER_LINEAR)
            t_read_end = time.time()
            
            if not ret:
                running = False
                break
            
            # Simulate "Arrival"
            with frame_lock:
                latest_frame = frame
                latest_frame_timestamp = time.time() # Mark arrival time
                frames_presented += 1
                decode_times.append((t_read_end - t_read_start) * 1000.0)
            
            # Sleep to match FPS (Throttle)
            if frame_interval > 0:
                elapsed = time.time() - loop_start
                sleep_time = frame_interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
            else:
                # Yield to prevent GIL starvation in unlimited mode
                time.sleep(0)
                
        cap.release()

    reader_thread = threading.Thread(target=reader_loop)
    reader_thread.start()
    
    # 4. Inference Loop (The Consumer)
    # We only take the LATEST frame. If we are slow, we skip intermediate ones.
    
    start_benchmark_time = time.time()
    
    last_processed_ts = 0
    
    # BUG FIX: 'latest_frame' never becomes None, so we strictly rely on 'running'.
    # In RTSP simulation, once the stream stops, we stop. We don't need to "drain" a queue 
    # because there is no queue, just a single slot.
    while running:
        # Grab latest
        current_frame = None
        arrival_ts = 0
        
        with frame_lock:
            # We only process if it's NEWER than what we last processed
            if latest_frame is not None and latest_frame_timestamp > last_processed_ts:
                current_frame = latest_frame.copy() # Copy to release lock
                arrival_ts = latest_frame_timestamp
                last_processed_ts = arrival_ts
        
        if current_frame is None:
            # No new frame yet (or we are faster than camera)
            time.sleep(0.001)
            continue
            
        # Run Inference
        t0 = time.time()
        
        results = None
        if model_type == "pytorch":
            import torch
            with torch.inference_mode():
                results = model(current_frame, verbose=False, half=False)
            torch.mps.synchronize()
        else:
            results = model(current_frame, verbose=False)
            
        t1 = time.time()
        infer_times.append((t1 - t0) * 1000.0)
        
        # Video Saving
        if video_writer is not None and results is not None:
            # Plot the results
            annotated_frame = results[0].plot()
            video_writer.write(annotated_frame)
        
        # Metrics
        inference_latency = (t1 - t0) * 1000.0
        system_latency = (t1 - arrival_ts) * 1000.0 # Time from "Camera Capture" to "Result"
        
        latencies.append(system_latency)
        frames_processed += 1

    reader_thread.join()
    if video_writer:
        video_writer.release()
    
    total_duration = time.time() - start_benchmark_time
    
    # Stats
    processed_fps = frames_processed / total_duration
    drop_rate = 1.0 - (frames_processed / frames_presented) if frames_presented > 0 else 0.0
    
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    p99 = sorted(latencies)[int(0.99 * len(latencies))] if latencies else 0
    
    avg_decode = sum(decode_times) / len(decode_times) if decode_times else 0
    avg_infer = sum(infer_times) / len(infer_times) if infer_times else 0
    
    results = {
        "model": model_type,
        "video": video_path,
        "target_fps": sim_fps,
        "run_id": run_id,
        "processed_fps": processed_fps,
        "drop_rate": drop_rate,
        "avg_latency_ms": avg_latency,
        "p99_latency_ms": p99,
        "frames_presented": frames_presented,
        "frames_processed": frames_processed,
        "decode_avg_ms": avg_decode,
        "inference_avg_ms": avg_infer
    }
    
    # Save
    safe_name = os.path.basename(video_path).replace(".", "_")
    out_file = os.path.join(run_dir, f"res_{model_type}_{safe_name}_{run_id}.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"[{model_type} @ {sim_fps}FPS] Processed: {processed_fps:.2f} FPS | Drop: {drop_rate*100:.1f}% | Latency: {avg_latency:.1f}ms | Infer: {avg_infer:.1f}ms | Decode: {avg_decode:.1f}ms")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--run-id", default="0")
    parser.add_argument("--save-video", action="store_true", help="Save output video with bounding boxes")
    parser.add_argument("--target-fps", type=float, default=0, help="Target FPS for simulation (0 = use video FPS)")
    args = parser.parse_args()
    
    os.makedirs("video_results", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    run_rtsp_simulation(args.model, args.video, args.run_id, save_video=args.save_video, target_fps=args.target_fps)

if __name__ == "__main__":
    main()
