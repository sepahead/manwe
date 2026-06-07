#!/bin/bash
set -e

RUN_DIR="runs/$(date +%Y%m%d-%H%M%S)"
VIDEO_RESULTS_DIR="$RUN_DIR/video_results"
mkdir -p "$VIDEO_RESULTS_DIR"
export RUN_DIR

# Clean old root-level outputs to avoid mixing runs
rm -rf video_results output

# Activate conda
source /opt/anaconda3/bin/activate py3-12

# FPS_LIST="15 30 60 90 120"
# Reducing list for speed if needed, but user asked for all.
FPS_LIST="15 30 60 90 120"

echo "========================================"
echo "Running RTSP Simulation Benchmarks"
echo "Run folder: $RUN_DIR"
echo "========================================"

for FPS in $FPS_LIST; do
    VIDEO_FILE="video_benchmarks/bench_${FPS}fps.mp4"
    
    echo ""
    echo "--- Testing Target FPS: $FPS ---"
    
    # 1. Single Stream
    echo "Running PyTorch Single..."
    RUN_DIR="$VIDEO_RESULTS_DIR" python benchmark_rtsp.py --model pytorch --video "$VIDEO_FILE" --run-id "single" --save-video --target-fps $FPS
    
    echo "Running CoreML Single..."
    RUN_DIR="$VIDEO_RESULTS_DIR" python benchmark_rtsp.py --model coreml --video "$VIDEO_FILE" --run-id "single" --save-video --target-fps $FPS

    echo "Running Rust Single..."
    RUN_DIR="$VIDEO_RESULTS_DIR" ./target/release/benchmark_video --video "$VIDEO_FILE" --run-id "single" --target-fps $FPS
    
    echo "Running Swift CoreML Single..."
    RUN_DIR="$VIDEO_RESULTS_DIR" ./benchmark_coreml yolov8s.mlmodelc "$VIDEO_FILE" $FPS "single"
    
    # 2. Concurrent Streams (3x)
    echo "Running PyTorch Concurrent (3x)..."
    RUN_DIR="$VIDEO_RESULTS_DIR" python benchmark_rtsp.py --model pytorch --video "$VIDEO_FILE" --run-id "conc_1" --save-video --target-fps $FPS &
    P1=$!
    RUN_DIR="$VIDEO_RESULTS_DIR" python benchmark_rtsp.py --model pytorch --video "$VIDEO_FILE" --run-id "conc_2" --save-video --target-fps $FPS &
    P2=$!
    RUN_DIR="$VIDEO_RESULTS_DIR" python benchmark_rtsp.py --model pytorch --video "$VIDEO_FILE" --run-id "conc_3" --save-video --target-fps $FPS &
    P3=$!
    wait $P1 $P2 $P3
    
    echo "Running CoreML Concurrent (3x)..."
    RUN_DIR="$VIDEO_RESULTS_DIR" python benchmark_rtsp.py --model coreml --video "$VIDEO_FILE" --run-id "conc_1" --save-video --target-fps $FPS &
    P1=$!
    RUN_DIR="$VIDEO_RESULTS_DIR" python benchmark_rtsp.py --model coreml --video "$VIDEO_FILE" --run-id "conc_2" --save-video --target-fps $FPS &
    P2=$!
    RUN_DIR="$VIDEO_RESULTS_DIR" python benchmark_rtsp.py --model coreml --video "$VIDEO_FILE" --run-id "conc_3" --save-video --target-fps $FPS &
    P3=$!
    wait $P1 $P2 $P3

    echo "Running Rust Concurrent (3x)..."
    RUN_DIR="$VIDEO_RESULTS_DIR" ./target/release/benchmark_video --video "$VIDEO_FILE" --run-id "conc_1" --target-fps $FPS &
    P1=$!
    RUN_DIR="$VIDEO_RESULTS_DIR" ./target/release/benchmark_video --video "$VIDEO_FILE" --run-id "conc_2" --target-fps $FPS &
    P2=$!
    RUN_DIR="$VIDEO_RESULTS_DIR" ./target/release/benchmark_video --video "$VIDEO_FILE" --run-id "conc_3" --target-fps $FPS &
    P3=$!
    wait $P1 $P2 $P3

    echo "Running Swift CoreML Concurrent (3x)..."
    RUN_DIR="$VIDEO_RESULTS_DIR" ./benchmark_coreml yolov8s.mlmodelc "$VIDEO_FILE" $FPS "conc_1" &
    P1=$!
    RUN_DIR="$VIDEO_RESULTS_DIR" ./benchmark_coreml yolov8s.mlmodelc "$VIDEO_FILE" $FPS "conc_2" &
    P2=$!
    RUN_DIR="$VIDEO_RESULTS_DIR" ./benchmark_coreml yolov8s.mlmodelc "$VIDEO_FILE" $FPS "conc_3" &
    P3=$!
    wait $P1 $P2 $P3
    
done

echo ""
echo "Generating Plots..."
VIDEO_RESULTS_DIR="$VIDEO_RESULTS_DIR" RUN_DIR="$RUN_DIR" python plot_video_results.py

echo "Done."
