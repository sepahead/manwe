#!/bin/bash
set -e

RUN_DIR="runs/$(date +%Y%m%d-%H%M%S)"
export RUN_DIR

# Activate conda for Python
source /opt/anaconda3/bin/activate py3-12

# Fix for OMP: Error #15 on macOS with PyTorch
export KMP_DUPLICATE_LIB_OK=TRUE

# Defaults
RUN_VERIFY=false
RUN_PERF=false
RUN_CONCURRENT=true
RUN_MAX_SINGLE=false
RUN_MAX_CONCURRENT=true
RUN_IMAGE=false

# Parse Arguments
if [ $# -eq 0 ]; then
    echo "No arguments provided. Defaulting to Concurrent 3x Benchmarks (30 FPS & Max)."
else
    # Reset defaults if args are provided, unless purely additive logic is desired.
    # Let's assume if args are provided, we only run what is asked, unless it's just setting a specific one.
    # But simpler: Default is CONC & MAX_CONC. If user passes flag, maybe they want ONLY that?
    # Standard CLI behavior: specific flags usually override default "all" or "main".
    # Here default is "3x ones".
    # If I pass --image, do I want ONLY image? Yes.
    # So reset all to false if any flag is passed, then enable.
    RUN_VERIFY=false
    RUN_PERF=false
    RUN_CONCURRENT=false
    RUN_MAX_SINGLE=false
    RUN_MAX_CONCURRENT=false
    RUN_IMAGE=false

    for arg in "$@"; do
        case $arg in
            --all)
                RUN_VERIFY=true
                RUN_PERF=true
                RUN_CONCURRENT=true
                RUN_MAX_SINGLE=true
                RUN_MAX_CONCURRENT=true
                RUN_IMAGE=true
                shift
                ;;
            --verify)
                RUN_VERIFY=true
                shift
                ;;
            --perf)
                RUN_PERF=true
                shift
                ;;
            --concurrent)
                RUN_CONCURRENT=true
                shift
                ;;
            --max-single)
                RUN_MAX_SINGLE=true
                shift
                ;;
            --max-concurrent)
                RUN_MAX_CONCURRENT=true
                shift
                ;;
            --image)
                RUN_IMAGE=true
                shift
                ;;
            --help)
                echo "Usage: ./run_benchmark_suite.sh [flags]"
                echo "Flags:"
                echo "  --all             Run all benchmarks"
                echo "  --verify          Run Phase 1: Verification (Video Output)"
                echo "  --perf            Run Phase 2: Realtime Performance (30/60/90 FPS Single)"
                echo "  --concurrent      Run Phase 3: Concurrent Load (3x Streams @ 30 FPS)"
                echo "  --max-single      Run Phase 4: Max Throughput (Single Stream Unlimited)"
                echo "  --max-concurrent  Run Phase 5: Concurrent Max Throughput (3x Streams Unlimited)"
                echo "  --image           Run Phase 6: Static Image Benchmarks"
                exit 0
                ;;
            *)
                echo "Unknown argument: $arg"
                exit 1
                ;;
        esac
    done
fi

# Cleanup old results
echo "Cleaning up old results and assets..."
rm -f video_results/*.json
rm -f output/*.mp4
rm -f *.png
# rm -f video_benchmarks/*.mp4
mkdir -p video_results
mkdir -p output
mkdir -p video_benchmarks

# Generate Benchmark Videos (Ensure correct FPS)
# echo "Generating benchmark videos..."
# python generate_benchmark_videos.py

VIDEO_FILE="video_benchmarks/bench_30fps.mp4"
VIDEO_RESULTS_DIR="$RUN_DIR/video_results"
mkdir -p "$VIDEO_RESULTS_DIR"

echo "=========================================================="
echo "       METAL YOLOv8 BENCHMARK SUITE (M-Series)            "
echo "=========================================================="
echo "Date: $(date)"
echo "System: $(uname -a)"
echo "----------------------------------------------------------"

run_benchmark() {
    MODEL=$1
    IMPL=$2 # python, rust, swift
    TARGET_FPS=$3
    RUN_ID=$4
    SAVE_VIDEO=$5
    
    echo "[Running] Model: $MODEL | Impl: $IMPL | Target: ${TARGET_FPS}FPS | Video: $SAVE_VIDEO"
    
    CMD=""
    if [ "$IMPL" == "python" ]; then
        CMD="RUN_DIR=\"$VIDEO_RESULTS_DIR\" python benchmark_rtsp.py --model $MODEL --video $VIDEO_FILE --run-id $RUN_ID --target-fps $TARGET_FPS"
        if [ "$SAVE_VIDEO" == "yes" ]; then
            CMD="$CMD --save-video"
        fi
    elif [ "$IMPL" == "rust" ]; then
        CMD="RUN_DIR=\"$VIDEO_RESULTS_DIR\" ./target/release/benchmark_video --video $VIDEO_FILE --run-id $RUN_ID --target-fps $TARGET_FPS"
        if [ "$SAVE_VIDEO" == "yes" ]; then
            CMD="$CMD --save-video"
        fi
    elif [ "$IMPL" == "swift" ]; then
        CMD="RUN_DIR=\"$VIDEO_RESULTS_DIR\" ./benchmark_coreml yolov8s.mlmodelc $VIDEO_FILE $TARGET_FPS $RUN_ID"
    fi
    
    eval $CMD
    echo "Done."
    echo ""
}

# 1. VERIFICATION RUNS
if [ "$RUN_VERIFY" = true ]; then
    echo "--- Phase 1: Verification (Video Output Enabled) ---"
    run_benchmark "pytorch" "python" 30 "verify_video" "yes"
    run_benchmark "coreml" "python" 30 "verify_video" "yes"
    run_benchmark "rust_candle" "rust" 30 "verify_video" "yes"
    run_benchmark "swift_coreml" "swift" 30 "verify_video" "no" 
fi

# 2. PERFORMANCE RUNS
if [ "$RUN_PERF" = true ]; then
    echo "--- Phase 2: Realtime Performance (30/60/90 FPS Target) ---"
    for FPS in 30 60 90; do
        echo ">> Running @ ${FPS} FPS..."
        run_benchmark "pytorch" "python" $FPS "perf_${FPS}_single" "no"
        run_benchmark "coreml" "python" $FPS "perf_${FPS}_single" "no"
        run_benchmark "rust_candle" "rust" $FPS "perf_${FPS}_single" "no"
        run_benchmark "swift_coreml" "swift" $FPS "perf_${FPS}_single" "no"
    done
fi

# 3. CONCURRENT RUNS
if [ "$RUN_CONCURRENT" = true ]; then
    echo "--- Phase 3: Concurrent Load (3x Streams) ---"
    echo "Running PyTorch 3x..."
    RUN_DIR="$VIDEO_RESULTS_DIR" python benchmark_rtsp.py --model pytorch --video $VIDEO_FILE --run-id conc_1 --target-fps 30 &
    P1=$!
    sleep 2
    RUN_DIR="$VIDEO_RESULTS_DIR" python benchmark_rtsp.py --model pytorch --video $VIDEO_FILE --run-id conc_2 --target-fps 30 &
    P2=$!
    sleep 2
    RUN_DIR="$VIDEO_RESULTS_DIR" python benchmark_rtsp.py --model pytorch --video $VIDEO_FILE --run-id conc_3 --target-fps 30 &
    P3=$!
    wait $P1 $P2 $P3

    echo "Running CoreML (Python) 3x..."
    RUN_DIR="$VIDEO_RESULTS_DIR" python benchmark_rtsp.py --model coreml --video $VIDEO_FILE --run-id conc_1 --target-fps 30 &
    P1=$!
    sleep 2
    RUN_DIR="$VIDEO_RESULTS_DIR" python benchmark_rtsp.py --model coreml --video $VIDEO_FILE --run-id conc_2 --target-fps 30 &
    P2=$!
    sleep 2
    RUN_DIR="$VIDEO_RESULTS_DIR" python benchmark_rtsp.py --model coreml --video $VIDEO_FILE --run-id conc_3 --target-fps 30 &
    P3=$!
    wait $P1 $P2 $P3

    echo "Running Rust Candle 3x..."
    RUN_DIR="$VIDEO_RESULTS_DIR" ./target/release/benchmark_video --video $VIDEO_FILE --run-id conc_1 --target-fps 30 &
    P1=$!
    sleep 2
    RUN_DIR="$VIDEO_RESULTS_DIR" ./target/release/benchmark_video --video $VIDEO_FILE --run-id conc_2 --target-fps 30 &
    P2=$!
    sleep 2
    RUN_DIR="$VIDEO_RESULTS_DIR" ./target/release/benchmark_video --video $VIDEO_FILE --run-id conc_3 --target-fps 30 &
    P3=$!
    wait $P1 $P2 $P3

    echo "Running Swift CoreML 3x..."
    RUN_DIR="$VIDEO_RESULTS_DIR" ./benchmark_coreml yolov8s.mlmodelc $VIDEO_FILE 30 conc_1 &
    P1=$!
    RUN_DIR="$VIDEO_RESULTS_DIR" ./benchmark_coreml yolov8s.mlmodelc $VIDEO_FILE 30 conc_2 &
    P2=$!
    RUN_DIR="$VIDEO_RESULTS_DIR" ./benchmark_coreml yolov8s.mlmodelc $VIDEO_FILE 30 conc_3 &
    P3=$!
    wait $P1 $P2 $P3
fi

# 4. MAXIMUM THROUGHPUT (Single)
if [ "$RUN_MAX_SINGLE" = true ]; then
    echo "--- Phase 4: Max Throughput (Unlimited FPS) ---"
    
    echo "Running PyTorch Max..."
    RUN_DIR="$VIDEO_RESULTS_DIR" python benchmark_rtsp.py --model pytorch --video $VIDEO_FILE --run-id max_fps --target-fps 0

    echo "Running CoreML (Python) Max..."
    RUN_DIR="$VIDEO_RESULTS_DIR" python benchmark_rtsp.py --model coreml --video $VIDEO_FILE --run-id max_fps --target-fps 0

    echo "Running Rust Max..."
    RUN_DIR="$VIDEO_RESULTS_DIR" ./target/release/benchmark_video --video $VIDEO_FILE --run-id max_fps --target-fps 0

    echo "Running Swift Max..."
    RUN_DIR="$VIDEO_RESULTS_DIR" ./benchmark_coreml yolov8s.mlmodelc $VIDEO_FILE 0 max_fps
fi

# 5. CONCURRENT MAX RUNS
if [ "$RUN_MAX_CONCURRENT" = true ]; then
    echo "--- Phase 5: Concurrent Max Throughput (Unlimited FPS) ---"
    echo "Running PyTorch 3x Max..."
    RUN_DIR="$VIDEO_RESULTS_DIR" python benchmark_rtsp.py --model pytorch --video $VIDEO_FILE --run-id conc_max_1 --target-fps 0 &
    P1=$!
    sleep 2
    RUN_DIR="$VIDEO_RESULTS_DIR" python benchmark_rtsp.py --model pytorch --video $VIDEO_FILE --run-id conc_max_2 --target-fps 0 &
    P2=$!
    sleep 2
    RUN_DIR="$VIDEO_RESULTS_DIR" python benchmark_rtsp.py --model pytorch --video $VIDEO_FILE --run-id conc_max_3 --target-fps 0 &
    P3=$!
    wait $P1 $P2 $P3

    echo "Running CoreML (Python) 3x Max..."
    RUN_DIR="$VIDEO_RESULTS_DIR" python benchmark_rtsp.py --model coreml --video $VIDEO_FILE --run-id conc_max_1 --target-fps 0 &
    P1=$!
    sleep 2
    RUN_DIR="$VIDEO_RESULTS_DIR" python benchmark_rtsp.py --model coreml --video $VIDEO_FILE --run-id conc_max_2 --target-fps 0 &
    P2=$!
    sleep 2
    RUN_DIR="$VIDEO_RESULTS_DIR" python benchmark_rtsp.py --model coreml --video $VIDEO_FILE --run-id conc_max_3 --target-fps 0 &
    P3=$!
    wait $P1 $P2 $P3

    echo "Running Rust Candle 3x Max..."
    RUN_DIR="$VIDEO_RESULTS_DIR" ./target/release/benchmark_video --video $VIDEO_FILE --run-id conc_max_1 --target-fps 0 &
    P1=$!
    sleep 2
    RUN_DIR="$VIDEO_RESULTS_DIR" ./target/release/benchmark_video --video $VIDEO_FILE --run-id conc_max_2 --target-fps 0 &
    P2=$!
    sleep 2
    RUN_DIR="$VIDEO_RESULTS_DIR" ./target/release/benchmark_video --video $VIDEO_FILE --run-id conc_max_3 --target-fps 0 &
    P3=$!
    wait $P1 $P2 $P3

    echo "Running Swift CoreML 3x Max..."
    RUN_DIR="$VIDEO_RESULTS_DIR" ./benchmark_coreml yolov8s.mlmodelc $VIDEO_FILE 0 conc_max_1 &
    P1=$!
    RUN_DIR="$VIDEO_RESULTS_DIR" ./benchmark_coreml yolov8s.mlmodelc $VIDEO_FILE 0 conc_max_2 &
    P2=$!
    RUN_DIR="$VIDEO_RESULTS_DIR" ./benchmark_coreml yolov8s.mlmodelc $VIDEO_FILE 0 conc_max_3 &
    P3=$!
    wait $P1 $P2 $P3
fi

# 6. STATIC IMAGE BENCHMARKS
if [ "$RUN_IMAGE" = true ]; then
    echo "--- Phase 6: Static Image Benchmarks (Latency/Throughput) ---"
    echo "Running PyTorch (Static Images)..."
    RUN_DIR="$RUN_DIR/benchmark_results" python performance_test_mps.py --model pytorch --num-images 500 --run-id static_images

    echo "Running CoreML (Static Images)..."
    RUN_DIR="$RUN_DIR/benchmark_results" python performance_test_mps.py --model coreml --num-images 500 --run-id static_images

    echo "Running Rust (Static Images)..."
    RUN_DIR="$RUN_DIR/benchmark_results" ./target/release/performance_test --num-images 500 --run-id static_images
fi

echo "=========================================================="
echo "Benchmarks Complete. Generating Plots..."
VIDEO_RESULTS_DIR="$VIDEO_RESULTS_DIR" RUN_DIR="$RUN_DIR" python plot_video_results.py

if [ "$RUN_IMAGE" = true ]; then
    RESULTS_DIR="$RUN_DIR/benchmark_results" RUN_DIR="$RUN_DIR" python plot_results.py
fi
echo "Done."
