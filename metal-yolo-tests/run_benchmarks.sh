#!/bin/bash
# YOLOv8s Benchmark Runner
# Tests: Rust/Candle (Metal), PyTorch (MPS), CoreML (ANE/GPU)

set -e

RUN_DIR="runs/$(date +%Y%m%d-%H%M%S)"
RESULTS_DIR="$RUN_DIR/benchmark_results"
NUM_IMAGES=500
mkdir -p "$RESULTS_DIR"
export RUN_DIR

# Activate conda environment for Python tests
CONDA_ENV="py3-12"
CONDA_BASE="/opt/anaconda3"

echo "========================================"
echo "YOLOv8s Benchmark Suite"
echo "Images: $NUM_IMAGES"
echo "========================================"

# Clean old results for this run dir only
rm -f "$RESULTS_DIR"/*.json || true

###########################################
# PHASE 1: Individual Runs (1 instance each)
###########################################
echo ""
echo "========================================"
echo "PHASE 1: Individual Runs"
echo "========================================"

# Rust/Candle Individual
echo ""
echo "--- Rust/Candle Individual Run ---"
RUN_DIR="$RESULTS_DIR" ./target/release/performance_test --num-images $NUM_IMAGES --run-id "rust_single"
sleep 2

# PyTorch MPS Individual  
echo ""
echo "--- PyTorch MPS Individual Run ---"
source "$CONDA_BASE/bin/activate" "$CONDA_ENV"
KMP_DUPLICATE_LIB_OK=TRUE RUN_DIR="$RESULTS_DIR" python3 performance_test_mps.py --model pytorch --num-images $NUM_IMAGES --run-id "pytorch_single"
sleep 2

# CoreML Individual
echo ""
echo "--- CoreML Individual Run ---"
KMP_DUPLICATE_LIB_OK=TRUE RUN_DIR="$RESULTS_DIR" python3 performance_test_mps.py --model coreml --num-images $NUM_IMAGES --run-id "coreml_single"
sleep 2

###########################################
# PHASE 2: Concurrent Runs (3 instances each)
###########################################
echo ""
echo "========================================"
echo "PHASE 2: Concurrent Runs (3 instances)"
echo "========================================"

# Rust/Candle Concurrent (3 instances)
echo ""
echo "--- Rust/Candle Concurrent Run (3 instances) ---"
RUN_DIR="$RESULTS_DIR" ./target/release/performance_test --num-images $NUM_IMAGES --run-id "rust_concurrent_1" &
PID1=$!
RUN_DIR="$RESULTS_DIR" ./target/release/performance_test --num-images $NUM_IMAGES --run-id "rust_concurrent_2" &
PID2=$!
RUN_DIR="$RESULTS_DIR" ./target/release/performance_test --num-images $NUM_IMAGES --run-id "rust_concurrent_3" &
PID3=$!
wait $PID1 $PID2 $PID3
echo "All Rust instances completed."
sleep 5

# PyTorch MPS Concurrent (3 instances)
echo ""
echo "--- PyTorch MPS Concurrent Run (3 instances) ---"
KMP_DUPLICATE_LIB_OK=TRUE RUN_DIR="$RESULTS_DIR" python3 performance_test_mps.py --model pytorch --num-images $NUM_IMAGES --run-id "pytorch_concurrent_1" &
PID1=$!
KMP_DUPLICATE_LIB_OK=TRUE RUN_DIR="$RESULTS_DIR" python3 performance_test_mps.py --model pytorch --num-images $NUM_IMAGES --run-id "pytorch_concurrent_2" &
PID2=$!
KMP_DUPLICATE_LIB_OK=TRUE RUN_DIR="$RESULTS_DIR" python3 performance_test_mps.py --model pytorch --num-images $NUM_IMAGES --run-id "pytorch_concurrent_3" &
PID3=$!
wait $PID1 $PID2 $PID3
echo "All PyTorch instances completed."
sleep 5

# CoreML Concurrent (3 instances)
echo ""
echo "--- CoreML Concurrent Run (3 instances) ---"
KMP_DUPLICATE_LIB_OK=TRUE RUN_DIR="$RESULTS_DIR" python3 performance_test_mps.py --model coreml --num-images $NUM_IMAGES --run-id "coreml_concurrent_1" &
PID1=$!
KMP_DUPLICATE_LIB_OK=TRUE RUN_DIR="$RESULTS_DIR" python3 performance_test_mps.py --model coreml --num-images $NUM_IMAGES --run-id "coreml_concurrent_2" &
PID2=$!
KMP_DUPLICATE_LIB_OK=TRUE RUN_DIR="$RESULTS_DIR" python3 performance_test_mps.py --model coreml --num-images $NUM_IMAGES --run-id "coreml_concurrent_3" &
PID3=$!
wait $PID1 $PID2 $PID3
echo "All CoreML instances completed."

###########################################
echo ""
echo "========================================"
echo "Generating Reports"
echo "========================================"

# Generate plots directly from JSON results
RESULTS_DIR="$RESULTS_DIR" RUN_DIR="$RUN_DIR" python3 plot_benchmark_results.py

echo ""
echo "All benchmarks completed!"
echo "Results saved to $RESULTS_DIR/"
ls -la "$RESULTS_DIR/"
