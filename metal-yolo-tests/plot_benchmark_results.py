#!/usr/bin/env python3
"""Plot benchmark results for YOLOv8s on Apple Silicon M4 Max"""

import json
import glob
import os
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = os.environ.get("RESULTS_DIR", "benchmark_results")
RUN_DIR = os.environ.get("RUN_DIR", ".")

def load_results():
    """Load and aggregate results directly from JSON files."""
    json_files = glob.glob(os.path.join(RESULTS_DIR, "*.json"))
    if not json_files:
        print(f"No JSON results found in {RESULTS_DIR}")
        return None

    results = []
    for f in json_files:
        with open(f, 'r') as file:
            try:
                data = json.load(file)
                results.append(data)
            except json.JSONDecodeError:
                print(f"Warning: Could not parse {f}")

    # Aggregate data structure
    summary = {
        "single": {},
        "concurrent_3x": {}
    }

    # Helper to standardize implementation names for the chart labels
    def get_impl_name(res):
        impl = res.get("implementation", "")
        if "Rust" in impl: return "Rust (Candle)"
        if "PyTorch" in impl: return "PyTorch MPS"
        if "CoreML" in impl: return "CoreML"
        return impl

    for res in results:
        run_id = res.get("run_id", "")
        impl_name = get_impl_name(res)
        
        # Extract latency metrics
        lat_raw = res["latency_ms"]
        if isinstance(lat_raw, dict):
            lat_metrics = {
                "avg": float(lat_raw["average"]),
                "min": float(lat_raw["min"]),
                "p50": float(lat_raw["p50"]),
                "p99": float(lat_raw["p99"])
            }
        else:
            val = float(lat_raw)
            lat_metrics = {"avg": val, "min": val, "p50": val, "p99": val}
        
        fps_val = float(res["fps"])

        if "single" in run_id:
            summary["single"][impl_name] = {
                "fps": fps_val,
                "latency": lat_metrics
            }
        elif "concurrent" in run_id:
            if impl_name not in summary["concurrent_3x"]:
                summary["concurrent_3x"][impl_name] = {"fps": [], "latency": []}
            
            summary["concurrent_3x"][impl_name]["fps"].append(fps_val)
            summary["concurrent_3x"][impl_name]["latency"].append(lat_metrics)
    
    return summary

def plot_single_instance(results):
    """Plot single instance benchmark results"""
    data = results.get("single", {})
    if not data:
        print("No single instance data found.")
        return

    implementations = list(data.keys())
    implementations.sort()
    
    fps_values = [data[impl]["fps"] for impl in implementations]
    
    # Extract latency metrics
    lat_avg = [data[impl]["latency"]["avg"] for impl in implementations]
    lat_min = [data[impl]["latency"]["min"] for impl in implementations]
    lat_p50 = [data[impl]["latency"]["p50"] for impl in implementations]
    lat_p99 = [data[impl]["latency"]["p99"] for impl in implementations]
    
    # Color mapping
    color_map = {
        "CoreML": '#e74c3c',      # Red
        "PyTorch MPS": '#3498db', # Blue
        "Rust (Candle)": '#2ecc71' # Green
    }
    bar_colors = [color_map.get(impl, '#95a5a6') for impl in implementations]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # FPS Chart
    bars1 = ax1.bar(implementations, fps_values, color=bar_colors, edgecolor='black', linewidth=1.2)
    ax1.set_ylabel('FPS (higher is better)', fontsize=12)
    ax1.set_title('YOLOv8s Single Instance - FPS\n(500 images, M4 Max)', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, max(fps_values) * 1.15)
    for bar, val in zip(bars1, fps_values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{val:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # Latency Chart
    # Bars for Average
    bars2 = ax2.bar(implementations, lat_avg, color=bar_colors, edgecolor='black', linewidth=1.2, alpha=0.7, label='Average')
    
    # Error bars for Min to P99 range
    # Calculate asymmetric error bars relative to the top of the bar (Average)
    # If Average is the bar height:
    # Lower error = Average - Min
    # Upper error = P99 - Average
    y_err_lower = [avg - m for avg, m in zip(lat_avg, lat_min)]
    y_err_upper = [p99 - avg for avg, p99 in zip(lat_avg, lat_p99)]
    
    ax2.errorbar(implementations, lat_avg, yerr=[y_err_lower, y_err_upper], 
                 fmt='none', ecolor='black', capsize=5, elinewidth=1.5, markeredgewidth=1.5, label='Min-P99 Range')
    
    # Scatter marker for P50 (Median)
    # Using 'D' for diamond marker
    # We need x-coordinates for the scatter plot
    x_coords = range(len(implementations))
    ax2.scatter(x_coords, lat_p50, color='white', marker='D', edgecolor='black', s=50, zorder=3, label='P50 (Median)')

    ax2.set_ylabel('Latency (ms, lower is better)', fontsize=12)
    ax2.set_title('YOLOv8s Single Instance - Latency Distribution\n(Avg Bar + Median + Min/P99 Whisker)', fontsize=14, fontweight='bold')
    ax2.set_ylim(0, max(lat_p99) * 1.15)
    
    # Label the Average values
    for bar, val in zip(bars2, lat_avg):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (max(lat_p99)*0.02), 
                f'{val:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    out_path = os.path.join(RUN_DIR, 'benchmark_single_instance.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved {out_path}")
    plt.close()

def plot_concurrent_instances(results):
    """Plot concurrent (3 instances) benchmark results"""
    data = results.get("concurrent_3x", {})
    if not data:
        print("No concurrent data found.")
        return

    implementations = list(data.keys())
    implementations.sort()

    # Calculate averages and totals
    # For FPS: Sum of averages per instance = Total Throughput
    # For Latency: Average of the metrics across the 3 instances
    
    total_fps = []
    avg_lat_metrics = {"avg": [], "min": [], "p50": [], "p99": []}
    
    for impl in implementations:
        fps_list = data[impl]["fps"]
        lat_list = data[impl]["latency"] # list of dicts
        
        total_fps.append(sum(fps_list))
        
        # Average the latency metrics across the 3 instances
        avg_lat_metrics["avg"].append(np.mean([l["avg"] for l in lat_list]))
        # For min/max/percentiles, taking the mean of percentiles is a reasonable approximation for "typical instance behavior"
        avg_lat_metrics["min"].append(np.mean([l["min"] for l in lat_list]))
        avg_lat_metrics["p50"].append(np.mean([l["p50"] for l in lat_list]))
        avg_lat_metrics["p99"].append(np.mean([l["p99"] for l in lat_list]))

    
    color_map = {
        "CoreML": '#e74c3c',      # Red
        "PyTorch MPS": '#3498db', # Blue
        "Rust (Candle)": '#2ecc71' # Green
    }
    bar_colors = [color_map.get(impl, '#95a5a6') for impl in implementations]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Total Throughput Chart (3 instances combined)
    bars1 = ax1.bar(implementations, total_fps, color=bar_colors, edgecolor='black', linewidth=1.2)
    ax1.set_ylabel('Total FPS (3 instances combined)', fontsize=12)
    ax1.set_title('YOLOv8s Concurrent (3x) - Total Throughput\n(500 images each, M4 Max)', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, max(total_fps) * 1.15)
    for bar, val in zip(bars1, total_fps):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
                f'{val:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # Latency Chart
    bars2 = ax2.bar(implementations, avg_lat_metrics["avg"], color=bar_colors, edgecolor='black', linewidth=1.2, alpha=0.7, label='Average')
    
    # Error bars
    y_err_lower = [avg - m for avg, m in zip(avg_lat_metrics["avg"], avg_lat_metrics["min"])]
    y_err_upper = [p99 - avg for avg, p99 in zip(avg_lat_metrics["avg"], avg_lat_metrics["p99"])]
    
    ax2.errorbar(implementations, avg_lat_metrics["avg"], yerr=[y_err_lower, y_err_upper], 
                 fmt='none', ecolor='black', capsize=5, elinewidth=1.5, markeredgewidth=1.5, label='Min-P99 Range')
    
    # Marker for P50
    x_coords = range(len(implementations))
    ax2.scatter(x_coords, avg_lat_metrics["p50"], color='white', marker='D', edgecolor='black', s=50, zorder=3, label='P50 (Median)')
    
    ax2.set_ylabel('Avg Latency per Instance (ms)', fontsize=12)
    ax2.set_title('YOLOv8s Concurrent (3x) - Latency Distribution\n(Avg Bar + Median + Min/P99 Whisker)', fontsize=14, fontweight='bold')
    
    # Set Y limit based on P99 to avoid cutting off whiskers
    ax2.set_ylim(0, max(avg_lat_metrics["p99"]) * 1.15)
    
    for bar, val in zip(bars2, avg_lat_metrics["avg"]):
        # Place text slightly above the bar, but check if error bar is higher
        # For clarity, let's just label the Average value near the top of the bar
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                f'{val:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold', color='black')

    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    out_path = os.path.join(RUN_DIR, 'benchmark_concurrent_3x.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved {out_path}")
    plt.close()

def plot_comparison(results):
    """Plot comparison between single and concurrent performance"""
    single_data = results.get("single", {})
    concurrent_data = results.get("concurrent_3x", {})
    
    # Use intersection of keys to ensure we only plot common implementations
    implementations = list(set(single_data.keys()) & set(concurrent_data.keys()))
    implementations.sort()
    
    if not implementations:
        print("No overlapping implementations found for comparison.")
        return

    single_fps = [single_data[impl]["fps"] for impl in implementations]
    concurrent_avg = [np.mean(concurrent_data[impl]["fps"]) for impl in implementations]
    concurrent_total = [np.sum(concurrent_data[impl]["fps"]) for impl in implementations]
    
    x = np.arange(len(implementations))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    bars1 = ax.bar(x - width, single_fps, width, label='Single Instance FPS', color='#2ecc71', edgecolor='black')
    bars2 = ax.bar(x, concurrent_avg, width, label='Concurrent Avg FPS (per instance)', color='#3498db', edgecolor='black')
    bars3 = ax.bar(x + width, concurrent_total, width, label='Concurrent Total FPS (3x combined)', color='#e74c3c', edgecolor='black')
    
    ax.set_ylabel('FPS', fontsize=12)
    ax.set_title('YOLOv8s Performance Comparison\nSingle vs Concurrent (3 instances) on M4 Max', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(implementations, fontsize=11)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 1, 
                   f'{height:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    out_path = os.path.join(RUN_DIR, 'benchmark_comparison.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved {out_path}")
    plt.close()

if __name__ == "__main__":
    print("Loading results from JSON files...")
    results = load_results()
    if results:
        plot_single_instance(results)
        plot_concurrent_instances(results)
        plot_comparison(results)
        print("\nAll plots generated!")
