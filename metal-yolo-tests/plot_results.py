import json
import matplotlib.pyplot as plt
import os
import numpy as np
import glob

RESULTS_DIR = os.environ.get("RESULTS_DIR", ".")
RUN_DIR = os.environ.get("RUN_DIR", ".")

def load_results():
    results = []
    
    # Glob all results_*.json files
    files = glob.glob(os.path.join(RESULTS_DIR, "results_*.json"))
    
    for f in files:
        try:
            with open(f, "r") as fd:
                data = json.load(fd)
                if isinstance(data, list):
                    results.extend(data)
                else:
                    # Ensure it has required fields
                    if "model" in data and "fps" in data:
                        results.append(data)
        except Exception as e:
            print(f"Error loading {f}: {e}")
                
    return results

def plot_performance(results):
    if not results:
        print("No results to plot.")
        return

    # Group by model
    models = sorted(list(set(r["model"] for r in results)))
    implementations = sorted(list(set(r["implementation"] for r in results)))
    
    fps_data = {impl: [] for impl in implementations}
    latency_data = {impl: [] for impl in implementations}
    
    for model in models:
        for impl in implementations:
            # Find match
            match = next((r for r in results if r["model"] == model and r["implementation"] == impl), None)
            
            if match:
                fps_data[impl].append(match.get("fps", 0))
                
                # Handle latency (nested or flat)
                if "latency_ms" in match and isinstance(match["latency_ms"], dict):
                    latency_data[impl].append(match["latency_ms"].get("average", 0))
                else:
                    latency_data[impl].append(match.get("latency", 0))
            else:
                fps_data[impl].append(0)
                latency_data[impl].append(0)

    x = np.arange(len(models))
    width = 0.35
    
    # Plot FPS
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    rects = []
    for i, impl in enumerate(implementations):
        offset = width * i - width/2 * (len(implementations) - 1)
        rects.append(ax1.bar(x + offset, fps_data[impl], width, label=impl))

    ax1.set_ylabel('FPS')
    ax1.set_title('YOLO Performance Comparison (FPS) - Higher is Better\nF32 Precision, Metal/MPS')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models)
    ax1.legend()
    
    for rect in rects:
        ax1.bar_label(rect, padding=3, fmt='%.2f')

    fig.tight_layout()
    out_path = os.path.join(RUN_DIR, 'performance_comparison_fps.png')
    plt.savefig(out_path)
    print(f"Saved {out_path}")
    
    # Plot Latency
    fig, ax2 = plt.subplots(figsize=(10, 6))
    
    rects = []
    for i, impl in enumerate(implementations):
        offset = width * i - width/2 * (len(implementations) - 1)
        rects.append(ax2.bar(x + offset, latency_data[impl], width, label=impl))

    ax2.set_ylabel('Latency (ms)')
    ax2.set_title('YOLO Performance Comparison (Latency) - Lower is Better\nF32 Precision, Metal/MPS')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models)
    ax2.legend()
    
    for rect in rects:
        ax2.bar_label(rect, padding=3, fmt='%.2f')

    fig.tight_layout()
    out_path = os.path.join(RUN_DIR, 'performance_comparison_latency.png')
    plt.savefig(out_path)
    print(f"Saved {out_path}")

if __name__ == "__main__":
    results = load_results()
    print("Loaded results:", json.dumps(results, indent=2))
    plot_performance(results)
