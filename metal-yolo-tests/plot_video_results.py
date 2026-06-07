import os
import json
import matplotlib.pyplot as plt
import glob
import collections
import numpy as np

# Standard Colors
COLORS = {
    'Decode': '#a6cee3',
    'Upload': '#fb9a99',
    'Compute': '#e31a1c',
    'Inference': '#1f78b4',
    'Overhead': '#b2df8a',
    'Single': '#1f78b4',
    'Concurrent': '#ff7f00'
}

VIDEO_RESULTS_DIR = os.environ.get("VIDEO_RESULTS_DIR", "video_results")
RUN_DIR = os.environ.get("RUN_DIR", ".")
os.makedirs(RUN_DIR, exist_ok=True)

def parse_results():
    # Structure: data[model][mode] = { ...metrics... }
    # We ignore FPS grouping for component plot, but keep it for others.
    # Actually, let's store everything flat list first.
    data = []
    
    files = glob.glob(os.path.join(VIDEO_RESULTS_DIR, "*.json"))
    
    for f in files:
        try:
            with open(f, "r") as fd:
                res = json.load(fd)
                res["filename"] = f
                
                base = os.path.basename(f)
                name_no_ext = os.path.splitext(base)[0]

                # Robustly get run_id: from JSON or filename
                if "run_id" not in res:
                    parts = name_no_ext.split('_')
                    if "verify_video" in name_no_ext:
                        res["run_id"] = "verify_video"
                    elif "perf_" in name_no_ext and "single" in name_no_ext:
                        try:
                            idx = name_no_ext.find("perf_")
                            res["run_id"] = name_no_ext[idx:]
                        except Exception:
                            res["run_id"] = "unknown"
                    elif "max_fps" in name_no_ext:
                        res["run_id"] = "max_fps"
                    elif "conc_max" in name_no_ext:
                        if len(parts) >= 3 and parts[-3] == "conc" and parts[-2] == "max":
                            res["run_id"] = f"conc_max_{parts[-1]}"
                    elif "conc" in name_no_ext:
                        if len(parts) >= 2 and parts[-2] == "conc":
                            res["run_id"] = f"conc_{parts[-1]}"
                    else:
                        res["run_id"] = "unknown"
                        print(f"Warning: Could not infer run_id for {f}")
                
                # Infer benchmark video FPS from filename
                if "bench_15fps" in name_no_ext:
                    res["benchmark_fps"] = 15
                elif "bench_30fps" in name_no_ext:
                    res["benchmark_fps"] = 30
                elif "bench_60fps" in name_no_ext:
                    res["benchmark_fps"] = 60
                elif "bench_90fps" in name_no_ext:
                    res["benchmark_fps"] = 90
                elif "bench_120fps" in name_no_ext:
                    res["benchmark_fps"] = 120
                else:
                    res["benchmark_fps"] = 30 # Default
                
                # Determine Mode (be strict to avoid double-counting old conc runs as max)
                rid = res.get("run_id", "")
                tfps = int(round(res.get("target_fps", 0) or 0))
                if "max" in rid:
                    res["mode"] = "max_throughput"
                elif "conc" in rid:
                    res["mode"] = "concurrent"
                elif tfps == 0 and "max" in rid:
                    res["mode"] = "max_throughput"
                else:
                    res["mode"] = "single"
                    
                # Normalize model names
                if res["model"] == "pytorch": res["model_name"] = "PyTorch (MPS)"
                elif res["model"] == "coreml": res["model_name"] = "CoreML (Python)"
                elif res["model"] == "rust_candle": res["model_name"] = "Rust (Candle)"
                elif res["model"] == "swift_coreml": res["model_name"] = "CoreML (Swift)"
                else: res["model_name"] = res["model"]
                
                if res.get("run_id", "") != "unknown":
                    data.append(res)
        except Exception as e:
            print(f"Skipping {f}: {e}")
            
    print(f"Loaded {len(data)} result files.")
    return data

def ordered_models(all_models):
    preferred = [
        "CoreML (Swift)",
        "CoreML (Python)",
        "PyTorch (MPS)",
        "Rust (Candle)",
    ]
    seen = []
    for m in preferred:
        if m in all_models:
            seen.append(m)
    for m in sorted(all_models):
        if m not in seen:
            seen.append(m)
    return seen

def plot_component_breakdown_all_fps(data):
    # Identify all Target FPS values from the data
    # We look for "single" mode runs.
    subset = [d for d in data if d["mode"] in ["single", "max_throughput"]]
    
    # Organize by Target FPS
    # fps_map[fps] = {model: stats}
    fps_map = collections.defaultdict(dict)
    
    for d in subset:
        # Determine FPS label
        fps = 0
        if d["mode"] == "max_throughput":
            fps = "Max"
        else:
            rid = d.get("run_id", "")
            tfps = int(round(d.get("target_fps", 0) or 0))
            if "perf_" in rid:
                try:
                    parts = rid.split('_') # perf, 30, single
                    fps = int(parts[1])
                except Exception:
                    fps = tfps if tfps else 30
            else:
                fps = tfps if tfps else 30
                
        fps_map[fps][d["model_name"]] = d

    target_fps_list = sorted([k for k in fps_map.keys() if isinstance(k, int)])
    if "Max" in fps_map:
        target_fps_list.append("Max")
        
    if not target_fps_list:
        print("No data for breakdown.")
        return

    # Create Subplots
    num_plots = len(target_fps_list)
    cols = 2
    rows = (num_plots + 1) // 2
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 6 * rows))
    axes = axes.flatten()
    
    for idx, fps in enumerate(target_fps_list):
        ax = axes[idx]
        stats_map = fps_map[fps]
        models = ordered_models(stats_map.keys())
        
        decodes = []
        infers = []
        uploads = [] 
        computes = [] 
        others = []
        
        has_detailed = False
        
        for m in models:
            s = stats_map[m]
            lat = s.get("avg_latency_ms", 0)
            dec = s.get("decode_avg_ms", 0)
            inf = s.get("inference_avg_ms", 0)
            up = s.get("upload_avg_ms", 0)
            comp = s.get("compute_avg_ms", 0)
            
            if up > 0 or comp > 0:
                has_detailed = True
                uploads.append(up)
                computes.append(comp)
                infers.append(0)
            else:
                uploads.append(0)
                computes.append(0)
                infers.append(inf)

            # "Other" calc
            if up > 0 or comp > 0:
                 oth = max(0, lat - dec - up - comp)
            else:
                 oth = max(0, lat - dec - inf)
            
            decodes.append(dec)
            others.append(oth)
            
        x = np.arange(len(models))
        width = 0.6
        
        p_dec = ax.bar(x, decodes, width, label='Decode/Read', color=COLORS['Decode'])
        bottoms = list(decodes)
        
        if has_detailed:
            p_up = ax.bar(x, uploads, width, bottom=bottoms, label='Upload (CPU->GPU)', color=COLORS['Upload'])
            bottoms = [b + u for b, u in zip(bottoms, uploads)]
            p_comp = ax.bar(x, computes, width, bottom=bottoms, label='Compute (GPU)', color=COLORS['Compute'])
            bottoms = [b + c for b, c in zip(bottoms, computes)]
            p_inf = ax.bar(x, infers, width, bottom=bottoms, label='Inference (Generic)', color=COLORS['Inference'])
            bottoms = [b + i for b, i in zip(bottoms, infers)]
        else:
            p_inf = ax.bar(x, infers, width, bottom=bottoms, label='Inference', color=COLORS['Inference'])
            bottoms = [b + i for b, i in zip(bottoms, infers)]
            
        p_oth = ax.bar(x, others, width, bottom=bottoms, label='System Overhead', color=COLORS['Overhead'])
        
        ax.set_ylabel('Time per Frame (ms)')
        ax.set_title(f'Latency Breakdown @ {fps} FPS Target')
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add labels
        for i in range(len(models)):
            total = decodes[i] + others[i] + infers[i] + uploads[i] + computes[i]
            ax.text(i, total + 1, f"{total:.1f}ms", ha='center', va='bottom', fontweight='bold')
            
        # Only add legend to the first plot to avoid clutter
        if idx == 0:
            ax.legend(loc='upper left')

    # Hide empty subplots
    for i in range(idx + 1, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig("video_breakdown_all_fps.png")
    print("Generated video_breakdown_all_fps.png")

def plot_fps_sweep(data):
    # Plot Latency vs Target FPS for Single Stream runs
    # Filter for "perf_X_single" AND "max_fps"
    
    # Organize: model -> {target_fps: latency}
    stats = collections.defaultdict(dict)
    
    for d in data:
        rid = d.get("run_id", "")
        
        if "perf_" in rid and "single" in rid:
            fps = int(d.get("target_fps", 30))
            stats[d["model_name"]][fps] = d.get("avg_latency_ms", 0)
        elif "max_fps" in rid or d.get("target_fps", 0) == 0:
            # Treat Max FPS as a special data point. 
            stats[d["model_name"]][120] = d.get("avg_latency_ms", 0)

    if not stats:
        print("No sweep data found.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    
    for model_name, runs in stats.items():
        sorted_fps = sorted(runs.keys())
        lats = [runs[f] for f in sorted_fps]
        
        ax.plot(sorted_fps, lats, marker='o', label=model_name)
        
    ax.set_xlabel('Target FPS (120 = Max/Unlimited)')
    ax.set_ylabel('Average Latency (ms)')
    ax.set_title('Latency vs Target FPS (Single Stream)')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Custom X ticks to show "Max"
    ticks = [30, 60, 90, 120]
    labels = ['30', '60', '90', 'Max']
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    
    out_path = os.path.join(RUN_DIR, "video_latency_sweep.png")
    plt.savefig(out_path)
    print(f"Generated {out_path}")

def plot_throughput_comparison(data):
    # Create subplots for Target 30, 60, 90, and Max
    # We want to show Processed FPS for each model in each scenario.
    # Scenarios:
    # 1. Target 30 (Single vs Concurrent)
    # 2. Target 60 (Single)
    # 3. Target 90 (Single)
    # 4. Max Throughput (Single)
    
    scenarios = [30, 60, 90, "Max"]
    
    # Organize data: stats[scenario][model] = {single: fps, concurrent: fps}
    stats = {s: collections.defaultdict(lambda: {"single": 0, "concurrent": 0}) for s in scenarios}
    
    for d in data:
        m = d["model_name"]
        rid = d.get("run_id", "")
        
        # Determine scenario based on target_fps or run_id
        scenario = None
        is_concurrent = "conc" in rid
        
        if "max" in rid:
            scenario = "Max"
        else:
            tfps = int(round(d.get("target_fps", 0)))
            if tfps == 0:
                scenario = "Max"
            elif tfps in [30, 60, 90, 120]:
                scenario = tfps if tfps != 120 else "Max"
            else:
                if "perf_30" in rid or "verify" in rid: scenario = 30
                elif "perf_60" in rid: scenario = 60
                elif "perf_90" in rid: scenario = 90
                elif "conc_max" in rid: scenario = "Max"
                elif "conc" in rid: scenario = 30
        
        if scenario is not None:
            if is_concurrent:
                # Accumulate concurrent streams
                # We store a list first to sum later? No, just add.
                # Wait, if we run the script multiple times, we might double count if we aren't careful.
                # But we parse one file per run.
                # We need to sum the FPS of all concurrent runs for a model.
                # Actually, the dict should store a list of samples to be safe, then sum.
                # But 'stats' structure above is simple.
                # Let's make concurrent a list.
                if isinstance(stats[scenario][m]["concurrent"], int):
                     stats[scenario][m]["concurrent"] = []
                stats[scenario][m]["concurrent"].append(d["processed_fps"])
            else:
                # Single stream - just take the latest or max? 
                # Should only be one per scenario usually.
                stats[scenario][m]["single"] = d["processed_fps"]

    # Prepare Plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    # Get all unique models for consistent X-axis
    all_models = ordered_models(set(d["model_name"] for d in data))
    
    for idx, scenario in enumerate(scenarios):
        ax = axes[idx]
        scenario_data = stats[scenario]
        
        single_vals = []
        conc_vals = []
        
        for model in all_models:
            s_val = scenario_data[model]["single"]
            
            # Concurrent: Sum of streams
            c_list = scenario_data[model]["concurrent"]
            if isinstance(c_list, list) and c_list:
                c_val = sum(c_list)
                print(f"  [{scenario}] {model}: Summed {len(c_list)} streams -> {c_val:.2f} FPS")
            else:
                c_val = 0
                
            single_vals.append(s_val)
            conc_vals.append(c_val)
            
        x = np.arange(len(all_models))
        width = 0.35
        
        # Only show concurrent bar if there is data
        has_conc = any(v > 0 for v in conc_vals)
        
        if has_conc:
            rects1 = ax.bar(x - width/2, single_vals, width, label='Single Stream', color=COLORS['Single'])
            rects2 = ax.bar(x + width/2, conc_vals, width, label='3x Concurrent (Total)', color=COLORS['Concurrent'])
        else:
            rects1 = ax.bar(x, single_vals, width, label='Single Stream', color=COLORS['Single'])

        ax.set_ylabel('Processed FPS')
        if scenario == "Max":
            ax.set_title(f'Throughput: Uncapped / Max Effort')
        else:
            ax.set_title(f'Throughput: Target {scenario} FPS')
            # Draw Target Line (Single)
            ax.axhline(y=scenario, color='r', linestyle='--', alpha=0.5, label=f'Target ({scenario})')
            # Draw Target Line (Concurrent 3x)
            if has_conc:
                ax.axhline(y=scenario * 3, color=COLORS['Concurrent'], linestyle=':', alpha=0.7, label=f'Target (3x {scenario})')
            
        ax.set_xticks(x)
        ax.set_xticklabels(all_models, rotation=15)
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add value labels
        def add_labels(rects):
            for rect in rects:
                height = rect.get_height()
                if height > 0:
                    ax.text(rect.get_x() + rect.get_width()/2., height + 1,
                            f'{int(height)}',
                            ha='center', va='bottom', fontsize=9)
        
        add_labels(rects1)
        if has_conc:
            add_labels(rects2)

    plt.tight_layout()
    out_path = os.path.join(RUN_DIR, "video_throughput_comparison.png")
    plt.savefig(out_path)
    print(f"Generated {out_path}")

def main():
    data = parse_results()
    plot_component_breakdown_all_fps(data)
    plot_throughput_comparison(data)
    plot_fps_sweep(data)

if __name__ == "__main__":
    main()
