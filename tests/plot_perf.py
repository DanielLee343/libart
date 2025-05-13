import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

q = sys.argv[1]
csv_dir = f"perf_out/{q}/perf_figs"

grouped_metrics = {
    "mlp_ipc": ["MLP", "IPC"],
    # "branch": ["branches", "branch-misses"],
    "cache": ["L2_HIT", "L2_MISS", "L3_HIT", "L3_MISS"],
    "stalls": ["STALLS_L1D_MISS", "STALLS_L2_MISS", "STALLS_L3_MISS", "STALLS_MEM_ANY", "BOUND_ON_STORES"],
    "instr_cyc": ["instructions", "cycles"],
    "miss_rates": ["L1_MISS_RATE", "L2_MISS_RATE", "L3_MISS_RATE"],
    # "normed_cache": ["norm_L1_MISS", "norm_FB_HIT", "norm_L2_HIT", "norm_L2_MISS", "norm_L3_HIT", "norm_L3_MISS"],
    # "normed_stalls": ["norm_STALLS_L1D_MISS", "norm_STALLS_L2_MISS", "norm_STALLS_L3_MISS", "norm_STALLS_MEM_ANY", "norm_BOUND_ON_STORES"],
    # "normed_branches": ["norm_branches", "norm_branch-misses"],
    "bandwidth": ["BW"]
}

for group_name, metrics in grouped_metrics.items():
    plt.figure(figsize=(12, 6))
    for metric in metrics:
        filepath = os.path.join(csv_dir, f"{metric}.csv")
        if not os.path.isfile(filepath):
            print(f"Warning: {filepath} not found. Skipping.")
            continue

        df = pd.read_csv(filepath)
        vecs = df.iloc[:, 0].astype(str)
        values = df.iloc[:, 1]

        plt.plot(vecs, values, label=metric)

    plt.xlabel("setup")
    plt.ylabel("Metric Value")
    plt.title(f"{group_name.replace('_', ' ').upper()} Metrics")
    plt.legend(loc="upper right", fontsize="large")
    plt.grid(True)
    plt.tight_layout()
    out_path = os.path.join(csv_dir, f"{group_name}_metrics.png")
    plt.savefig(out_path)
    print(f"Saved plot to {out_path}")
    plt.close()
