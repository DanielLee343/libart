import os
import csv
import sys
import pandas as pd
import numpy as np
q = sys.argv[1]
directory = f"perf_out/{q}"
perf_figs = os.path.join(directory, "perf_figs")

if not os.path.exists(perf_figs):
    os.makedirs(perf_figs)
    print(f"Directory '{perf_figs}' created.")
else:
    print(f"Directory '{perf_figs}' already exists.")

metrics = ["instructions", "cycles",
           #    "branches", "branch-misses",
           "L1_HIT", "L1_MISS", "FB_HIT", "L2_HIT", "L2_MISS", "L3_HIT", "L3_MISS",
           "STALLS_L1D_MISS", "STALLS_L2_MISS", "STALLS_L3_MISS", "STALLS_MEM_ANY", "BOUND_ON_STORES"]
data = {metric: {} for metric in metrics}
mlp_data = {}
ipc_data = {}
bw_data = {}
miss_rates = {
    "L1_MISS_RATE": {},
    "L2_MISS_RATE": {},
    "L3_MISS_RATE": {},
}

# vec_order = ["512", "1024", "2048", "4096", "8192"]
vec_order = ["base", "node48", "cxl"]

for filename in os.listdir(directory):
    if filename.endswith(".csv"):
        # parts = filename.split("_")
        # db, vec = parts[0], parts[1]  # tpch100 1024.csv
        vec = filename.removesuffix(".csv")  # 1024.csv -> 1024

        with open(os.path.join(directory, filename), "r") as file:
            reader = csv.reader(file)
            temp_data = {metric: [] for metric in metrics}
            temp_mlp = []
            temp_mlp_cycles = []
            temp_ipc = []
            for row in reader:
                if not row or row[0].startswith("#"):
                    continue
                for metric in metrics:
                    if row[3] in (metric, f"MEM_LOAD_RETIRED.{metric}", f"CYCLE_ACTIVITY.{metric}", f"EXE_ACTIVITY.{metric}"):
                        if row[1] == '<not counted>':
                            continue
                        temp_data[metric].append(int(row[1]))

                if row[3] == "OFFCORE_REQUESTS_OUTSTANDING.L3_MISS_DEMAND_DATA_RD":
                    if row[1] == '<not counted>':
                        continue
                    temp_mlp.append(int(row[1]))
                elif row[3] == "OFFCORE_REQUESTS_OUTSTANDING.CYCLES_WITH_L3_MISS_DEMAND_DATA_RD":
                    if row[1] == '<not counted>':
                        continue
                    temp_mlp_cycles.append(int(row[1]))

                # Extract IPC values
                if row[3] == "instructions":
                    if row[-2].strip():
                        temp_ipc.append(float(row[-2]))
            for metric in metrics:
                if temp_data[metric]:
                    data[metric][vec] = np.sum(temp_data[metric])

            if temp_mlp and temp_mlp_cycles:
                mlp_data[vec] = np.sum(temp_mlp) / np.sum(temp_mlp_cycles)

            if temp_ipc:
                ipc_data[vec] = np.mean(temp_ipc)

# Process .mem files for BW data
for filename in os.listdir(directory):
    if filename.endswith(".mem"):
        parts = filename.split("_")
        # if len(parts) < 3:
        #     continue  # Skip files with unexpected format
        db, vec = parts[0], parts[1]
        vec = vec.removesuffix(".mem")  # 1024.mem -> 1024

        # if vec not in vec_order or env not in ["base", "cxl"]:
        #     continue

        with open(os.path.join(directory, filename), "r") as file:
            for line in file:
                if line.startswith("DRAM_BW:"):
                    bw_data[vec] = float(line.split(":")[1].strip())
                # elif env == "cxl" and line.startswith("CXL_BW:"):
                #     bw_data[env][vec] = float(line.split(":")[1].strip())

for vec in vec_order:
    for level in ["L1", "L2", "L3"]:
        hit = data.get(f"{level}_HIT", {}).get(vec)
        miss = data.get(f"{level}_MISS", {}).get(vec)
        if hit is not None and miss is not None and (hit + miss) > 0:
            rate = miss / (hit + miss)
            miss_rates[f"{level}_MISS_RATE"][vec] = round(rate, 6)


def save_metric_to_csv(metric_name, metric_data, output_name):
    df = pd.DataFrame(
        {vec: metric_data.get(vec, None) for vec in vec_order},
        index=[metric_name]
    ).T
    df.columns = [metric_name]
    output_file = os.path.join(perf_figs, output_name)
    df.index.name = "vec"
    df.to_csv(output_file)
    print(f"{metric_name} data saved to {output_file}")


for metric in metrics:
    save_metric_to_csv(metric, data[metric], f"{metric}.csv")
for metric_name, metric_data in miss_rates.items():
    save_metric_to_csv(metric_name, metric_data, f"{metric_name}.csv")

save_metric_to_csv("MLP", mlp_data, "MLP.csv")
save_metric_to_csv("IPC", ipc_data, "IPC.csv")
save_metric_to_csv("BW", bw_data, "BW.csv")
