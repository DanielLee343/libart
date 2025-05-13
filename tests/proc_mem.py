import sys
import csv
import matplotlib.pyplot as plt
import numpy as np
import os
import math

metrics = [
    "instructions",
    "cycles",
    "CYCLE_ACTIVITY.STALLS_MEM_ANY",
    "EXE_ACTIVITY.BOUND_ON_STORES",
    "CYCLE_ACTIVITY.STALLS_L3_MISS"
]

# Map each metric to its column index
metricsmap = {
    "instructions": 1,
    "cycles": 1,
    "CYCLE_ACTIVITY.STALLS_MEM_ANY": 1,
    "EXE_ACTIVITY.BOUND_ON_STORES": 1,
    "CYCLE_ACTIVITY.STALLS_L3_MISS": 1
}


def trim_2_arr(x1, x2):
    if len(x1) < len(x2):
        x2 = x2[:len(x1)-1]
        x1 = x1[:-1]
    elif len(x2) < len(x1):
        x1 = x1[:len(x2)-1]
        x2 = x2[:-1]
    else:
        x1 = x1[:-1]
        x2 = x2[:-1]
    assert len(x1) == len(x2)
    return np.asarray(x1), np.asarray(x2)


def check_length(arrays):
    assert len(arrays) > 0
    length = len(arrays[0])
    for i in range(1, len(arrays)):
        if len(arrays[i]) != length:
            print(i, len(arrays[i]))
            return False
    return True


def read_file(file):
    res = [[] for _ in range(len(metrics))]

    with open(file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')

        for row in csv_reader:
            # Skip empty lines or improperly formatted rows
            if len(row) < 4:
                continue

            # Fourth column contains the metric name
            metric_name = row[3].strip()

            if metric_name in metricsmap:
                index = metrics.index(metric_name)
                # Second column holds the value
                value_index = metricsmap[metric_name]

                # Ensure the value is a valid number
                try:
                    # Remove commas in large numbers
                    value = float(row[value_index].replace(',', ''))
                    res[index].append(value)
                except ValueError:
                    continue  # Skip invalid values

    # for i, metric in enumerate(metrics):
        # print(f"{metric}: {res[i]}")
    lengths = [len(x) for x in res if len(x) > 0]

    if not lengths:
        return res  # Prevents crash if no valid data

    min_length = min(lengths)
    res = [res[i][:min_length] if len(res[i]) > 0 else res[i]
           for i in range(len(res))]

    assert check_length(res)
    return res


def cal_intervals(num):
    res = []
    # for filename in files:
    # f = os.path.join(directory, filename)
    f1 = os.path.join(directory, file_local)
    f2 = os.path.join(directory, file_cxl)
    assert os.path.isfile(f1)
    assert os.path.isfile(f2)
    time_series_1 = read_file(f1)
    time_series_2 = read_file(f2)
    instr_total_1 = sum(time_series_1[0])
    instr_total_2 = sum(time_series_2[0])
    res.append(int(min(instr_total_1, instr_total_2)/num))
    return res


def process_integrate_data(data_arr, instr_interval):
    instructions = data_arr[0]
    results = [[] for i in range(len(data_arr))]

    others_count = len(data_arr) - 1
    delta_instr = 0.0
    sum_instr = 0.0
    delta_others = [0.0 for i in range(others_count)]
    instr_in_prev = 0.0
    others_in_prev = [0.0 for i in range(others_count)]

    for i in range(len(instructions)):
        # Update delta_instr && delta_others[]
        delta_instr += instructions[i]
        for j in range(others_count):
            if len(data_arr[j+1]) < 1:
                continue
            delta_others[j] += data_arr[j+1][i]

        if delta_instr >= instr_interval:
            while delta_instr >= instr_interval:
                instr_in_new_interval = instr_interval - instr_in_prev
                proportion = float(instr_in_new_interval) / \
                    float(instructions[i])

                sum_instr += instr_in_new_interval + instr_in_prev
                results[0].append(sum_instr)

                # Update others && Add to results
                for j in range(others_count):
                    if len(data_arr[j+1]) < 1:
                        continue
                    results[j+1].append(proportion*data_arr[j+1]
                                        [i] + others_in_prev[j])

                delta_instr -= instr_interval
                for j in range(others_count):
                    if len(data_arr[j+1]) < 1:
                        continue
                    delta_others[j] -= results[j+1][-1]

                instr_in_prev = 0.0
                for j in range(len(others_in_prev)):
                    others_in_prev[j] = 0.0

        assert delta_instr < instr_interval
        # Update instr_in_prev && others_in_prev
        instr_in_prev = delta_instr
        for j in range(len(others_in_prev)):
            if len(data_arr[j+1]) < 1:
                continue
            others_in_prev[j] = delta_others[j]

    if delta_instr > 0:
        sum_instr += delta_instr
        results[0].append(sum_instr)
        for j in range(others_count):
            if len(data_arr[j+1]) < 1:
                continue
            results[j+1].append(delta_others[j])

    return results


def plot_one(data, output_path, plot_name, xlabel, ylabel, title):
    x, y = data[0], data[1]
    assert len(x) == len(y)
    plt.plot(x, y, marker='', linewidth=0.8)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(output_path + '/' + plot_name + '.png')
    plt.clf()


def plot_breakdown(data, output_path, plot_name, xlabel, ylabel, title):
    [instr, [store_sd, dram_sd, caches_sd, sd]] = data
    plt.plot(instr, store_sd, marker='', linewidth=1.0)
    plt.fill_between(instr, store_sd, color='lightcoral', label='store')
    plt.plot(instr, store_sd+dram_sd, marker='', linewidth=1.0)
    plt.fill_between(instr, store_sd, store_sd+dram_sd,
                     color='cornflowerblue', label='dram (load)')
    plt.plot(instr, store_sd+dram_sd+caches_sd, marker='', linewidth=1.0)
    plt.fill_between(instr, store_sd+dram_sd, store_sd+dram_sd +
                     caches_sd, color='palegreen', label='cache (load)')
    plt.plot(instr, sd, marker='', linewidth=1.0)
    plt.fill_between(instr, store_sd+dram_sd+caches_sd,
                     sd, color='gold', label='other')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.savefig(output_path + '/' + plot_name + '.png')
    plt.clf()


if __name__ == "__main__":
    directory = sys.argv[1]  # perf_out
    # vec = sys.argv[2].strip() if len(sys.argv) > 2 and sys.argv[2].strip() else ""
    vec = ""
    file_local = "email_workloada_base.csv"
    file_cxl = "email_workloada_node48.csv"
    file_name = "slowdown"
    intervals = cal_intervals(200)
    # print(intervals)

    cyc_idx = metrics.index("cycles")
    stall_mem_any_idx = metrics.index("CYCLE_ACTIVITY.STALLS_MEM_ANY")
    store_idx = metrics.index("EXE_ACTIVITY.BOUND_ON_STORES")
    stall_l3_idx = metrics.index("CYCLE_ACTIVITY.STALLS_L3_MISS")

    f1 = os.path.join(directory, file_local)
    f2 = os.path.join(directory, file_cxl)

    time_series_1 = read_file(f1)
    time_series_2 = read_file(f2)
    data_per_instr_1 = process_integrate_data(time_series_1[:], intervals[0])
    data_per_instr_2 = process_integrate_data(time_series_2[:], intervals[0])
    plot_path = file_name

    names = ["instr_"+metric for metric in metrics]
    xlabels = ["instructions" for x in metrics]
    ylabels = [metric for metric in metrics]
    titles = [file_name for x in metrics]

    output_path_sd = directory+"/" + "perf_figs"
    isExist = os.path.exists(output_path_sd)
    if not isExist:
        os.makedirs(output_path_sd)

    instr_1, instr_2 = np.asarray(
        data_per_instr_1[0]), np.asarray(data_per_instr_2[0])
    instr_1, instr_2 = trim_2_arr(instr_1, instr_2)
    length = len(instr_1)
    cyc_1, cyc_2 = np.asarray(data_per_instr_1[cyc_idx][:length]), np.asarray(
        data_per_instr_2[cyc_idx][:length])
    llc_stall_1, llc_stall_2 = np.asarray(data_per_instr_1[stall_l3_idx][:length]), np.asarray(
        data_per_instr_2[stall_l3_idx][:length])
    mem_stall_1, mem_stall_2 = np.asarray(data_per_instr_1[stall_mem_any_idx][:length]), np.asarray(
        data_per_instr_2[stall_mem_any_idx][:length])
    store_1, store_2 = np.asarray(data_per_instr_1[store_idx][:length]), np.asarray(
        data_per_instr_2[store_idx][:length])

    sd = (cyc_2-cyc_1)/cyc_1
    dram_sd = (llc_stall_2-llc_stall_1)/cyc_1
    caches_sd = ((mem_stall_2-llc_stall_2) - (mem_stall_1-llc_stall_1))/cyc_1
    store_sd = (store_2-store_1)/cyc_1
    # print("dram_sd:", dram_sd)
    # print("caches_sd:", caches_sd)
    # print("store_sd:", store_sd)

    print(vec)
    plot_one([instr_1, sd], output_path_sd, "slowdown_{}".format(vec),
             "instructions", "slowdown", "Overall slowdown {}".format(vec))

    plot_breakdown([instr_1, [store_sd, dram_sd, caches_sd, sd]],
                   output_path_sd, "breakdown_{}".format(vec), "instructions", "slowdown", "Slowdown = dram (load) + cache (load) + store {}".format(vec))
