#!/bin/bash
wl="email_workloada"
FP_FILE=total_fp.txt
NUMA_FP_FILE=numa_fp.txt

# source /home/lyuze/workspace/ClickHouse/tpch/tools.sh
output_dir="perf_out/fp"
[[ ! -d ${output_dir} ]] && mkdir -p ${output_dir}
rm -rf $FP_FILE
rm -rf $NUMA_FP_FILE
echo 3 | sudo tee /proc/sys/vm/drop_caches
numactl -m1 -- cat /mnt/data_bk/zipfian/load_email_workloada >/dev/null
numactl -m1 -- cat /mnt/data_bk/zipfian/txn_email_workloada >/dev/null
{
    fppng="${output_dir}/${wl}_fp.png"
    numactl --physcpubind=0 -- ./value "${wl}" &
    check_pid=$!
    while [ -d "/proc/${check_pid}" ]; do
        ps -o rss= -p "$check_pid" >>$FP_FILE
        grep -o 'N[01]=[0-9]\+' /proc/${check_pid}/numa_maps |
            awk -F'[=N]' '{
            count[$2]+=$3
        } END {
            printf "%d %d\n", count[0], count[1]
        }' >>"$NUMA_FP_FILE"
        sleep 0.5
    done
    awk '{gsub(/[[:space:]]/, "", $0); if ($NF > max) max = $NF} END {printf "Max RSS: %.2f MB\n", max/1024}' $FP_FILE
    awk '{
        node0[$1]++
        node1[$2]++
        if ($1 > max0) max0 = $1
        if ($2 > max1) max1 = $2
    } END {
        printf "Max Node 0 RSS: %.2f MB\n", max0 * 4 / 1024
        printf "Max Node 1 RSS: %.2f MB\n", max1 * 4 / 1024
    }' "$NUMA_FP_FILE"
}
