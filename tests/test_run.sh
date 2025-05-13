#!/bin/bash
env=$1
build=$2
# wls=("email_workloada" "email_workloadb" "email_workloadc" "email_workloadd" "email_workloadla"
#     "randint_workloada" "randint_workloadb" "randint_workloadc" "randint_workloadd" "randint_workloade" "randint_workloadla")
wls=("randint_workloadd")
LIBART_DIR=$HOME/workspace/libart

if [ "$env" = "cxl" ]; then
    cmd_prefix="numactl --physcpubind=0 -m 1 -- "
elif [ "$env" = "base" ]; then
    cmd_prefix="numactl --physcpubind=0 -m 0 -- "
elif [ "$env" = "interleave" ]; then
    cmd_prefix="numactl --physcpubind=0 --interleave=all -- "
elif [ "$env" = "cpu0" ]; then
    cmd_prefix="numactl --physcpubind=0 -- "
else
    echo "wrong env, pls try again!"
    exit 1
fi

if [[ -z "$LIBART_DIR" ]]; then
    echo "Error: LIBART_DIR not set."
    exit 1
fi

if [[ ${#wls[@]} -eq 0 ]]; then
    echo "Error: wls array is empty."
    exit 1
fi

declare -A configs=(
    [base]=""
    [cxl]="-DLEAF_CXL=1 -DNODE4_CXL=1 -DNODE16_CXL=1 -DNODE48_CXL=1 -DNODE256_CXL=1"
    [leaf]="-DLEAF_CXL=1"
    [node4]="-DNODE4_CXL=1"
    [node16]="-DNODE16_CXL=1"
    [node48]="-DNODE48_CXL=1"
    [node256]="-DNODE256_CXL=1"
)

for mode in "${!configs[@]}"; do
    echo -e "\n========== Building ${mode} =========="
    pushd "$LIBART_DIR" >/dev/null || exit 1
    make clean
    make CFLAGS_EXTRA="${configs[$mode]}"
    sudo make install
    popd >/dev/null || exit 1

    echo -e "\n========== Running ${mode} tests =========="
    for wl in "${wls[@]}"; do
        echo -e "\n========= starting ${wl}_${mode} =========="
        echo 3 | sudo tee /proc/sys/vm/drop_caches
        ${cmd_prefix} ./value "${wl}"
    done
done

# dry run below:
# source /home/lyuze/workspace/ClickHouse/tpch/tools.sh
# output_dir="perf_out"
# [[ ! -d ${output_dir} ]] && mkdir -p ${output_dir}
# for wl in "${wls[@]}"; do
#     echo -e "\n========= starting ${wl}_${build} =========="
#     echo 3 | sudo tee /proc/sys/vm/drop_caches
#     perfoutput=${output_dir}/${build}.csv
#     bwf=${output_dir}/${build}.bw
#     bwfigure=${output_dir}/${build}_bw.png
#     memf=${output_dir}/${build}.mem
#     fppng=${output_dir}/${build}_fp.png

#     ${cmd_prefix} ./value ${wl} &
#     # sleep 5 &
#     check_pid=$!
#     # gen_bw &
#     run_perf $check_pid $perfoutput
#     # kill_and_plot_pcm_bw $bwf $bwfigure
#     # gen_fp $check_pid raw_fp.txt $fppng
# done
