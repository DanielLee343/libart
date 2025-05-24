#!/bin/bash
env=$1
build=$2
wls=("email_workloada" "email_workloadb" "email_workloadc" "email_workloadd")
#     "randint_workloada" "randint_workloadb" "randint_workloadc" "randint_workloadd" "randint_workloade" "randint_workloadla")
# wls=("randint_workloada" "randint_workloadb" "randint_workloadc" "randint_workloadd" "randint_workloade" "randint_workloadla")
# wls=("randint_workloade")
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
    exit 1˜
fi

declare -A configs=(
    [0]="-DDEPTH_THRESH=0"
    [3]="-DDEPTH_THRESH=3"
    [6]="-DDEPTH_THRESH=6"
    [9]="-DDEPTH_THRESH=9"
    [12]="-DDEPTH_THRESH=12"
)
# [15]="-DDEPTH_THRESH=15"
# [18]="-DDEPTH_THRESH=18"
# [21]="-DDEPTH_THRESH=21"
# [24]="-DDEPTH_THRESH=24"
# [27]="-DDEPTH_THRESH=27"
# [30]="-DDEPTH_THRESH=30"
# [33]="-DDEPTH_THRESH=33"
# [base]=""
# [cxl]="-DLEAF_CXL=1 -DNODE4_CXL=1 -DNODE16_CXL=1 -DNODE48_CXL=1 -DNODE256_CXL=1"
# [leaf]="-DLEAF_CXL=1"
# [node4]="-DNODE4_CXL=1"
# [node16]="-DNODE16_CXL=1"
# [node48]="-DNODE48_CXL=1"
# [node256]="-DNODE256_CXL=1"

# [0]="-DDEPTH_THRESH=0"
# [1]="-DDEPTH_THRESH=1"
# [2]="-DDEPTH_THRESH=2"
# [3]="-DDEPTH_THRESH=3"
# [4]="-DDEPTH_THRESH=4"
# [5]="-DDEPTH_THRESH=5"
# [6]="-DDEPTH_THRESH=6"
# [7]="-DDEPTH_THRESH=7"
# [8]="-DDEPTH_THRESH=8"
# [9]="-DDEPTH_THRESH=9"
# [10]="-DDEPTH_THRESH=10"
# [11]="-DDEPTH_THRESH=11"
# [12]="-DDEPTH_THRESH=12"

# for mode in "${!configs[@]}"; do
for mode in $(printf "%s\n" "${!configs[@]}" | sort -n); do
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
        if [[ "$wl" == "randint_workloade" ]]; then
            ${cmd_prefix} ./scan "${wl}"
        else
            ${cmd_prefix} ./value "${wl}"
        fi
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

#     ${cmd_prefix} ./scan ${wl} &
#     # sleep 5 &
#     check_pid=$!
#     # gen_bw &
#     # run_perf $check_pid $perfoutput
#     # kill_and_plot_pcm_bw $bwf $bwfigure
#     gen_fp $check_pid raw_fp.txt $fppng
# done
