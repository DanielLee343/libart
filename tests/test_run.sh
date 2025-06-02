#!/bin/bash
env=$1
build=$2
# wls=("email_workloada" "email_workloadb" "email_workloadc" "email_workloadd")
# "randint_workloada" "randint_workloadb" "randint_workloadc" "randint_workloadd" "randint_workloade" "randint_workloadla")
wls=("randint_workloada" "randint_workloadb" "randint_workloadc" "randint_workloadd" "randint_workloade" "randint_workloadla")
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
    exit 1
fi

declare -A configs=(
    [0]="-DNODE4_CUTOFF=0"
    [1]="-DNODE4_CUTOFF=1"
    [2]="-DNODE4_CUTOFF=2"
    [3]="-DNODE4_CUTOFF=3"
    [4]="-DNODE4_CUTOFF=4"
    [5]="-DNODE4_CUTOFF=5"
    [6]="-DNODE4_CUTOFF=6"
    [7]="-DNODE4_CUTOFF=7"
    [8]="-DNODE4_CUTOFF=8"
    [9]="-DNODE4_CUTOFF=9"
    [10]="-DNODE4_CUTOFF=10"
    [11]="-DNODE4_CUTOFF=11"
    [12]="-DNODE4_CUTOFF=12"
    [13]="-DNODE16_CUTOFF=0"
    [14]="-DNODE16_CUTOFF=1"
    [15]="-DNODE16_CUTOFF=2"
    [16]="-DNODE16_CUTOFF=3"
    [17]="-DNODE16_CUTOFF=4"
    [18]="-DNODE16_CUTOFF=5"
    [19]="-DNODE16_CUTOFF=6"
    [20]="-DNODE16_CUTOFF=7"
    [21]="-DNODE16_CUTOFF=8"
    [22]="-DNODE16_CUTOFF=9"
    [23]="-DNODE16_CUTOFF=10"
    [24]="-DNODE16_CUTOFF=11"
    [25]="-DNODE16_CUTOFF=12"
    [26]="-DNODE48_CUTOFF=0"
    [27]="-DNODE48_CUTOFF=1"
    [28]="-DNODE48_CUTOFF=2"
    [29]="-DNODE48_CUTOFF=3"
    [30]="-DNODE48_CUTOFF=4"
    [31]="-DNODE48_CUTOFF=5"
    [32]="-DNODE48_CUTOFF=6"
    [33]="-DNODE48_CUTOFF=7"
    [34]="-DNODE48_CUTOFF=8"
    [35]="-DNODE48_CUTOFF=9"
    [36]="-DNODE48_CUTOFF=10"
    [37]="-DNODE48_CUTOFF=11"
    [38]="-DNODE48_CUTOFF=12"
)
# [0]="-DNODE4_CUTOFF=0"
# [1]="-DNODE4_CUTOFF=3"
# [2]="-DNODE4_CUTOFF=6"
# [3]="-DNODE4_CUTOFF=9"
# [4]="-DNODE4_CUTOFF=12"
# [5]="-DNODE4_CUTOFF=15"
# [6]="-DNODE4_CUTOFF=18"
# [7]="-DNODE4_CUTOFF=21"
# [8]="-DNODE4_CUTOFF=24"
# [9]="-DNODE4_CUTOFF=27"
# [10]="-DNODE4_CUTOFF=30"
# [11]="-DNODE4_CUTOFF=33"
# [12]="-DNODE16_CUTOFF=0"
# [13]="-DNODE16_CUTOFF=3"
# [14]="-DNODE16_CUTOFF=6"
# [15]="-DNODE16_CUTOFF=9"
# [16]="-DNODE16_CUTOFF=12"
# [17]="-DNODE16_CUTOFF=15"
# [18]="-DNODE16_CUTOFF=18"
# [19]="-DNODE16_CUTOFF=21"
# [20]="-DNODE16_CUTOFF=24"
# [21]="-DNODE16_CUTOFF=27"
# [22]="-DNODE16_CUTOFF=30"
# [23]="-DNODE16_CUTOFF=33"
# [24]="-DNODE48_CUTOFF=0"
# [25]="-DNODE48_CUTOFF=3"
# [26]="-DNODE48_CUTOFF=6"
# [27]="-DNODE48_CUTOFF=9"
# [28]="-DNODE48_CUTOFF=12"
# [29]="-DNODE48_CUTOFF=15"
# [30]="-DNODE48_CUTOFF=18"
# [31]="-DNODE48_CUTOFF=21"
# [32]="-DNODE48_CUTOFF=24"
# [33]="-DNODE48_CUTOFF=27"
# [34]="-DNODE48_CUTOFF=30"
# [35]="-DNODE48_CUTOFF=33"

# [base]=""
# [cxl]="-DLEAF_CXL=1 -DNODE4_CXL=1 -DNODE16_CXL=1 -DNODE48_CXL=1 -DNODE256_CXL=1"
# [leaf]="-DLEAF_CXL=1"
# [node4]="-DNODE4_CXL=1"
# [node16]="-DNODE16_CXL=1"
# [node48]="-DNODE48_CXL=1"
# [node256]="-DNODE256_CXL=1"

# for mode in "${!configs[@]}"; do
# for mode in $(printf "%s\n" "${!configs[@]}" | sort -n); do
#     echo -e "\n========== Building ${mode} =========="
#     pushd "$LIBART_DIR" >/dev/null || exit 1
#     make clean
#     make CFLAGS_EXTRA="${configs[$mode]}"
#     sudo make install
#     popd >/dev/null || exit 1

#     echo -e "\n========== Running ${mode} tests =========="
#     for wl in "${wls[@]}"; do
#         echo -e "\n========= starting ${wl}_${mode} =========="
#         echo 3 | sudo tee /proc/sys/vm/drop_caches
#         if [[ "$wl" == "randint_workloade" ]]; then
#             ${cmd_prefix} ./scan "${wl}"
#         else
#             ${cmd_prefix} ./value "${wl}"
#         fi
#     done
# done

# dry run below:
source /home/lyuze/workspace/ClickHouse/tpch/tools.sh
for mode in $(printf "%s\n" "${!configs[@]}" | sort -n); do
    echo -e "\n========== Building ${mode} =========="
    pushd "$LIBART_DIR" >/dev/null || exit 1
    make clean
    make CFLAGS_EXTRA="${configs[$mode]}"
    sudo make install
    popd >/dev/null || exit 1

    output_dir="perf_out/${configs[$mode]}"
    [[ ! -d ${output_dir} ]] && mkdir -p ${output_dir}
    echo -e "\n========== Running ${mode} tests =========="
    for wl in "${wls[@]}"; do
        perfoutput="${output_dir}/${wl}.csv"
        bwf="${output_dir}/${wl}.bw"
        bwfigure="${output_dir}/${wl}_bw.png"
        memf="${output_dir}/${wl}.mem"
        fppng="${output_dir}/${wl}_fp.png"
        echo -e "\n========= starting ${wl}_${configs[$mode]} =========="
        echo 3 | sudo tee /proc/sys/vm/drop_caches
        if [[ "$wl" == "randint_workloade" ]]; then
            ${cmd_prefix} ./scan "${wl}" &
        else
            ${cmd_prefix} ./value "${wl}" &
        fi
        check_pid=$!
        gen_fp $check_pid raw_fp.txt $fppng
    done
done

#     ${cmd_prefix} ./scan ${wl} &
#     # sleep 5 &
# gen_bw &
#     # run_perf $check_pid $perfoutput
#     # kill_and_plot_pcm_bw $bwf $bwfigure
