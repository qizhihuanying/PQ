#!/usr/bin/env bash
# PQ量化模型训练实验脚本（方案一: 动态分配 GPU）

# 创建基本目录
mkdir -p logs/pq_experiments

# 配置参数
MODEL_NAME="intfloat/multilingual-e5-base"
INPUT_DIM=768
LOG_DIR="logs/pq_experiments"

# 定义 GPU 列表和每卡最大并发数
GPUS=(4 5 6 2)
MAX_PARALLEL=4

# 参数数组
SVS=(256)
CSS=(64)
LRS=(1e-5 3e-5 7e-5 1e-4 3e-4 7e-4 1e-6 3e-6 7e-6)
L2S=(0 1e-6 3e-6 7e-6 1e-7 3e-7 7e-7 1e-8)

# 初始化并发计数和 PID 映射
declare -A gpu_load
declare -A pid2gpu
for gpu in "${GPUS[@]}"; do
    gpu_load[$gpu]=0
done

# 启动实验函数
start_experiment() {
    local sv=$1 cs=$2 lr=$3 l2=$4 gpu=$5
    local EXP_NAME="sv=${sv}+cs=${cs}+lr=${lr}+l2=${l2}"
    local OUTPUT_DIR="project/models/pq_trained/${EXP_NAME}"
    local LOG_FILE="pq_experiment_${EXP_NAME}.log"
    mkdir -p "$OUTPUT_DIR"
    echo "在 GPU $gpu 上启动实验: sv=$sv, cs=$cs, lr=$lr, l2=$l2"
    (
        export CUDA_VISIBLE_DEVICES=$gpu
        python main.py \
            --local_model_names "$MODEL_NAME" \
            --output_dir "$OUTPUT_DIR" \
            --device 0 \
            --epochs 5 \
            --lr "$lr" \
            --l2 "$l2" \
            --batch_size 128 \
            --use_pq \
            --input_dim "$INPUT_DIM" \
            --num_subvectors "$sv" \
            --init_pq_path "project/models/pq_head_kmeans_init/pq_head_best.pt" \
            --code_size "$cs" \
            --log_dir "$LOG_DIR" \
            --log_file "$LOG_FILE"
    ) &
    local pid=$!
    pid2gpu[$pid]=$gpu
    gpu_load[$gpu]=$((gpu_load[$gpu] + 1))
}

# 检查并回收资源的函数
wait_for_available_gpu() {
    while true; do
        for pid in "${!pid2gpu[@]}"; do
            if ! kill -0 "$pid" 2>/dev/null; then
                finished_gpu=${pid2gpu[$pid]}
                gpu_load[$finished_gpu]=$((gpu_load[$finished_gpu] - 1))
                unset pid2gpu[$pid]
                return
            fi
        done
        sleep 1
    done
}

# 循环提交任务
for sv in "${SVS[@]}"; do
    for cs in "${CSS[@]}"; do
        for lr in "${LRS[@]}"; do
            for l2 in "${L2S[@]}"; do
                while :; do
                    for gpu in "${GPUS[@]}"; do
                        if (( gpu_load[$gpu] < MAX_PARALLEL )); then
                            selected_gpu=$gpu
                            break 2
                        fi
                    done
                    # 等待某个 GPU 空出来
                    wait_for_available_gpu
                done

                # 启动实验
                start_experiment $sv $cs $lr $l2 $selected_gpu
            done
        done
    done
done

# 等待所有后台任务完成
wait
echo "所有实验已完成"
