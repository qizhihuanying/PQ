#!/usr/bin/env bash

# 创建基本目录
mkdir -p logs/dpq_kmeans_init_multiattention

# 配置参数
MODEL_NAME="intfloat/multilingual-e5-base"
CORPUS_PATH="datasets/miracl/zh/train/corpus.jsonl"
INPUT_DIM=768
LANGS="zh"
LOG_DIR="logs/dpq_kmeans_init_multiattention"

# 定义参数数组 - 保持原始配置
GPU_IDS=(7 3 3 3 3)
SVS=(256)
CSS=(64)
ATTENTION_LRS=(1e-6 5e-6 1e-5 5e-5 1e-4)
LRS=(2e-5)
ATTENTION_HEADS=(4 8 16 32)

# 获取唯一的GPU ID
unique_gpus=($(echo "${GPU_IDS[@]}" | tr ' ' '\n' | sort -u | tr '\n' ' '))

# 每个GPU最多同时运行的任务数
MAX_TASKS_PER_GPU=4

# 生成所有参数组合
combinations=()
for sv in "${SVS[@]}"; do
    for cs in "${CSS[@]}"; do
        for attn_lr in "${ATTENTION_LRS[@]}"; do
            for lr in "${LRS[@]}"; do
                for num_heads in "${ATTENTION_HEADS[@]}"; do
                    combinations+=("$sv $cs $attn_lr $lr $num_heads")
                done
            done
        done
    done
done

# 为每个GPU创建任务队列
declare -A gpu_tasks
for gpu_id in "${unique_gpus[@]}"; do
    gpu_tasks[$gpu_id]=0  # 初始化每个GPU的运行任务计数为0
done

# 处理所有组合
total_combinations=${#combinations[@]}
processed=0

echo "总共需要处理 $total_combinations 个参数组合"

while [ $processed -lt $total_combinations ]; do
    for gpu_id in "${unique_gpus[@]}"; do
        # 检查此GPU上运行的任务数量
        current_tasks=${gpu_tasks[$gpu_id]}
        
        # 如果此GPU可以运行更多任务，且还有组合需要处理
        if [ $current_tasks -lt $MAX_TASKS_PER_GPU ] && [ $processed -lt $total_combinations ]; then
            # 获取当前组合
            combination=${combinations[$processed]}
            read -r sv cs attn_lr lr num_heads <<< "$combination"
            
            # 为每个参数组合创建唯一目录
            EXPERIMENT_NAME="sv${sv}_cs${cs}_lr${lr}_attn_lr${attn_lr}_heads${num_heads}"
            INIT_DIR="project/models/pq_head_kmeans_init/${EXPERIMENT_NAME}"
            OUTPUT_DIR="project/models/pq_head_kmeans_trained/${EXPERIMENT_NAME}"
            log_filename="kmeans_init_${EXPERIMENT_NAME}.log"
            
            # 创建实验目录
            mkdir -p "$INIT_DIR"
            mkdir -p "$OUTPUT_DIR"

            # 生成一个临时标记文件，用于跟踪进程
            task_id="task_${gpu_id}_${processed}"
            flag_file="/tmp/${task_id}.running"
            touch "$flag_file"

            # 后台启动任务
            (
                echo "在设备 $gpu_id 上启动实验: sv=$sv, cs=$cs, lr=$lr, attention_lr=$attn_lr, attention_heads=$num_heads"
                
                # python init_codebooks.py \
                #     --model_name "$MODEL_NAME" \
                #     --corpus_path "$CORPUS_PATH" \
                #     --output_dir "$INIT_DIR" \
                #     --device "$gpu_id" \
                #     --sample_ratio 1.0 \
                #     --batch_size 32 \
                #     --input_dim "$INPUT_DIM" \
                #     --num_subvectors "$sv" \
                #     --code_size "$cs"

                python main.py \
                    --local_model_names "$MODEL_NAME" \
                    --output_dir "$OUTPUT_DIR" \
                    --device "$gpu_id" \
                    --epochs 20 \
                    --lr "$lr" \
                    --attention_lr "$attn_lr" \
                    --num_attention_heads "$num_heads" \
                    --l2 0.0 \
                    --batch_size 32 \
                    --use_pq \
                    --input_dim "$INPUT_DIM" \
                    --num_subvectors "$sv" \
                    --code_size "$cs" \
                    --langs "$LANGS" \
                    --init_pq_path "project/models/pq_head_kmeans_init/pq_head_best.pt" \
                    --log_dir "$LOG_DIR" \
                    --log_file "$log_filename"
                
                # 任务完成后删除标记文件
                rm -f "$flag_file"
                echo "设备 $gpu_id 上的实验已完成: sv=$sv, cs=$cs, lr=$lr, attention_lr=$attn_lr, attention_heads=$num_heads"
            ) &
            
            # 更新任务计数
            gpu_tasks[$gpu_id]=$((current_tasks + 1))
            echo "设备 $gpu_id 上添加了新任务，当前任务数: ${gpu_tasks[$gpu_id]}/$MAX_TASKS_PER_GPU"
            
            # 更新已处理的组合数
            processed=$((processed+1))
        fi
    done
    
    # 等待片刻，给进程一些时间来完成
    sleep 5
    
    # 更新每个GPU的任务计数
    for gpu_id in "${unique_gpus[@]}"; do
        # 计算此GPU上正在运行的任务数
        running_count=$(ls /tmp/task_${gpu_id}_*.running 2>/dev/null | wc -l)
        gpu_tasks[$gpu_id]=$running_count
    done
done

# 等待所有后台任务完成
wait
echo "所有实验已完成"