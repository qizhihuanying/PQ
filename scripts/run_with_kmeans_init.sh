#!/usr/bin/env bash

# 创建基本目录
mkdir -p logs/dpq_kmeans_init_multiattention

# 配置参数
MODEL_NAME="intfloat/multilingual-e5-base"
CORPUS_PATH="datasets/miracl/zh/train/corpus.jsonl"
INPUT_DIM=768
LANGS="zh"
LOG_DIR="logs/dpq_kmeans_init_multiattention"

# 定义参数数组
GPU_IDS=(7 3 3 3 3)
SVS=(256)
CSS=(64)
ATTENTION_LRS=(1e-6 5e-6 1e-5 5e-5 1e-4)
LRS=(2e-5)
ATTENTION_HEADS=(4 8 16 32)

NUM_GPUS=${#GPU_IDS[@]}
index=0

# 按参数组合循环执行实验
for sv in "${SVS[@]}"; do
    for cs in "${CSS[@]}"; do
        for attn_lr in "${ATTENTION_LRS[@]}"; do
            for lr in "${LRS[@]}"; do
                for num_heads in "${ATTENTION_HEADS[@]}"; do
                    # 等待直到有空闲GPU
                    while [ "$(jobs -p | wc -l)" -ge "$NUM_GPUS" ]; do
                        sleep 1
                    done

                    # 获取当前设备ID
                    gpu_id=${GPU_IDS[$((index % NUM_GPUS))]}
                    
                    # 为每个参数组合创建唯一目录，添加多头注意力参数
                    EXPERIMENT_NAME="sv${sv}_cs${cs}_lr${lr}_attn_lr${attn_lr}_heads${num_heads}"
                    INIT_DIR="project/models/pq_head_kmeans_init/${EXPERIMENT_NAME}"
                    OUTPUT_DIR="project/models/pq_head_kmeans_trained/${EXPERIMENT_NAME}"
                    log_filename="kmeans_init_${EXPERIMENT_NAME}.log"
                    
                    # 创建实验目录
                    mkdir -p "$INIT_DIR"
                    mkdir -p "$OUTPUT_DIR"

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
                    ) &

                    # 更新索引
                    ((index++))
                done
            done
        done
    done
done

wait
echo "所有实验已完成"