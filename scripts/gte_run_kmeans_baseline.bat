@echo off
set PYTHON_PATH="C:\Users\QZHYc\Anaconda3\envs\bge\python.exe"
if not exist %PYTHON_PATH% set PYTHON_PATH="C:\Users\QZHYc\Anaconda3\python.exe"

%PYTHON_PATH% main.py ^
    --local_model_names Alibaba-NLP/gte-multilingual-base ^
    --output_dir project/models/gte_kmeans_baseline ^
    --device 0 ^
    --epochs 0 ^
    --lr 2e-5 ^
    --l2 0.0 ^
    --batch_size 32 ^
    --train_sample_ratio 0 ^
    --val_ratio 0 ^
    --test_ratio 1.0 ^
    --base_trainable_layers 0 ^
    --input_dim 768 ^
    --num_subvectors 256 ^
    --code_size 64 ^
    --use_pq ^
    --dataset miracl ^
    --log_dir logs/kmeans_baseline ^
    --log_file kmeans_baseline.log ^
    --init_pq_path project/pq_head_initialized/gte-multilingual-base/pq_head_cs64_ns256.pt/pq_head_best.pt