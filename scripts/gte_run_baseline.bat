@echo off
set PYTHON_PATH="C:\Users\QZHYc\Anaconda3\envs\bge\python.exe"
if not exist %PYTHON_PATH% set PYTHON_PATH="C:\Users\QZHYc\Anaconda3\python.exe"

%PYTHON_PATH% main.py ^
    --local_model_names Alibaba-NLP/gte-multilingual-base ^
    --output_dir project/models/gte_baseline ^
    --device 0 ^
    --epochs 0 ^
    --lr 2e-5 ^
    --l2 0.0 ^
    --batch_size 32 ^
    --train_sample_ratio 0.0 ^
    --test_ratio 1.0 ^
    --base_trainable_layers 0 ^
    --input_dim 768 ^
    --num_subvectors 256 ^
    --code_size 64 ^
    --dataset miracl ^
    --log_dir logs/baseline ^
    --log_file gte_baseline.log 