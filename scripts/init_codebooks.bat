@echo off
set PYTHON_PATH="C:\Users\QZHYc\Anaconda3\envs\bge\python.exe"
if not exist %PYTHON_PATH% set PYTHON_PATH="C:\Users\QZHYc\Anaconda3\python.exe"

%PYTHON_PATH% init_codebooks.py ^
    --model_name "Alibaba-NLP/gte-multilingual-base" ^
    --output_dir "project/pq_head_initialized" ^
    --num_subvectors 256 ^
    --code_size 64 ^
    --device 0 ^
    --batch_size 32 ^
    --langs ar bn en es fi fr hi id ja ko ru sw te th zh fa ^
    --dataset_split train
