@echo off
echo 开始初始化码本...

set PYTHON_PATH="C:\Users\QZHYc\Anaconda3\envs\bge\python.exe"
if not exist %PYTHON_PATH% set PYTHON_PATH="C:\Users\QZHYc\Anaconda3\python.exe"

set MODEL_NAME=intfloat/multilingual-e5-base
set LANGS=ar bn en es fi fr hi id ja ko ru sw te th zh fa
set INPUT_DIM=768
set NUM_SUBVECTORS=256
set CODE_SIZE=64
set DATASET_SPLIT=train

mkdir logs 2>nul

%PYTHON_PATH% init_codebooks.py ^
    --model_name "%MODEL_NAME%" ^
    --langs %LANGS% ^
    --output_dir "project/models/pq_head_kmeans_init" ^
    --device 0 ^
    --batch_size 32 ^
    --input_dim %INPUT_DIM% ^
    --num_subvectors %NUM_SUBVECTORS% ^
    --code_size %CODE_SIZE% ^
    --dataset_split "%DATASET_SPLIT%"

echo 码本初始化完成！
pause 