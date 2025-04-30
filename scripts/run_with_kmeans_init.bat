@echo off
mkdir logs/dpq_kmeans_init_attention_no_norm 2>nul

set PYTHON_PATH="C:\Users\QZHYc\Anaconda3\envs\bge\python.exe"
if not exist %PYTHON_PATH% set PYTHON_PATH="C:\Users\QZHYc\Anaconda3\python.exe"

set MODEL_NAME=intfloat/multilingual-e5-base
set LANGS=ar bn en es fi fr hi id ja ko ru sw te th zh fa
set INPUT_DIM=768
set LOG_DIR=logs/dpq_kmeans_init_attention_no_norm
set DATASET_SPLIT=dev

set SV_OPTIONS=256
set CS_OPTIONS=64
set ATTENTION_LR_OPTIONS=3e-6


for %%a in (0 1) do (
    for %%s in (%SV_OPTIONS%) do (
        for %%c in (%CS_OPTIONS%) do (
            for %%l in (%ATTENTION_LR_OPTIONS%) do (

                @REM %PYTHON_PATH% init_codebooks.py ^
                @REM     --model_name "%MODEL_NAME%" ^
                @REM     --langs %LANGS% ^
                @REM     --output_dir "project/models/pq_head_kmeans_init" ^
                @REM     --device 0 ^
                @REM     --batch_size 32 ^
                @REM     --input_dim %INPUT_DIM% ^
                @REM     --num_subvectors %%s ^
                @REM     --code_size %%c ^
                @REM     --dataset_split "%DATASET_SPLIT%"

                %PYTHON_PATH% main.py ^
                    --local_model_names "%MODEL_NAME%" ^
                    --output_dir "project/models/pq_head_kmeans_trained" ^
                    --device 0 ^
                    --epochs 20 ^
                    --lr 2e-5 ^
                    --attention_lr %%l ^
                    --l2 0.0 ^
                    --batch_size 32 ^
                    --use_pq ^
                    --input_dim %INPUT_DIM% ^
                    --num_subvectors %%s ^
                    --code_size %%c ^
                    --langs "%LANGS%" ^
                    --init_pq_path "project/models/pq_head_kmeans_init/pq_head_best.pt" ^
                    --log_dir "%LOG_DIR%" ^
                    --log_file "kmeans_init_sv%%s_cs%%c_attention_lr%%l.log"
            )
        )
    )
)
    
