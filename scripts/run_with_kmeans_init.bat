@echo off
mkdir logs\dpq_kmeans_init_attention 2>nul

set PYTHON_PATH="C:\Users\QZHYc\Anaconda3\envs\bge\python.exe"
if not exist %PYTHON_PATH% set PYTHON_PATH="C:\Users\QZHYc\Anaconda3\python.exe"

set MODEL_NAME=intfloat/multilingual-e5-base
set CORPUS_PATH=datasets/miracl/zh/train/corpus.jsonl
set INPUT_DIM=768
set LANGS=zh
set LOG_DIR=logs/dpq_kmeans_init_attention

set SV_OPTIONS=256
set CS_OPTIONS=64
set ATTENTION_LR_OPTIONS=1e-5 5e-5 1e-6 5e-6 1e-4 5e-4


for %%a in (0 1) do (
    for %%s in (%SV_OPTIONS%) do (
        for %%c in (%CS_OPTIONS%) do (
            for %%l in (%ATTENTION_LR_OPTIONS%) do (

                %PYTHON_PATH% init_codebooks.py ^
                    --model_name "%MODEL_NAME%" ^
                    --corpus_path "%CORPUS_PATH%" ^
                    --output_dir "project/models/pq_head_kmeans_init" ^
                    --device 0 ^
                    --sample_ratio 1.0 ^
                    --batch_size 32 ^
                    --input_dim %INPUT_DIM% ^
                    --num_subvectors %%s ^
                    --code_size %%c

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
    
