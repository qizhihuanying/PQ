@echo off
REM PQ量化模型训练实验脚本

REM 创建日志目录
mkdir logs\experiments 2>nul

REM 基础参数
set BASE_ARGS=--local_model_names intfloat/multilingual-e5-base --dataset miracl --langs zh --epochs 20 --lr 2e-5 --l2 1e-6 --batch_size 32 --model_name_with_params --log_dir logs/experiments

REM 运行基础实验
echo 基础实验 - 不使用PQ量化
python main.py %BASE_ARGS% --log_file base_no_pq.log

REM 运行不同PQ参数的实验
echo PQ量化 - 默认参数 (768维, 8子向量, 码本大小256)
python main.py %BASE_ARGS% --use_pq --log_file pq_default.log

echo PQ量化 - 4子向量
python main.py %BASE_ARGS% --use_pq --num_subvectors 4 --log_file pq_sv4.log

echo PQ量化 - 16子向量
python main.py %BASE_ARGS% --use_pq --num_subvectors 16 --log_file pq_sv16.log

echo PQ量化 - 码本大小128
python main.py %BASE_ARGS% --use_pq --code_size 128 --log_file pq_cs128.log

echo PQ量化 - 码本大小512
python main.py %BASE_ARGS% --use_pq --code_size 512 --log_file pq_cs512.log

REM 多语言实验
echo PQ量化 - 英语数据
python main.py %BASE_ARGS% --use_pq --langs en --log_file pq_en.log

echo PQ量化 - 多语言数据
python main.py %BASE_ARGS% --use_pq --langs zh en fr --log_file pq_multi.log

REM 不同学习率
echo PQ量化 - 学习率1e-5
python main.py %BASE_ARGS% --use_pq --lr 1e-5 --log_file pq_lr1e-5.log

echo PQ量化 - 学习率5e-5
python main.py %BASE_ARGS% --use_pq --lr 5e-5 --log_file pq_lr5e-5.log

echo 所有实验已完成
