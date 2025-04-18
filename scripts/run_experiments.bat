@echo off
REM PQ量化模型训练实验脚本

REM 创建日志目录
mkdir logs\dbq_base 2>nul

REM 基础参数
set LOG_DIR=logs/dpq_base
set BASE_ARGS=--use_pq --log_dir %LOG_DIR% 

set PYTHON_PATH="C:\Users\QZHYc\Anaconda3\envs\bge\python.exe"
if not exist %PYTHON_PATH% set PYTHON_PATH="C:\Users\QZHYc\Anaconda3\python.exe"

REM 不同的num_subvectors和code_size组合实验
REM 16, 32, 64, 128, 256的所有组合

REM num_subvectors=16的组合
@REM %PYTHON_PATH% main.py %BASE_ARGS% --num_subvectors 16 --code_size 16 --log_file sv16_cs16.log
@REM %PYTHON_PATH% main.py %BASE_ARGS% --num_subvectors 16 --code_size 32 --log_file sv16_cs32.log
@REM %PYTHON_PATH% main.py %BASE_ARGS% --num_subvectors 16 --code_size 64 --log_file sv16_cs64.log
@REM %PYTHON_PATH% main.py %BASE_ARGS% --num_subvectors 16 --code_size 128 --log_file sv16_cs128.log
@REM %PYTHON_PATH% main.py %BASE_ARGS% --num_subvectors 16 --code_size 256 --log_file sv16_cs256.log

@REM REM num_subvectors=32的组合
@REM %PYTHON_PATH% main.py %BASE_ARGS% --num_subvectors 32 --code_size 16 --log_file sv32_cs16.log
@REM %PYTHON_PATH% main.py %BASE_ARGS% --num_subvectors 32 --code_size 32 --log_file sv32_cs32.log
@REM %PYTHON_PATH% main.py %BASE_ARGS% --num_subvectors 32 --code_size 64 --log_file sv32_cs64.log
@REM %PYTHON_PATH% main.py %BASE_ARGS% --num_subvectors 32 --code_size 128 --log_file sv32_cs128.log
@REM %PYTHON_PATH% main.py %BASE_ARGS% --num_subvectors 32 --code_size 256 --log_file sv32_cs256.log

@REM REM num_subvectors=64的组合
@REM %PYTHON_PATH% main.py %BASE_ARGS% --num_subvectors 64 --code_size 16 --log_file sv64_cs16.log
@REM %PYTHON_PATH% main.py %BASE_ARGS% --num_subvectors 64 --code_size 32 --log_file sv64_cs32.log
@REM %PYTHON_PATH% main.py %BASE_ARGS% --num_subvectors 64 --code_size 64 --log_file sv64_cs64.log
@REM %PYTHON_PATH% main.py %BASE_ARGS% --num_subvectors 64 --code_size 128 --log_file sv64_cs128.log
@REM %PYTHON_PATH% main.py %BASE_ARGS% --num_subvectors 64 --code_size 256 --log_file sv64_cs256.log

@REM REM num_subvectors=128的组合
@REM %PYTHON_PATH% main.py %BASE_ARGS% --num_subvectors 128 --code_size 16 --log_file sv128_cs16.log
@REM %PYTHON_PATH% main.py %BASE_ARGS% --num_subvectors 128 --code_size 32 --log_file sv128_cs32.log
@REM %PYTHON_PATH% main.py %BASE_ARGS% --num_subvectors 128 --code_size 64 --log_file sv128_cs64.log
@REM %PYTHON_PATH% main.py %BASE_ARGS% --num_subvectors 128 --code_size 128 --log_file sv128_cs128.log
@REM %PYTHON_PATH% main.py %BASE_ARGS% --num_subvectors 128 --code_size 256 --log_file sv128_cs256.log

REM num_subvectors=256的组合
@REM %PYTHON_PATH% main.py %BASE_ARGS% --num_subvectors 256 --code_size 256 --log_file sv256_cs256.log
@REM %PYTHON_PATH% main.py %BASE_ARGS% --num_subvectors 256 --code_size 128 --log_file sv256_cs128.log
@REM %PYTHON_PATH% main.py %BASE_ARGS% --num_subvectors 256 --code_size 64 --log_file sv256_cs64.log
%PYTHON_PATH% main.py %BASE_ARGS% --num_subvectors 256 --code_size 32 --log_file sv256_cs32.log
@REM %PYTHON_PATH% main.py %BASE_ARGS% --num_subvectors 256 --code_size 16 --log_file sv256_cs16.log


