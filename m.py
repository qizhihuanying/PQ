import torch
import time
import threading
import random
import numpy as np
import multiprocessing

# 检查可用 GPU
if torch.cuda.device_count() == 0:
    print("错误: 未检测到 GPU")
    exit(1)

print(f"可用 GPU 数量: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

# 检查可用 CPU 核心
cpu_count = multiprocessing.cpu_count()
print(f"可用 CPU 核心数: {cpu_count}")

# 指定要占用的 GPU ID
gpu_ids = [0,3,5]

# 验证 GPU ID 是否有效
for gpu_id in gpu_ids:
    if gpu_id >= torch.cuda.device_count():
        print(f"错误: GPU {gpu_id} 不可用，系统中只有 {torch.cuda.device_count()} 块 GPU")
        exit(1)

# 基础显存占用（21GB）
base_memory_to_allocate = 42 * 1024 * 1024 * 1024  # 21GB in bytes
max_extra_memory = 1000 * 1024 * 1024  # 最大额外 100MB

# 存储张量的列表
tensors = []

# 每个 GPU 的高强度计算任务
def run_gpu_compute_task(gpu_id):
    try:
        torch.cuda.set_device(gpu_id)
        
        # 为每个 GPU 随机增加 0-100MB 显存
        extra_memory = random.randint(0, max_extra_memory)
        memory_to_allocate = base_memory_to_allocate + extra_memory
        
        # 分配显存
        num_elements = memory_to_allocate // 4
        tensor = torch.randn(num_elements, dtype=torch.float32, device=f'cuda:{gpu_id}')
        tensors.append(tensor)
        print(f"GPU {gpu_id}: 成功分配约 {memory_to_allocate / (1024**3):.2f} GB 显存")

        # 创建用于计算的大矩阵（8192x8192）
        matrix_size = 8192
        a = torch.randn(matrix_size, matrix_size, dtype=torch.float32, device=f'cuda:{gpu_id}')
        b = torch.randn(matrix_size, matrix_size, dtype=torch.float32, device=f'cuda:{gpu_id}')
        c = torch.randn(matrix_size, matrix_size, dtype=torch.float32, device=f'cuda:{gpu_id}')
        tensors.extend([a, b, c])

        # 高强度计算循环
        print(f"GPU {gpu_id}: 开始运行高强度矩阵计算...")
        while True:
            intensity = random.random()
            if intensity < 0.2:  # 20% 概率降低计算强度
                d = torch.matmul(a, b)
                torch.cuda.synchronize()
                time.sleep(random.uniform(0.05, 0.1))
            else:  # 80% 概率高强度计算
                d = torch.matmul(a, b)
                d = torch.matmul(d, c)
                d = d * 2.0 + torch.sin(d)
                d = torch.matmul(d, b.t())
                torch.cuda.synchronize()
                time.sleep(random.uniform(0.0, 0.01))

    except RuntimeError as e:
        print(f"GPU {gpu_id} 分配显存或运行任务失败: {e}")
    except KeyboardInterrupt:
        print(f"GPU {gpu_id} 任务终止")

# CPU 密集计算任务
def run_cpu_compute_task(thread_id):
    try:
        # 创建大矩阵（5000x5000）
        matrix_size = 5000
        a = np.random.randn(matrix_size, matrix_size)
        b = np.random.randn(matrix_size, matrix_size)

        print(f"CPU 线程 {thread_id}: 开始运行高强度计算...")
        while True:
            intensity = random.random()
            if intensity < 0.3:  # 30% 概率降低 CPU 负载
                c = np.dot(a, b)[:100, :100]  # 小规模计算
                time.sleep(random.uniform(0.1, 0.2))  # 降低占用
            else:  # 70% 概率高强度计算
                c = np.dot(a, b)  # 矩阵乘法
                c = c * 2.0 + np.sin(c)  # 复杂运算
                c = np.dot(c, b.T)  # 转置矩阵乘法
                time.sleep(random.uniform(0.0, 0.05))  # 微小延时

    except KeyboardInterrupt:
        print(f"CPU 线程 {thread_id} 任务终止")

# 主程序
try:
    # 启动 GPU 线程
    gpu_threads = []
    for gpu_id in gpu_ids:
        thread = threading.Thread(target=run_gpu_compute_task, args=(gpu_id,))
        thread.daemon = True
        gpu_threads.append(thread)
        thread.start()

    while True:
        time.sleep(10)

except KeyboardInterrupt:
    print("程序终止，释放显存和资源...")
    tensors = []
    torch.cuda.empty_cache()
except Exception as e:
    print(f"程序错误: {e}")
    tensors = []
    torch.cuda.empty_cache()