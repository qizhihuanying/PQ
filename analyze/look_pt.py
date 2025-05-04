import torch
from pathlib import Path

def print_codebooks(init_pq_path):
    # 确保提供的路径存在
    init_pq_path = Path(init_pq_path)
    if not init_pq_path.exists():
        print(f"错误：文件 {init_pq_path} 不存在")
        return

    # 加载检查点
    checkpoint = torch.load(init_pq_path, map_location='cpu')
    
    # 检查是否有 'codebooks' 键
    if 'codebooks' in checkpoint:
        codebooks = checkpoint['codebooks']
        print("Codebooks 值：")
        print(codebooks)
        
        # 打印更多详细信息（形状、类型等）
        if isinstance(codebooks, torch.Tensor):
            print(f"形状: {codebooks.shape}")
            print(f"数据类型: {codebooks.dtype}")
        elif isinstance(codebooks, list):
            print(f"列表长度: {len(codebooks)}")
            for i, cb in enumerate(codebooks):
                print(f"Codebook {i} 形状: {cb.shape}, 数据类型: {cb.dtype}")
    else:
        print("检查点中未找到 'codebooks' 键")
        print("可用键：", list(checkpoint.keys()))

# 从你的代码中获取 init_pq_path
init_pq_path = "project/models/pq_head_kmeans_init/pq_head_best.pt"

# 调用函数
print_codebooks(init_pq_path)