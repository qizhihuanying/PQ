import argparse
from pathlib import Path
import pandas as pd
import torch
from functools import partial
from transformers import AutoModel, AutoTokenizer
import os
import json
import logging
from datetime import datetime

from _models.model import get_embedding_func_batched
from trainer import train

# 配置日志
def setup_logging(model_name, lang, lr, l2, log_dir=None, log_file=None):
    log_dir = Path(log_dir) if log_dir else Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    if log_file:
        log_file = log_dir / log_file
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"{model_name}+{lang}+lr={lr}+l2={l2}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

MIRACL_LANGUAGES = [
    "ar", "bn", "en", "es", "fi", "fr", "hi", "id", 
    "ja", "ko", "ru", "sw", "te", "th", "zh", "de", "yo", "fa"
]

def parse_args():
    parser = argparse.ArgumentParser(description="训练和评估PQ量化模型")
    parser.add_argument("--local_model_names", type=str, nargs="+", default=["intfloat/multilingual-e5-base"], help="本地模型名称")
    parser.add_argument("--api_model_names", type=str, nargs="+", default=[], help="API模型名称")
    parser.add_argument("--output_dir", type=str, default="project/models/pq_head", help="模型输出目录")
    parser.add_argument("--device", type=str, default='0', help="使用设备，如'0'表示cuda:0，'cpu'表示CPU")
    parser.add_argument("--epochs", type=int, default=10, help="训练轮数")
    parser.add_argument("--lr", type=float, default=2e-5, help="学习率")
    parser.add_argument('--l2', type=float, default=0.0, help='权重衰减')
    parser.add_argument("--batch_size", type=int, default=32, help="批次大小")
    parser.add_argument("--train_sample_ratio", type=float, default=1.0, help="训练数据采样比例")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="验证集比例")
    parser.add_argument("--test_ratio", type=float, default=1.0, help="测试集比例")
    parser.add_argument("--base_trainable_layers", type=int, default=0, help="基础模型可训练层数")
    parser.add_argument("--use_pq", action="store_true", default=False, help="是否使用PQ量化")
    parser.add_argument("--input_dim", type=int, default=768, help="输入维度")
    parser.add_argument("--num_subvectors", type=int, default=128, help="子向量数量")
    parser.add_argument("--code_size", type=int, default=32, help="码本大小")
    parser.add_argument("--dataset", type=str, default="miracl", help="数据集名称")
    parser.add_argument("--langs", type=str, nargs="+", default=["zh"], help="处理的语言")
    parser.add_argument("--log_dir", type=str, default="logs/debug", help="日志目录")
    parser.add_argument("--log_file", type=str, default="debug.log", help="日志文件名")
    parser.add_argument("--model_name_with_params", action="store_true", default=False, help="模型名称是否包含参数")
    
    return parser.parse_args()

def load_local_models(model_names, device):
    models = []
    tokenizers = []
    
    for model_name in model_names:
        print(f"加载本地模型: {model_name}")
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        models.append(model)
        tokenizers.append(tokenizer)
    
    return models, tokenizers

def prepare_api_embedding_funcs(model_names, device_id=None):
    embedding_funcs = []
    
    for model_name in model_names:
        print(f"准备API模型: {model_name}")
        embedding_funcs.append(partial(get_embedding_func_batched(model_name), device_id=device_id))
    
    return embedding_funcs

def prepare_data(args):
    if args.dataset != "miracl":
        raise ValueError(f"不支持的数据集: {args.dataset}")
        
    # 处理单语言情况
    if args.langs and len(args.langs) == 1:
        lang = args.langs[0]
        
        # 加载数据文件
        dev_data_path = Path(f"datasets/miracl/{lang}/dev/processed_data.pkl")
        train_data_path = Path(f"datasets/miracl/{lang}/train/processed_data.pkl")
        
        if not dev_data_path.exists() or not train_data_path.exists():
            raise FileNotFoundError(f"找不到处理好的{lang}数据文件，请先运行make_dataset.py")
        
        # 加载测试集和训练集
        test_data = pd.read_pickle(dev_data_path)
        full_train_data = pd.read_pickle(train_data_path)
        
        # 训练集采样
        if args.train_sample_ratio < 1.0:
            train_size = max(1, int(len(full_train_data) * args.train_sample_ratio))
            train_data = full_train_data.sample(n=train_size, random_state=42)
        else:
            train_data = full_train_data
        
        # 拆分验证集
        if args.val_ratio > 0:
            val_size = int(len(train_data) * args.val_ratio)
            val_size = max(1, val_size)
            val_data = train_data.sample(n=val_size, random_state=42)
            train_data = train_data.drop(val_data.index)
        else:
            val_data = pd.DataFrame(columns=train_data.columns)
        
        return train_data, val_data, test_data
    
    # 处理多语言情况
    data_frames = []
    for lang in args.langs if args.langs else MIRACL_LANGUAGES:
        data_path = Path(f"datasets/miracl/{lang}/dev/processed_data.pkl")
        if data_path.exists():
            lang_data = pd.read_pickle(data_path)
            data_frames.append(lang_data)
    
    if not data_frames:
        raise ValueError("没有找到任何处理好的数据文件")
        
    data = pd.concat(data_frames, ignore_index=True)
    
    # 特殊处理：test_ratio=1.0时
    if args.test_ratio >= 1.0:
        return pd.DataFrame(), pd.DataFrame(), data
    
    # 拆分数据集
    test_size = int(len(data) * args.test_ratio)
    test_data = data.sample(n=test_size, random_state=42)
    remaining_data = data.drop(test_data.index)
    
    val_size = int(len(remaining_data) * args.val_ratio)
    val_size = max(1, val_size)
    val_data = remaining_data.sample(n=val_size, random_state=42)
    train_data = remaining_data.drop(val_data.index)

    return train_data, val_data, test_data

def get_unique_model_path(args):
    """生成唯一的模型保存路径"""
    lang_str = "_".join(sorted(args.langs)) if args.langs else "all"
    model_name_short = "+".join([name.split("/")[-1] for name in args.local_model_names + args.api_model_names])
    
    # 基础路径
    path = f"{args.output_dir}/{model_name_short}/{lang_str}"
    
    if args.model_name_with_params:
        # 添加参数
        params = []
        params.append(f"e{args.epochs}")
        params.append(f"lr{args.lr}")
        params.append(f"l2{args.l2}")
        params.append(f"bs{args.batch_size}")
        
        if args.use_pq:
            params.append(f"pq-dim{args.input_dim}")
            params.append(f"sv{args.num_subvectors}")
            params.append(f"cs{args.code_size}")
        else:
            params.append("no-pq")
        
        if args.base_trainable_layers > 0:
            params.append(f"btl{args.base_trainable_layers}")
            
        # 连接参数
        param_str = "_".join(params)
        path = f"{path}/{param_str}"
    
    return path

def save_results(results, args):
    if results is None:
        return
        
    # 获取保存路径
    output_dir = get_unique_model_path(args)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 提取模型名称
    model_names = []
    for model_name in args.local_model_names:
        model_names.append(f"local_{model_name.replace('/', '_')}")
    for model_name in args.api_model_names:
        model_names.append(f"api_{model_name.replace('/', '_')}")
    
    # 创建结果摘要
    summary = {
        "dataset": args.dataset,
        "langs": args.langs,
        "use_pq": args.use_pq,
        "hyperparams": {
            "lr": args.lr,
            "l2": args.l2,
            "epochs": args.epochs,
            "batch_size": args.batch_size
        },
        "models": model_names,
        "results": results
    }
    
    # 保存结果
    with open(output_path / "evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

def main():
    args = parse_args()
    
    # 初始化日志
    combined_model_name = "+".join([name.split("/")[-1] for name in args.local_model_names + args.api_model_names])
    lang_str = "_".join(sorted(args.langs)) if args.langs else "all"
    logger = setup_logging(combined_model_name, lang_str, args.lr, args.l2, args.log_dir, args.log_file)
    
    # 设置设备
    device_str = args.device.lower()
    if device_str == 'cpu':
        device = torch.device('cpu')
        logger.info("使用CPU进行训练")
    else:
        if torch.cuda.is_available():
            device = torch.device(f'cuda:{device_str}' if device_str.isdigit() else 'cuda')
            logger.info(f"使用{device}进行训练")
        else:
            device = torch.device('cpu')
            logger.warning("CUDA不可用，使用CPU训练")
    
    # 加载模型和数据
    local_models, local_tokenizers = load_local_models(args.local_model_names, device)
    api_embedding_funcs = prepare_api_embedding_funcs(args.api_model_names, 
                                                     device_str if device_str.isdigit() else None)
    train_data, val_data, test_data = prepare_data(args)
    
    # 获取模型保存路径
    unique_model_path = get_unique_model_path(args)
    logger.info(f"模型将保存到: {unique_model_path}")
    
    # 训练模型
    logger.info(f"开始训练: epochs={args.epochs}, lr={args.lr}, batch_size={args.batch_size}, use_pq={args.use_pq}")
    if args.use_pq:
        logger.info(f"PQ参数: input_dim={args.input_dim}, num_subvectors={args.num_subvectors}, code_size={args.code_size}")
    
    results, pq_head = train(
        models=local_models,
        tokenizers=local_tokenizers,
        embedding_funcs=api_embedding_funcs,
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        l2=args.l2,
        batch_size=args.batch_size,
        num_trainable_layers=args.base_trainable_layers,
        output_dir=unique_model_path,
        use_pq=args.use_pq,
        input_dim=args.input_dim,
        num_subvectors=args.num_subvectors,
        code_size=args.code_size,
        logger=logger
    )
    
    # 保存结果
    save_results(results, args)
    logger.info("训练完成")

if __name__ == "__main__":
    main()