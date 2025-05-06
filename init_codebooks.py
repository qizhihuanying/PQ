import torch
import torch.nn as nn
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from sklearn.cluster import KMeans
import argparse
import logging
import os
from product_quantization import PQHead

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/init_codebooks.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 支持的语言列表
ALL_LANGUAGES = ["ar", "bn", "en", "es", "fi", "fr", "hi", "id", "ja", "ko", "ru", "sw", "te", "th", "zh", "fa", "de", "yo"]

def load_corpus(corpus_path):
    """加载语料库数据"""
    logger.info(f"加载语料库: {corpus_path}")
    
    documents = {}
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="加载语料"):
            try:
                data = json.loads(line.strip())
                if "text" in data:
                    doc_id = str(data.get("docid", ""))
                    doc_text = data["text"]
                    documents[doc_id] = doc_text
            except Exception as e:
                logger.error(f"处理语料时出错: {e}")
    
    logger.info(f"共加载了 {len(documents)} 个文档")
    return documents

def load_qrels(qrel_path):
    """加载相关性评分文件，获取所有在qrel中使用的文档ID"""
    logger.info(f"加载qrel文件: {qrel_path}")
    
    doc_ids = set()
    try:
        with open(qrel_path, encoding="utf-8") as f:
            for line in tqdm(f, desc="加载qrel"):
                try:
                    qid, _, docid, _ = line.strip().split('\t')
                    doc_ids.add(docid)
                except ValueError:
                    logger.error(f"格式错误的行: {line}")
                    continue
    except Exception as e:
        logger.error(f"加载相关性文件失败: {qrel_path}, 错误: {e}")
    
    logger.info(f"qrel中包含 {len(doc_ids)} 个唯一文档ID")
    return doc_ids

def batch_encode_texts(texts, tokenizer, model, device, batch_size=32):
    """批量编码文本获取嵌入向量"""
    all_embeddings = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="编码文档"):
        batch_texts = texts[i:i+batch_size]
        encoded = tokenizer(batch_texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model(**encoded)
            embeddings = torch.nn.functional.normalize(output[0][:, 0], p=2, dim=1)
            all_embeddings.append(embeddings.cpu())
    
    # 合并所有批次的嵌入向量
    return torch.cat(all_embeddings, dim=0)

def kmeans_initialize_codebooks(embeddings, num_subvectors, code_size, subvector_dim):
    """使用K-means初始化码本"""
    logger.info(f"使用K-means初始化码本: {num_subvectors}个子向量, 每个码本大小{code_size}")
    
    # 创建码本存储
    codebooks = torch.zeros(num_subvectors, code_size, subvector_dim)
    
    # 对每个子向量维度执行K-means聚类
    for i in tqdm(range(num_subvectors), desc="执行K-means聚类"):
        # 提取当前子向量空间的所有样本
        start_idx = i * subvector_dim
        end_idx = (i + 1) * subvector_dim
        subset = embeddings[:, start_idx:end_idx].numpy()
        
        # 执行K-means聚类
        kmeans = KMeans(n_clusters=code_size, random_state=42, n_init=10)
        kmeans.fit(subset)
        
        # 存储聚类中心作为码本
        codebooks[i] = torch.tensor(kmeans.cluster_centers_)
    
    return codebooks

def update_pq_head_codebooks(pq_head, codebooks):
    """更新PQ头的码本参数"""
    logger.info("更新PQ头的码本参数")
    if pq_head.codebooks.shape != codebooks.shape:
        raise ValueError(f"形状不匹配: PQ头码本形状 {pq_head.codebooks.shape}, 新码本形状 {codebooks.shape}")
    with torch.no_grad():
        pq_head.codebooks.copy_(codebooks)
    
    return pq_head

def parse_args():
    parser = argparse.ArgumentParser(description="使用K-means初始化PQ码本")
    parser.add_argument("--model_name", type=str, default="intfloat/multilingual-e5-base", help="基础模型名称")
    parser.add_argument("--langs", nargs="+", default=["ar", "bn", "en", "es", "fi", "fr", "hi", "id", "ja", "ko", "ru", "sw", "te", "th", "zh", "fa"], help="要处理的语言列表")
    parser.add_argument("--output_dir", type=str, default="project/models/pq_head_initialized", help="输出目录")
    parser.add_argument("--device", type=str, default='0', help="使用设备，如'0'表示cuda:0，'cpu'表示CPU")
    parser.add_argument("--batch_size", type=int, default=32, help="批处理大小")
    parser.add_argument("--input_dim", type=int, default=768, help="输入嵌入维度")
    parser.add_argument("--num_subvectors", type=int, default=128, help="子向量数量")
    parser.add_argument("--code_size", type=int, default=32, help="码本大小")
    parser.add_argument("--dataset_split", type=str, default="train", help="数据集分割，如dev或train")
    
    return parser.parse_args()

def process_language(lang, dataset_split, device, tokenizer, model, batch_size):
    """处理单个语言的语料库和qrel"""
    # 定义路径
    corpus_path = f"datasets/miracl/{lang}/corpus/corpus.jsonl"
    qrel_path = f"datasets/miracl/{lang}/{dataset_split}/qrels.tsv"
    
    # 检查文件是否存在
    if not os.path.exists(corpus_path):
        logger.warning(f"语料库文件不存在: {corpus_path}")
        return None, 0
    
    if not os.path.exists(qrel_path):
        logger.warning(f"qrel文件不存在: {qrel_path}")
        return None, 0
    
    # 加载语料库和qrel
    corpus_docs = load_corpus(corpus_path)
    qrel_doc_ids = load_qrels(qrel_path)
    
    # 只使用在qrel中出现的文档
    filtered_docs = []
    for doc_id in qrel_doc_ids:
        if doc_id in corpus_docs:
            filtered_docs.append(corpus_docs[doc_id])
        else:
            logger.warning(f"文档ID {doc_id} 在语料库中不存在")
    
    doc_count = len(filtered_docs)
    logger.info(f"从 {len(corpus_docs)} 个文档中筛选出 {doc_count} 个在qrel中使用的文档")
    
    if not filtered_docs:
        logger.error(f"语言 {lang} 没有找到符合条件的文档，跳过")
        return None, 0
    
    # 获取文档嵌入向量
    embeddings = batch_encode_texts(filtered_docs, tokenizer, model, device, batch_size)
    logger.info(f"获取了 {embeddings.shape[0]} 个文档的嵌入向量，每个维度 {embeddings.shape[1]}")
    
    return embeddings, doc_count

def main():
    args = parse_args()
    
    device_str = args.device.lower()
    if device_str == 'cpu':
        device = torch.device('cpu')
        logger.info("使用CPU进行处理")
    else:
        if torch.cuda.is_available():
            device = torch.device(f'cuda:{device_str}' if device_str.isdigit() else 'cuda')
            logger.info(f"使用{device}进行处理")
        else:
            device = torch.device('cpu')
            logger.warning("CUDA不可用，使用CPU处理")
    
    # 加载模型和分词器
    logger.info(f"加载模型: {args.model_name}")
    model = AutoModel.from_pretrained(args.model_name, trust_remote_code=True).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model.eval()
    
    # 从所有语言收集嵌入向量
    all_embeddings = []
    total_docs = 0
    
    for lang in args.langs:
        if lang not in ALL_LANGUAGES:
            logger.warning(f"不支持的语言: {lang}，跳过")
            continue
            
        logger.info(f"处理语言: {lang}")
        lang_embeddings, doc_count = process_language(
            lang, 
            args.dataset_split, 
            device, 
            tokenizer, 
            model, 
            args.batch_size
        )
        
        if lang_embeddings is not None and doc_count > 0:
            all_embeddings.append(lang_embeddings)
            total_docs += doc_count
    
    if not all_embeddings:
        logger.error("没有找到任何符合条件的文档，无法进行编码和初始化")
        return
    
    # 合并所有语言的嵌入向量
    combined_embeddings = torch.cat(all_embeddings, dim=0)
    logger.info(f"合并了所有语言的嵌入向量，总共 {combined_embeddings.shape[0]} 个文档")
    
    # 计算子向量维度
    subvector_dim = args.input_dim // args.num_subvectors
    
    # 使用K-means初始化码本
    codebooks = kmeans_initialize_codebooks(
        combined_embeddings, 
        args.num_subvectors, 
        args.code_size, 
        subvector_dim
    )
    
    # 创建新的PQ头
    pq_head = PQHead(
        input_dim=args.input_dim,
        num_subvectors=args.num_subvectors,
        code_size=args.code_size,
        use_pq=True
    )
    
    # 用K-means结果更新码本
    pq_head = update_pq_head_codebooks(pq_head, codebooks)
    
    # 保存初始化的模型
    model_name_short = args.model_name.split("/")[-1]
    output_dir = Path(f"{args.output_dir}/{model_name_short}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建包含参数信息的文件名
    model_file_name = f"pq_head_cs{args.code_size}_ns{args.num_subvectors}.pt"
    output_path = output_dir / model_file_name
    
    # 保存模型
    pq_head.save_model(output_path)
    logger.info(f"已保存初始化的PQ模型到: {output_path}")

if __name__ == "__main__":
    main() 