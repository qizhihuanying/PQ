import os
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pickle
import argparse
import json
import requests
import gzip
from typing import Dict, List, Tuple
# 延迟导入 HuggingFace datasets（避免被本地 datasets 目录遮蔽）
import random

# 支持的语言定义
SURPRISE_LANGUAGES = ['de', 'yo']
BASE_LANGUAGES = ['ar', 'bn', 'en', 'es', 'fa', 'fi', 'fr', 'hi', 'id', 'ja', 'ko', 'ru', 'sw', 'te', 'th', 'zh']
NEW_LANGUAGES = ['es', 'fa', 'fr', 'hi', 'zh'] + SURPRISE_LANGUAGES
ALL_LANGUAGES = BASE_LANGUAGES + SURPRISE_LANGUAGES

# 数据集URL定义
DATASET_URLS = {
    lang: {
        'dev': [
            f'https://huggingface.co/datasets/miracl/miracl/resolve/main/miracl-v1.0-{lang}/topics/topics.miracl-v1.0-{lang}-dev.tsv',
            f'https://huggingface.co/datasets/miracl/miracl/resolve/main/miracl-v1.0-{lang}/qrels/qrels.miracl-v1.0-{lang}-dev.tsv',
        ],
        'testB': [
            f'https://huggingface.co/datasets/miracl/miracl/resolve/main/miracl-v1.0-{lang}/topics/topics.miracl-v1.0-{lang}-test-b.tsv',
        ],
    } for lang in ALL_LANGUAGES
}

# 添加train分割（排除惊喜语言）
for lang in ALL_LANGUAGES:
    if lang not in SURPRISE_LANGUAGES:
        DATASET_URLS[lang]['train'] = [
            f'https://huggingface.co/datasets/miracl/miracl/resolve/main/miracl-v1.0-{lang}/topics/topics.miracl-v1.0-{lang}-train.tsv',
            f'https://huggingface.co/datasets/miracl/miracl/resolve/main/miracl-v1.0-{lang}/qrels/qrels.miracl-v1.0-{lang}-train.tsv',
        ]

def download_file(url, output_path, force_download=False):
    """下载文件到指定路径"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if os.path.exists(output_path):
        if not force_download:
            return output_path
        os.remove(output_path)
    
    print(f"下载文件: {url}")
    response = requests.get(url)
    if response.status_code == 200:
        with open(output_path, 'wb') as f:
            f.write(response.content)
        return output_path
    else:
        print(f"下载失败: {url}, 状态码: {response.status_code}")
        return None

def load_topic(fn):
    """加载主题文件"""
    qid2topic = {}
    try:
        with open(fn, encoding="utf-8") as f:
            for line in f:
                qid, topic = line.strip().split('\t')
                qid2topic[qid] = topic
        return qid2topic
    except Exception as e:
        print(f"加载主题文件失败: {fn}, 错误: {e}")
        return {}

def load_qrels(fn):
    """加载相关性评分文件"""
    if fn is None or not os.path.exists(fn):
        return None

    qrels = {}
    try:
        with open(fn, encoding="utf-8") as f:
            for line in f:
                try:
                    qid, _, docid, rel = line.strip().split('\t')
                    if qid not in qrels:
                        qrels[qid] = {}
                    qrels[qid][docid] = int(rel)
                except ValueError:
                    print(f"格式错误的行: {line}")
                    continue
        return qrels
    except Exception as e:
        print(f"加载相关性文件失败: {fn}, 错误: {e}")
        return None

def parse_topic_text(text: str) -> Dict[str, str]:
    """从文本内容解析 topics TSV 为 {qid: topic}."""
    qid2topic: Dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split('\t')
        if len(parts) >= 2:
            qid, topic = parts[0], parts[1]
            qid2topic[qid] = topic
    return qid2topic

def parse_qrels_text(text: str) -> Dict[str, Dict[str, int]]:
    """从文本内容解析 qrels TSV 为 {qid: {docid: rel}}."""
    qrels: Dict[str, Dict[str, int]] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            qid, _, docid, rel = line.split('\t')
        except ValueError:
            continue
        qrels.setdefault(qid, {})[docid] = int(rel)
    return qrels

def download_corpus(lang, base_path, force_download=False):
    """获取语料库句柄：优先使用本地分片目录，否则以 HF 流式方式返回数据集对象。
    不会在 datasets/miracl 或 datasets/miracl-corpus 下写入任何新文件。
    """
    local_shard_dir = Path(f"datasets/miracl-corpus/miracl-corpus-v1.0-{lang}")
    if local_shard_dir.exists() and local_shard_dir.is_dir():
        shard_files = list(local_shard_dir.glob("docs-*.jsonl.gz"))
        print(f"使用本地分片: {local_shard_dir} ({len(shard_files)} 个文件)")
        if not shard_files:
            print(f"警告: {local_shard_dir} 中未找到 docs-*.jsonl.gz 文件")
        return local_shard_dir

    print(f"尝试通过 HuggingFace 数据集流式读取 {lang} 语料库（不落盘）...")
    try:
        import importlib
        hf_datasets = importlib.import_module('datasets')
        load_dataset = getattr(hf_datasets, 'load_dataset', None)
        if load_dataset is None:
            raise ImportError("本地 datasets 目录遮蔽了 HuggingFace datasets 库，无法找到 load_dataset")

        corpus = load_dataset('miracl/miracl-corpus', lang, trust_remote_code=True)['train']
        return corpus
    except Exception as e:
        print(f"下载语料库失败: {e}")
        if local_shard_dir.exists() and local_shard_dir.is_dir():
            print("回退到现有的本地分片")
            return local_shard_dir
        return None

def load_corpus(corpus_source, lang=None):
    """从本地 JSONL 或分片目录加载语料库"""
    documents = {}

    try:
        if corpus_source is None:
            return documents

        # HF 数据集对象：直接迭代（避免写盘）
        if not isinstance(corpus_source, (str, os.PathLike, Path)):
            for doc in tqdm(corpus_source, desc=f"加载语料库({lang or 'hf'})"):
                doc_id = str(doc.get('docid'))
                documents[doc_id] = {
                    "title": doc.get('title', ""),
                    "text": doc['text']
                }
            return documents

        corpus_path = Path(corpus_source)
        if corpus_path.is_dir():
            gz_files = sorted(corpus_path.glob("docs-*.jsonl.gz"), key=lambda p: p.name)
            if not gz_files:
                print(f"警告: {corpus_path} 中未找到 docs-*.jsonl.gz 文件")
                return documents

            for gz_file in tqdm(gz_files, desc=f"加载语料库({lang or corpus_path.name})"):
                with gzip.open(gz_file, "rt", encoding="utf-8") as f:
                    for line in f:
                        doc = json.loads(line.strip())
                        doc_id = str(doc['docid'])
                        documents[doc_id] = {
                            "title": doc.get('title', ""),
                            "text": doc['text']
                        }
            return documents

        with open(corpus_path, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc="加载语料库"):
                doc = json.loads(line.strip())
                doc_id = str(doc['docid'])
                documents[doc_id] = {
                    "title": doc.get('title', ""),
                    "text": doc['text']
                }
        return documents
    except Exception as e:
        print(f"加载语料库失败: {corpus_source}, 错误: {e}")
        return {}


def load_miracl_data(lang: str, split: str = "dev", force_download: bool = False) -> Tuple[Dict[str, str], Dict[str, Dict[str, str]], Dict[str, Dict[str, int]]]:
    """
    加载MIRACL数据集，优先从本地加载，如果本地不存在或指定强制下载，则从Hugging Face下载
    
    Args:
        lang: 语言代码
        split: 数据集拆分，'dev'、'train'或'testB'
        force_download: 是否强制从Hugging Face下载
        
    Returns:
        queries: 查询字典 {query_id: query_text}
        documents: 文档字典 {doc_id: {"title": title, "text": text}}
        qrels: 相关性字典 {query_id: {doc_id: relevance}}
    """
    # 不在 miracl/ 下创建任何新目录，topics/qrels 直接按现有 MIRACL 结构读取
    
    # 检查是否有链接可用
    if lang not in ALL_LANGUAGES:
        raise ValueError(f"不支持的语言: {lang}，支持的语言有: {ALL_LANGUAGES}")
    
    if split not in DATASET_URLS[lang]:
        raise ValueError(f"语言 {lang} 不支持数据集拆分: {split}")
    
    # 下载或加载文件
    # 优先读取本地 MIRACL 目录结构（datasets/miracl/miracl-v1.0-<lang>）
    miracl_lang_dir = Path(f"datasets/miracl/miracl-v1.0-{lang}")
    urls = DATASET_URLS[lang][split]
    topic_local = miracl_lang_dir / "topics" / f"topics.miracl-v1.0-{lang}-{split}.tsv"
    qrels_local = miracl_lang_dir / "qrels" / f"qrels.miracl-v1.0-{lang}-{split}.tsv"
    
    if topic_local.exists():
        queries = load_topic(topic_local)
    else:
        resp = requests.get(urls[0])
        queries = parse_topic_text(resp.text) if resp.status_code == 200 else {}
    
    if len(urls) >= 2:
        if qrels_local.exists():
            qrels = load_qrels(qrels_local)
        else:
            resp = requests.get(urls[1])
            qrels = parse_qrels_text(resp.text) if resp.status_code == 200 else None
    else:
        qrels = None
    
    # 获取并加载语料库（仅读取，不写入 miracl/ 或 miracl-corpus/）
    corpus_handle = download_corpus(lang, base_path=None, force_download=force_download)
    documents = load_corpus(corpus_handle, lang=lang)
    
    print(f"已加载 {lang} 数据集 ({split}): {len(queries)} 查询, {len(documents)} 文档")
    if qrels:
        print(f"已加载相关性信息: {len(qrels)} 查询有相关性判断")
    
    return queries, documents, qrels

def create_training_data(queries: Dict[str, str], documents: Dict[str, Dict[str, str]], qrels: Dict[str, Dict[str, int]]) -> pd.DataFrame:
    """
    创建训练数据，使用qrels中的正负样本
    
    Args:
        queries: 查询字典
        documents: 文档字典，包含title和text
        qrels: 相关性字典
        
    Returns:
        训练数据DataFrame
    """
    data = []
    # 为每个查询创建样本对
    for query_id, query_text in tqdm(queries.items(), desc="创建数据"):
        if qrels is None or query_id not in qrels:
            continue
            
        # 对qrels中所有文档处理（包括正负样本）
        for doc_id, relevance in qrels[query_id].items():
            if doc_id in documents:  # 确保文档存在
                doc = documents[doc_id]
                doc_text = doc["text"]
                doc_title = doc["title"]
                
                # 组合标题和文本
                doc_full_text = doc_title + " " + doc_text if doc_title else doc_text
                
                data.append({
                    'query': query_text,
                    'doc_text': doc_text,
                    'doc_title': doc_title,
                    'sample1': query_text,  # 添加sample1列
                    'sample2': doc_text,  # 添加sample2列
                    'query_id': query_id,
                    'doc_id': doc_id,
                    'relevance': relevance
                })

    df = pd.DataFrame(data)
    relevant_count = sum(df['relevance'] > 0)
    irrelevant_count = len(df) - relevant_count
    print(f"创建了 {len(df)} 个样本，其中包含 {relevant_count} 个相关文档样本和 {irrelevant_count} 个不相关文档样本")
    return df

def process_miracl_dataset(lang_list: List[str] = None, force_download: bool = False) -> Dict[str, pd.DataFrame]:
    """
    处理指定语言的MIRACL数据集
    
    Args:
        lang_list: 要处理的语言列表，None表示处理所有语言
        force_download: 是否强制从Hugging Face下载
        
    Returns:
        处理后的数据集字典 {语言: DataFrame}
    """
    # 如果未指定语言，则使用所有支持的语言
    if lang_list is None:
        lang_list = ALL_LANGUAGES
    elif isinstance(lang_list, str):
        lang_list = [lang_list]
    
    all_data = {}
    for lang in lang_list:
        try:
            print(f"\n开始处理 {lang} 语言的MIRACL数据...")
            
            # 加载dev数据（评估集，必须处理）
            queries, documents, qrels = load_miracl_data(lang, "dev", force_download=force_download)
            dev_data = create_training_data(queries, documents, qrels)
            
            # 保存处理后的dev数据（仅写入 datasets/processed）
            dev_dir = Path(f"datasets/processed/{lang}/dev")
            dev_dir.mkdir(parents=True, exist_ok=True)
            dev_data.to_pickle(dev_dir / "processed_data.pkl")
            
            # 合并数据结果
            lang_data = dev_data
            
            # 处理训练集（对于非惊喜语言）
            if lang not in SURPRISE_LANGUAGES and 'train' in DATASET_URLS[lang]:
                try:
                    train_queries, train_documents, train_qrels = load_miracl_data(lang, "train", force_download=force_download)
                    train_data = create_training_data(train_queries, train_documents, train_qrels)
                    
                    # 保存处理后的train数据（仅写入 datasets/processed）
                    train_dir = Path(f"datasets/processed/{lang}/train")
                    train_dir.mkdir(parents=True, exist_ok=True)
                    train_data.to_pickle(train_dir / "processed_data.pkl")
                    
                    # 合并dev和train数据
                    lang_data = pd.concat([lang_data, train_data], ignore_index=True)
                    print(f"已合并 {lang} 语言的dev数据 ({len(dev_data)} 条) 和 train数据 ({len(train_data)} 条)")
                except Exception as e:
                    print(f"处理 {lang} 的train数据时出错: {e}")
            
            all_data[lang] = lang_data
        except Exception as e:
            print(f"处理 {lang} 数据集时出错: {e}")
            continue
    
    # 如果只处理一种语言，直接返回该语言的数据
    if len(lang_list) == 1:
        return all_data.get(lang_list[0])
    
    # 返回所有语言数据的字典
    return all_data

def load_train_dev_data(lang: str, train_sample_ratio: float = 1.0) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    加载训练和测试数据
    
    Args:
        lang: 语言代码
        train_sample_ratio: 从完整训练集中采样的比例，默认为1.0表示使用全部数据
        
    Returns:
        train_data, test_data
    """
    print(f"加载 {lang} 语言的训练和测试数据...")
    
    # 检查处理好的数据是否存在，不存在则处理
    dev_data_path = Path(f"datasets/processed/{lang}/dev/processed_data.pkl")
    if not dev_data_path.exists():
        process_miracl_dataset([lang])
    
    # 加载测试集(dev数据)
    test_data = pd.read_pickle(dev_data_path)
    
    # 加载训练集
    train_data_path = Path(f"datasets/processed/{lang}/train/processed_data.pkl")
    if not train_data_path.exists():
        process_miracl_dataset([lang])
        if not train_data_path.exists():
            print(f"警告: {lang} 语言没有训练集数据")
            return pd.DataFrame(), test_data
    
    full_train_data = pd.read_pickle(train_data_path)
    
    # 从训练集中采样(如果比例小于1)
    if train_sample_ratio < 1.0:
        train_size = max(1, int(len(full_train_data) * train_sample_ratio))
        train_data = full_train_data.sample(n=train_size)
    else:
        train_data = full_train_data
    
    print(f"数据准备完成: 训练集 {len(train_data)} 样本, 测试集 {len(test_data)} 样本")
    
    return train_data, test_data

def main():
    parser = argparse.ArgumentParser(description="处理MIRACL数据集")
    parser.add_argument("--langs", nargs="+", default=None, help="要处理的语言列表，如ar bn")
    parser.add_argument("--force_download", action="store_true", default=False, help="强制从Hugging Face下载数据集")
    args = parser.parse_args()
    
    process_miracl_dataset(
        args.langs, 
        force_download=args.force_download
    )

if __name__ == "__main__":
    main()
