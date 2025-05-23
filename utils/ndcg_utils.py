import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Dict, List, Callable, Any

def calculate_ndcg10(
    data: pd.DataFrame,
    get_embeddings: Callable,
    pq_head,
    logger=None
) -> dict:
    """
    计算NDCG@10指标
    
    Args:
        data: 包含查询和文档的DataFrame，必须有以下列：query_id, doc_id, sample1(查询), sample2(文档), relevance
        get_embeddings: 获取embedding的函数，接收文本列表和模型索引作为参数
        pq_head: PQ量化头模型
        logger: 日志记录器
        
    Returns:
        dict: 语言到模型到NDCG@10分数的映射字典
    """
    results = {}

    # 如果有lang列，就按语言区分，否则统一当做'all'
    if 'lang' in data.columns:
        languages = data['lang'].unique()
    else:
        languages = ['all']
        data['lang'] = 'all'

    for lang in languages:
        lang_data = data[data['lang'] == lang]
        results[lang] = {}

        # 计算模型数量（通过get_embeddings的实现确定）
        model_count = get_embeddings.total_model_count if hasattr(get_embeddings, 'total_model_count') else 1

        for model_idx in range(model_count):
            # 获取真实的模型名称（如果get_embeddings提供）
            if hasattr(get_embeddings, 'get_model_name'):
                model_name = get_embeddings.get_model_name(model_idx)
            else:
                model_name = f"model_{model_idx}"
                
            if logger:
                logger.info(f"开始计算 {lang} 语言使用{model_name}的NDCG@10")
            
            # 收集所有唯一的查询和文档
            queries = {}         # 查询ID -> 查询文本
            documents = {}       # 文档ID -> 文档文本
            qrels = {}           # 查询ID -> {文档ID: 相关性}
            
            # 1. 首先收集所有唯一的查询和标注的文档关系
            for _, row in tqdm(lang_data.iterrows(), desc="收集查询和标注数据", total=len(lang_data)):
                query_id = row['query_id']
                doc_id = row['doc_id']
                query_text = row['sample1']
                doc_text = row['sample2']
                relevance = row['relevance']
                
                queries[query_id] = query_text
                documents[doc_id] = doc_text
                
                if query_id not in qrels:
                    qrels[query_id] = {}
                qrels[query_id][doc_id] = relevance
            
            # 2. 收集语料库中所有文档（这里我们使用已有数据集中的所有文档作为语料库）
            corpus_docs = documents  # 使用已有的所有文档作为语料库
            
            if logger:
                logger.info(f"收集到 {len(queries)} 个唯一查询和 {len(corpus_docs)} 个语料库文档")
            
            # 批量计算所有查询的embedding
            query_embeddings = {}  # 查询ID -> embedding
            batch_size = 64  # 可以根据GPU内存调整
            
            # 将查询ID和文本转换为列表，以便批处理
            query_ids = list(queries.keys())
            query_texts = [queries[qid] for qid in query_ids]
            
            # 批量计算查询embedding
            for i in tqdm(range(0, len(query_texts), batch_size), desc="计算查询embedding"):
                batch_texts = query_texts[i:i+batch_size]
                batch_ids = query_ids[i:i+batch_size]
                
                with torch.no_grad():
                    batch_embs = get_embeddings(batch_texts, model_idx)
                    if pq_head.use_pq:
                        batch_embs = pq_head(batch_embs)
                
                # 将embedding保存到字典中
                for j, qid in enumerate(batch_ids):
                    query_embeddings[qid] = batch_embs[j:j+1]
            
            # 批量计算所有语料库文档的embedding
            corpus_embeddings = {}  # 文档ID -> embedding
            
            # 将文档ID和文本转换为列表，以便批处理
            corpus_doc_ids = list(corpus_docs.keys())
            corpus_doc_texts = [corpus_docs[did] for did in corpus_doc_ids]
            
            # 批量计算文档embedding
            for i in tqdm(range(0, len(corpus_doc_texts), batch_size), desc="计算语料库文档embedding"):
                batch_texts = corpus_doc_texts[i:i+batch_size]
                batch_ids = corpus_doc_ids[i:i+batch_size]
                
                with torch.no_grad():
                    batch_embs = get_embeddings(batch_texts, model_idx)
                    if pq_head.use_pq:
                        batch_embs = pq_head(batch_embs)
                
                # 将embedding保存到字典中
                for j, did in enumerate(batch_ids):
                    corpus_embeddings[did] = batch_embs[j:j+1]
            
            # 将corpus_embeddings转换为tensor用于向量化计算
            corpus_emb_tensor = torch.cat([corpus_embeddings[did] for did in corpus_doc_ids])
            if torch.cuda.is_available():
                corpus_emb_tensor = corpus_emb_tensor.to('cuda')
            
            # 计算每个查询的NDCG@10
            ndcg_list = []
            
            for query_id in tqdm(queries.keys(), desc="计算NDCG@10"):
                if query_id not in query_embeddings:
                    if logger:
                        logger.info(f"警告: 查询 {query_id} 没有embedding")
                    continue
                
                query_emb = query_embeddings[query_id]
                if torch.cuda.is_available():
                    query_emb = query_emb.to('cuda')
                
                # 使用向量化操作计算查询与所有文档的相似度
                sim_scores = torch.nn.functional.cosine_similarity(query_emb, corpus_emb_tensor, dim=1)
                sim_scores = sim_scores.detach().cpu().numpy()
                
                # 获取排序后的文档ID
                sorted_indices = np.argsort(sim_scores)[::-1]
                sorted_doc_ids = [corpus_doc_ids[idx] for idx in sorted_indices]
                sorted_sim_scores = sim_scores[sorted_indices]
                
                # 取前K个进行评估
                k = 10
                
                # 构建真实相关性列表和预测相似度列表
                y_true = [qrels.get(query_id, {}).get(did, 0) for did in sorted_doc_ids]
                y_score = sorted_sim_scores.tolist()
                
                # 计算NDCG@10
                try:
                    k = min(k, len(y_true))
                    # 计算DCG
                    dcg = 0
                    for i in range(k):
                        dcg += y_true[i] / np.log2(i + 2) 
                    
                    # 计算IDCG
                    ideal_relevance = sorted(y_true, reverse=True)
                    idcg = 0
                    for i in range(k):
                        if i < len(ideal_relevance):
                            idcg += ideal_relevance[i] / np.log2(i + 2)
                    
                    # 计算NDCG
                    ndcg_val = dcg / idcg if idcg > 0 else 0.0
                    ndcg_list.append(ndcg_val)
                    
                    # 打印一些样本的详细信息以便调试
                    if logger and len(ndcg_list) <= 1:  # 只打印前1个查询的详细信息
                        logger.info(f"查询ID: {query_id}")
                        logger.info(f"查询文本: {queries[query_id]}")
                        logger.info(f"Top {k} 结果:")
                        for i, did in enumerate(sorted_doc_ids[:k]):
                            relevance = qrels.get(query_id, {}).get(did, 0)
                            sim = y_score[i]
                            doc_snippet = corpus_docs[did][:50] + "..." if len(corpus_docs[did]) > 50 else corpus_docs[did]
                            logger.info(f"  {i+1}. 文档ID: {did}, 相关性: {relevance}, 相似度: {sim:.4f}")
                            logger.info(f"     文档片段: {doc_snippet}")
                        logger.info(f"NDCG@{k}: {ndcg_val:.4f}")
                    
                except Exception as e:
                    if logger:
                        logger.info(f"查询 {query_id} 计算NDCG出错: {e}")
                    continue
            
            # 计算该模型在当前语言上的平均NDCG@10
            if ndcg_list:
                avg_ndcg = float(np.mean(ndcg_list))
                results[lang][model_name] = avg_ndcg
                if logger:
                    logger.info(f"{lang}语言使用{model_name}的平均NDCG@10: {avg_ndcg:.4f}")
            else:
                results[lang][model_name] = 0.0
                if logger:
                    logger.info(f"{lang}语言使用{model_name}没有有效的NDCG@10结果")

    # 计算并记录所有语言的平均NDCG@10
    all_ndcg_values = []
    for lang_results in results.values():
        for model_score in lang_results.values():
            all_ndcg_values.append(model_score)
            
    if all_ndcg_values and logger:
        overall_avg_ndcg = sum(all_ndcg_values) / len(all_ndcg_values)
        logger.info(f"所有语言平均NDCG@10: {overall_avg_ndcg:.4f}")

    return results