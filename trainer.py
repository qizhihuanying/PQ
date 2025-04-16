import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from typing import List, Callable, Dict
import numpy as np
from sklearn.metrics import ndcg_score
from collections import defaultdict
import pandas as pd
import os

from product_quantization import PQHead

class MultiModelTrainer:
    def __init__(
        self,
        models: List[nn.Module],
        tokenizers: List[any],
        embedding_funcs: List[Callable],
        device: torch.device,
        num_trainable_layers: int = 0,
        lr: float = 2e-5,
        l2: float = 0.0,
        use_pq: bool = True,
        input_dim: int = 768,
        num_subvectors: int = 8,
        code_size: int = 256,
        logger = None
    ):
        self.models = models
        self.tokenizers = tokenizers
        self.embedding_funcs = embedding_funcs
        self.device = device
        self.pq_head = PQHead(
            input_dim=input_dim,
            num_subvectors=num_subvectors,
            code_size=code_size,
            use_pq=use_pq
        ).to(device)
        self.cos_criterion = nn.CosineEmbeddingLoss()
        self.logger = logger
        
        # 设置模型训练参数
        if num_trainable_layers > 0:
            for model in self.models:
                self._prepare_model_for_training(model, num_trainable_layers)
            
        # 准备优化器
        params = list(self.pq_head.parameters())
        if num_trainable_layers > 0:
            for model in self.models:
                params.extend(filter(lambda p: p.requires_grad, model.parameters()))
        self.optimizer = optim.Adam(params, lr=lr, weight_decay=l2)

    def _prepare_model_for_training(self, model, num_trainable_layers):
        """冻结或解冻基础模型层"""
        for param in model.parameters():
            param.requires_grad = False
        
        if num_trainable_layers > 0 and hasattr(model, 'encoder') and hasattr(model.encoder, 'layer'):
            for layer in model.encoder.layer[-num_trainable_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True
        elif num_trainable_layers > 0 and hasattr(model, 'layers'):
            for layer in model.layers[-num_trainable_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True

    def get_embeddings(self, texts, model_idx):
        """获取文本的embedding向量"""
        if model_idx < len(self.models):  # 使用本地模型
            model = self.models[model_idx]
            tokenizer = self.tokenizers[model_idx]
            
            enc = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
            with torch.no_grad():
                out = model(**enc)
                emb = torch.nn.functional.normalize(out[0][:, 0], p=2, dim=1)
            emb = emb.detach().clone().requires_grad_(True)
            return emb
        else:  # 使用API方式获取embedding
            func_idx = model_idx - len(self.models)
            embeddings = self.embedding_funcs[func_idx](prompts=texts)
            emb = torch.tensor(embeddings, device=self.device, requires_grad=True)
            return emb

    def train_epoch(self, data, batch_size):
        """训练一个epoch"""
        # 设置模型状态
        for model in self.models:
            if any(p.requires_grad for p in model.parameters()):
                model.train()
            else:
                model.eval()
        self.pq_head.train()
        
        # 训练一个epoch
        total_loss = 0.0
        num_batches = max(int(len(data) / batch_size + 0.99), 1)
        data_shuffled = data.sample(frac=1, random_state=42)

        for i in tqdm(range(num_batches), desc="训练中"):
            batch = data_shuffled[i * batch_size : (i + 1) * batch_size]
            batch_loss = 0
            
            # 对每个模型计算embeddings并累积loss
            for idx in range(len(self.models) + len(self.embedding_funcs)):
                emb1 = self.get_embeddings(list(batch["sample1"]), idx)
                emb2 = self.get_embeddings(list(batch["sample2"]), idx)
                
                # 计算标签
                cos_labels = torch.tensor([1 if rel > 0 else -1 for rel in batch["relevance"].values]).float().to(self.device)
                
                if not self.pq_head.use_pq:
                    # 非量化模式
                    batch_loss += self.cos_criterion(emb1, emb2, cos_labels)
                else:
                    # 量化模式，先应用PQ，然后计算余弦损失
                    pq_out1 = self.pq_head(emb1)
                    pq_out2 = self.pq_head(emb2)
                    batch_loss += self.cos_criterion(pq_out1, pq_out2, cos_labels)

            # 反向传播
            loss = batch_loss / (len(self.models) + len(self.embedding_funcs))
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.pq_head.parameters(), max_norm=1.0)
            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / num_batches

    def eval_epoch(self, data, batch_size):
        """评估一个epoch"""
        self.pq_head.eval()
        for model in self.models:
            model.eval()
            
        total_loss = 0.0
        num_batches = max(int(len(data) / batch_size + 0.99), 1)

        with torch.no_grad():
            for i in tqdm(range(num_batches), desc="评估中"):
                batch = data[i * batch_size : (i + 1) * batch_size]
                batch_loss = 0
                
                for idx in range(len(self.models) + len(self.embedding_funcs)):
                    emb1 = self.get_embeddings(list(batch["sample1"]), idx)
                    emb2 = self.get_embeddings(list(batch["sample2"]), idx)
                    
                    cos_labels = torch.tensor([1 if rel > 0 else -1 for rel in batch["relevance"].values]).float().to(self.device)
                    
                    if not self.pq_head.use_pq:
                        batch_loss += self.cos_criterion(emb1, emb2, cos_labels)
                    else:
                        pq_out1 = self.pq_head(emb1)
                        pq_out2 = self.pq_head(emb2)
                        batch_loss += self.cos_criterion(pq_out1, pq_out2, cos_labels)

                total_loss += batch_loss.item() / (len(self.models) + len(self.embedding_funcs))

        return total_loss / num_batches
        
    def calculate_ndcg10(self, data: pd.DataFrame) -> dict:
        
        self.pq_head.eval()
        for model in self.models:
            model.eval()

        results = {}

        # 如果有 lang 列，就按语言区分，否则统一当做 'all'
        if 'lang' in data.columns:
            languages = data['lang'].unique()
        else:
            languages = ['all']
            data['lang'] = 'all'

        for lang in languages:
            lang_data = data[data['lang'] == lang]
            results[lang] = {}

            # 模型数量（本地模型 + embedding_funcs）
            total_model_count = len(self.models) + len(self.embedding_funcs)

            for model_idx in range(total_model_count):
                model_name = f"model_{model_idx}"
                self.logger.info(f"开始计算 {lang} 语言模型 {model_idx} 的NDCG@10")
                
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
                # 在实际应用中，你可能需要加载完整的语料库文件
                corpus_docs = documents  # 使用已有的所有文档作为语料库
                
                self.logger.info(f"收集到 {len(queries)} 个唯一查询和 {len(corpus_docs)} 个语料库文档")
                
                # 批量计算所有查询的embedding
                query_embeddings = {}  # 查询ID -> embedding
                batch_size = 32  # 可以根据GPU内存调整
                
                # 将查询ID和文本转换为列表，以便批处理
                query_ids = list(queries.keys())
                query_texts = [queries[qid] for qid in query_ids]
                
                # 批量计算查询embedding
                for i in tqdm(range(0, len(query_texts), batch_size), desc="计算查询embedding"):
                    batch_texts = query_texts[i:i+batch_size]
                    batch_ids = query_ids[i:i+batch_size]
                    
                    with torch.no_grad():
                        batch_embs = self.get_embeddings(batch_texts, model_idx)
                        if self.pq_head.use_pq:
                            batch_embs = self.pq_head(batch_embs)
                    
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
                        batch_embs = self.get_embeddings(batch_texts, model_idx)
                        if self.pq_head.use_pq:
                            batch_embs = self.pq_head(batch_embs)
                    
                    # 将embedding保存到字典中
                    for j, did in enumerate(batch_ids):
                        corpus_embeddings[did] = batch_embs[j:j+1]
                
                # 计算每个查询的NDCG@10
                ndcg_list = []
                
                for query_id in tqdm(queries.keys(), desc="计算NDCG@10"):
                    if query_id not in query_embeddings:
                        self.logger.info(f"警告: 查询 {query_id} 没有embedding")
                        continue
                    
                    query_emb = query_embeddings[query_id]
                    
                    # 计算查询与语料库中每个文档的相似度
                    doc_similarities = []  # [(doc_id, similarity), ...]
                    
                    for doc_id, doc_emb in corpus_embeddings.items():
                        if self.pq_head.use_pq:
                            # 使用余弦相似度计算PQ向量之间的相似度
                            sim = torch.nn.functional.cosine_similarity(query_emb, doc_emb).item()
                        else:
                            # 非量化向量使用余弦相似度
                            sim = torch.nn.functional.cosine_similarity(query_emb, doc_emb).item()
                        doc_similarities.append((doc_id, sim))
                    
                    # 按相似度降序排序，当相似度相等时，使用相关性作为第二排序键
                    doc_similarities.sort(key=lambda x: (x[1], qrels.get(query_id, {}).get(x[0], 0)), reverse=True)
                    
                    # 取前K个进行评估
                    k = 10
                    
                    # 构建真实相关性列表（对所有文档，未标注的文档相关性为0）
                    y_true = []
                    y_score = []
                    for doc_id, sim in doc_similarities:
                        relevance = qrels.get(query_id, {}).get(doc_id, 0)
                        y_true.append(relevance)
                        y_score.append(sim)
                    
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
                        if len(ndcg_list) <= 1:  # 只打印前1个查询的详细信息
                            self.logger.info(f"查询ID: {query_id}")
                            self.logger.info(f"查询文本: {queries[query_id]}")
                            self.logger.info(f"Top {k} 结果:")
                            for i, (doc_id, sim) in enumerate(doc_similarities[:k]):
                                relevance = qrels.get(query_id, {}).get(doc_id, 0)
                                doc_snippet = corpus_docs[doc_id][:50] + "..." if len(corpus_docs[doc_id]) > 50 else corpus_docs[doc_id]
                                self.logger.info(f"  {i+1}. 文档ID: {doc_id}, 相关性: {relevance}, 相似度: {sim:.4f}")
                                self.logger.info(f"     文档片段: {doc_snippet}")
                            self.logger.info(f"NDCG@{k}: {ndcg_val:.4f}")
                        
                    except Exception as e:
                        self.logger.info(f"查询 {query_id} 计算NDCG出错: {e}")
                        continue
                
                # 计算该模型在当前语言上的平均NDCG@10
                if ndcg_list:
                    avg_ndcg = float(np.mean(ndcg_list))
                    results[lang][model_name] = avg_ndcg
                    self.logger.info(f"{lang} 语言模型 {model_idx} 的平均NDCG@10: {avg_ndcg:.4f}")
                else:
                    results[lang][model_name] = 0.0
                    self.logger.info(f"{lang} 语言模型 {model_idx} 没有有效的NDCG@10结果")

        return results

    def evaluate_retrieval(self, data: pd.DataFrame, batch_size: int) -> Dict[str, Dict[str, float]]:
        """评估检索性能"""
        return self.calculate_ndcg10(data)

    def train_pq_head(self, train_data, val_data, test_data, epochs, batch_size, output_dir):
        """训练PQ量化模型"""
        # 检查训练集
        if train_data is None or len(train_data) == 0:
            if self.logger:
                self.logger.info("训练集为空，跳过训练")
            if test_data is not None and len(test_data) > 0:
                return self.evaluate_retrieval(test_data, batch_size)
            return {}
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        best_loss = float('inf')
        best_epoch = -1
        
        # 训练循环
        for epoch in range(epochs):
            # 训练一个epoch
            train_loss = self.train_epoch(train_data, batch_size)
            log_msg = f"Epoch {epoch+1}/{epochs}, 训练损失: {train_loss:.4f}"
            
            # 验证
            if val_data is not None and len(val_data) > 0:
                val_loss = self.eval_epoch(val_data, batch_size)
                log_msg += f", 验证损失: {val_loss:.4f}"
                
                # 保存最佳模型
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_epoch = epoch
                    self.pq_head.save_model(output_dir)
                    if self.logger:
                        self.logger.info(f"保存最佳模型 (Epoch {epoch+1})")
            else:
                # 每个epoch保存
                self.pq_head.save_model(output_dir)
            
            # 输出日志
            if self.logger:
                self.logger.info(log_msg)
            else:
                print(log_msg)
        
        # 保存最终模型
        if best_epoch == -1:
            self.pq_head.save_model(output_dir)
        
        # 评估测试集
        if test_data is not None and len(test_data) > 0:
            eval_results = self.evaluate_retrieval(test_data, batch_size)
            if self.logger:
                self.logger.info(f"最终评估结果: {eval_results}")
            return eval_results
        return {}

def train(
    models: List[nn.Module],
    tokenizers: List[any],
    embedding_funcs: List[Callable],
    train_data,
    val_data=None,
    test_data=None,
    device: torch.device = None,
    epochs: int = 10,
    lr: float = 2e-5,
    l2: float = 0.0,
    batch_size: int = 32,
    num_trainable_layers: int = 0,
    output_dir: str = "project/models/pq_head",
    use_pq: bool = True,
    input_dim: int = 768,
    num_subvectors: int = 8,
    code_size: int = 256,
    logger = None
):
    """训练主函数"""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    trainer = MultiModelTrainer(
        models=models,
        tokenizers=tokenizers,
        embedding_funcs=embedding_funcs,
        device=device,
        num_trainable_layers=num_trainable_layers,
        lr=lr,
        l2=l2,
        use_pq=use_pq,
        input_dim=input_dim,
        num_subvectors=num_subvectors,
        code_size=code_size,
        logger=logger
    )
    
    results = trainer.train_pq_head(
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        epochs=epochs,
        batch_size=batch_size,
        output_dir=output_dir
    )
    
    return results, trainer.pq_head
        
