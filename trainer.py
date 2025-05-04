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
from utils.ndcg_utils import calculate_ndcg10

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
        init_pq_path: str = "",
        logger = None,
        attention_hidden_dim: int = 64,
        attention_lr: float = None,
        num_attention_heads: int = 4,
        use_attention: bool = True
    ):
        self.models = models
        self.tokenizers = tokenizers
        self.embedding_funcs = embedding_funcs
        self.device = device
        self.logger = logger
        
        # 初始化码本参数
        init_codebooks = None
        
        # 尝试从预训练路径加载码本
        if init_pq_path and os.path.exists(init_pq_path):
            self.logger.info(f"从路径加载预初始化的PQ头: {init_pq_path}") if self.logger else print(f"从路径加载预初始化的PQ头: {init_pq_path}")
            try:
                state = torch.load(init_pq_path, map_location=device)
                
                # 检查参数是否匹配
                if (state['input_dim'] != input_dim or 
                    state['num_subvectors'] != num_subvectors or 
                    state['code_size'] != code_size):
                    msg = f"参数不匹配: 预训练({state['input_dim']},{state['num_subvectors']},{state['code_size']}) vs 当前({input_dim},{num_subvectors},{code_size})"
                    self.logger.warning(msg) if self.logger else print(msg)
                
                init_codebooks = state['codebooks']
                self.logger.info("成功加载预训练码本") if self.logger else print("成功加载预训练码本")
            except Exception as e:
                msg = f"加载预训练PQ头失败: {e}"
                self.logger.error(msg) if self.logger else print(msg)
        else:
            msg = "采用随机初始化PQ头参数"
            self.logger.info(msg) if self.logger else print(msg)
        
        # 创建PQ头
        self.pq_head = PQHead(
            input_dim=input_dim,
            num_subvectors=num_subvectors,
            code_size=code_size,
            use_pq=use_pq,
            init_codebooks=init_codebooks,
            attention_hidden_dim=attention_hidden_dim,
            num_attention_heads=num_attention_heads,
            use_attention=use_attention
        ).to(device)
        
        self.cos_criterion = nn.CosineEmbeddingLoss()
        
        # 设置模型训练参数
        if num_trainable_layers > 0:
            for model in self.models:
                self._prepare_model_for_training(model, num_trainable_layers)
            
        # 准备优化器参数组
        main_params = list(self.pq_head.parameters())
        if num_trainable_layers > 0:
            for model in self.models:
                main_params.extend(filter(lambda p: p.requires_grad, model.parameters()))
        
        # 排除 bias_module 参数
        bias_module_params = list(self.pq_head.bias_module.parameters())
        bias_module_param_ids = [id(p) for p in bias_module_params]
        main_params = [p for p in main_params if id(p) not in bias_module_param_ids]
        
        self.logger.info(f"为偏置模块设置单独的学习率: {attention_lr}")
        self.optimizer = optim.Adam([
            {"params": main_params, "lr": lr},
            {"params": bias_module_params, "lr": attention_lr}
        ], weight_decay=l2)

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
        data_shuffled = data.sample(frac=1)
        
        # 用于统计所有数据的bias相对于原始值的比例
        total_original_norm = 0.0
        total_bias_norm = 0.0
        total_ratio_sum = 0.0
        total_samples = 0

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
                    # 如果是第一个模型，计算该批次所有样本的bias统计数据
                    if idx == 0:
                        with torch.no_grad():
                            # 对每个样本计算
                            batch_size_actual = emb1.size(0)
                            for j in range(batch_size_actual):
                                # 获取样本的子向量
                                subvectors = emb1[j].reshape(1, self.pq_head.num_subvectors, self.pq_head.subvector_dim)
                                # 计算bias
                                bias = self.pq_head.bias_module(subvectors)
                                # 计算原始范数和bias范数
                                original_norm = torch.norm(subvectors).item()
                                bias_norm = torch.norm(bias).item()
                                ratio = bias_norm / original_norm if original_norm > 0 else 0
                                
                                # 累加统计
                                total_original_norm += original_norm
                                total_bias_norm += bias_norm
                                total_ratio_sum += ratio
                                total_samples += 1
                    
                    # 量化模式，先应用PQ，然后计算余弦损失
                    pq_out1 = self.pq_head(emb1)
                    pq_out2 = self.pq_head(emb2)
                    batch_loss += self.cos_criterion(pq_out1, pq_out2, cos_labels)

            # 反向传播
            loss = batch_loss / (len(self.models) + len(self.embedding_funcs))
            self.optimizer.zero_grad()
            
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.pq_head.parameters(), max_norm=1.0)
            self.optimizer.step()
                
            total_loss += loss.item()
        
        # 计算并打印所有数据的平均值
        if total_samples > 0:
            avg_original_norm = total_original_norm / total_samples
            avg_bias_norm = total_bias_norm / total_samples
            avg_ratio = total_ratio_sum / total_samples
            log_msg = f"平均原始范数: {avg_original_norm:.6f}, 平均Bias范数: {avg_bias_norm:.6f}, 平均比例: {avg_ratio:.6f}"
            self.logger.info(log_msg)

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
        # 创建一个包装函数，以满足ndcg_utils接口要求
        def get_embeddings_wrapper(texts, model_idx):
            return self.get_embeddings(texts, model_idx)
        
        # 添加模型总数属性，以便ndcg_utils识别
        get_embeddings_wrapper.total_model_count = len(self.models) + len(self.embedding_funcs)
        
        # 添加获取模型名称的方法
        def get_model_name(model_idx):
            if model_idx < len(self.models):  # 本地模型
                # 从预训练模型名称中提取
                if hasattr(self.models[model_idx], 'name_or_path'):
                    return self.models[model_idx].name_or_path.split('/')[-1]
                elif hasattr(self.models[model_idx], 'config') and hasattr(self.models[model_idx].config, 'name_or_path'):
                    return self.models[model_idx].config.name_or_path.split('/')[-1]
                else:
                    return f"local_model_{model_idx}"
            else:  # API模型
                api_idx = model_idx - len(self.models)
                if hasattr(self.embedding_funcs[api_idx], '__name__'):
                    return self.embedding_funcs[api_idx].__name__
                elif hasattr(self.embedding_funcs[api_idx], '__self__') and hasattr(self.embedding_funcs[api_idx].__self__, 'func'):
                    return self.embedding_funcs[api_idx].__self__.func.__name__
                else:
                    return f"api_model_{api_idx}"
        
        # 将获取模型名称的方法附加到包装函数
        get_embeddings_wrapper.get_model_name = get_model_name
        
        # 调用工具函数计算NDCG@10
        return calculate_ndcg10(
            data=data,
            get_embeddings=get_embeddings_wrapper,
            pq_head=self.pq_head,
            logger=self.logger
        )

    def train_pq_head(self, train_data, val_data, test_data, epochs, batch_size, output_dir):
        """训练PQ量化模型"""
        # 检查训练集
        if train_data is None or len(train_data) == 0:
            if self.logger:
                self.logger.info("训练集为空，跳过训练")
            if test_data is not None and len(test_data) > 0:
                # 只记录NDCG@10到日志，不返回结果
                ndcg_results = self.calculate_ndcg10(test_data)
                if self.logger:
                    self.logger.info(f"测试集NDCG@10评估: {ndcg_results}")
            return
        
        # 创建输出目录仅用于保存模型
        os.makedirs(output_dir, exist_ok=True)
        best_loss = float('inf')
        best_epoch = -1
        self.logger.info(f"训练设置：use_pq={self.pq_head.use_pq}, training={self.pq_head.training}")
        
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
        
        if test_data is not None and len(test_data) > 0:
            self.logger.info("重新加载最佳模型进行评估...")
            original_use_pq = self.pq_head.use_pq
            # 加载最佳模型并传递回设备
            self.pq_head = PQHead.load_model(
                f"{output_dir}/pq_head_best.pt",
                self.device
            )
            self.pq_head.use_pq = original_use_pq
            self.pq_head.train(False)  # 设置为评估模式
            self.logger.info(f"评估设置：use_pq={self.pq_head.use_pq}, training={self.pq_head.training}")
            ndcg_results = self.calculate_ndcg10(test_data)
            self.logger.info(f"最终NDCG@10评估: {ndcg_results}")


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
    init_pq_path: str = "",
    logger = None,
    attention_hidden_dim: int = 64,
    attention_lr: float = None,
    num_attention_heads: int = 4,
    use_attention: bool = True
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
        init_pq_path=init_pq_path,
        logger=logger,
        attention_hidden_dim=attention_hidden_dim,
        attention_lr=attention_lr,
        num_attention_heads=num_attention_heads,
        use_attention=use_attention
    )
    
    trainer.train_pq_head(
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        epochs=epochs,
        batch_size=batch_size,
        output_dir=output_dir
    )

