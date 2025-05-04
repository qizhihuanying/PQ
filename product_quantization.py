import json
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, subvector_dim, hidden_dim=256, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        assert hidden_dim % num_heads == 0, f"隐藏维度 {hidden_dim} 必须能被注意力头数 {num_heads} 整除"
        self.head_dim = hidden_dim // num_heads
        
        self.query = nn.Linear(subvector_dim, hidden_dim)
        self.key = nn.Linear(subvector_dim, hidden_dim)
        self.value = nn.Linear(subvector_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, subvector_dim)
        
        # 将输出层权重和偏置初始化为零
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        
    def forward(self, x):
        # x shape: [batch_size, num_subvectors, subvector_dim]
        batch_size, num_subvectors, _ = x.size()
        
        # 投影查询、键、值 [batch_size, num_subvectors, hidden_dim]
        q = self.query(x)  
        k = self.key(x)    
        v = self.value(x)

        # 将隐藏维度分成多个头 [batch_size, num_subvectors, num_heads, head_dim]
        q = q.view(batch_size, num_subvectors, self.num_heads, self.head_dim)
        k = k.view(batch_size, num_subvectors, self.num_heads, self.head_dim)
        v = v.view(batch_size, num_subvectors, self.num_heads, self.head_dim)
        
        # 转置以便在头上进行批处理计算 [batch_size, num_heads, num_subvectors, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # 注意力分数计算 [batch_size, num_heads, num_subvectors, num_subvectors]
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention_weights = F.softmax(scores, dim=-1)
        
        # 应用注意力权重 [batch_size, num_heads, num_subvectors, head_dim]
        context = torch.matmul(attention_weights, v)
        
        # 转置回原始形状并合并头 [batch_size, num_subvectors, hidden_dim]
        context = context.transpose(1, 2).reshape(batch_size, num_subvectors, self.hidden_dim)
        
        # 输出投影
        bias = self.output(context)  # [batch_size, num_subvectors, subvector_dim]
        
        return bias

class LinearBias(nn.Module):
    def __init__(self, subvector_dim, hidden_dim=256):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.fc1 = nn.Linear(subvector_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, subvector_dim)
        
        # 将输出层权重和偏置初始化为零
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        
    def forward(self, x):
        # x shape: [batch_size, num_subvectors, subvector_dim]
        hidden = F.relu(self.fc1(x))
        bias = self.fc2(hidden)
        return bias

class PQHead(nn.Module):
    def __init__(self, input_dim=768, num_subvectors=128, code_size=32, use_pq=True, init_codebooks=None, 
                 attention_hidden_dim=64, num_attention_heads=4, use_attention=True):
        super().__init__()
        self.input_dim = input_dim
        self.num_subvectors = num_subvectors
        self.code_size = code_size
        self.use_pq = use_pq
        self.use_attention = use_attention

        assert input_dim % num_subvectors == 0, f"Input dimension {input_dim} must be divisible by number of subvectors {num_subvectors}"
        self.subvector_dim = input_dim // num_subvectors
        self.codebooks = nn.Parameter(torch.randn(num_subvectors, code_size, self.subvector_dim))
        
        if use_attention:
            self.bias_module = MultiHeadSelfAttention(self.subvector_dim, hidden_dim=attention_hidden_dim, num_heads=num_attention_heads)
            print(f"初始化模型：使用注意力机制计算偏置，隐藏维度：{attention_hidden_dim}，注意力头数：{num_attention_heads}")
        else:
            self.bias_module = LinearBias(self.subvector_dim, hidden_dim=attention_hidden_dim)
            print(f"初始化模型：使用线性层计算偏置，隐藏维度：{attention_hidden_dim}")
        
        if init_codebooks is not None:
            assert init_codebooks.shape == (num_subvectors, code_size, self.subvector_dim), \
                f"初始化码本的形状 {init_codebooks.shape} 与期望形状 {(num_subvectors, code_size, self.subvector_dim)} 不匹配"
            self.codebooks.data.copy_(init_codebooks)
        else:
            nn.init.normal_(self.codebooks, mean=0.0, std=0.01)

    def forward(self, x):
        batch_size = x.size(0)

        if not self.use_pq:
            if not x.requires_grad:
                x = x.detach().clone().requires_grad_(True)
            return x

        subvectors = x.reshape(batch_size, self.num_subvectors, self.subvector_dim)

        bias = self.bias_module(subvectors)
        enhanced_subvectors = subvectors + bias

        # Compute dot products [batch_size, num_subvectors, code_size]
        dot_products = torch.sum(enhanced_subvectors.unsqueeze(2) * self.codebooks.unsqueeze(0), dim=-1)
        
        if self.training:
            # Compute soft assignment [batch_size, num_subvectors, code_size]
            soft_assignment = F.softmax(dot_products, dim=2)
            
            # Compute soft quantized subvectors [batch_size, num_subvectors, subvector_dim]
            soft_quantized_subvectors = torch.einsum('bmk,mkd->bmd', soft_assignment, self.codebooks)
            
            # Compute discrete indices [batch_size, num_subvectors]
            indices = torch.argmax(dot_products, dim=2)  
            subvec_indices = torch.arange(self.num_subvectors, device=x.device).view(1, -1).repeat(batch_size, 1) # [batch_size, num_subvectors]
            discrete_quantized_subvectors = self.codebooks[subvec_indices, indices, :] # [batch_size, num_subvectors, subvector_dim]

            quantized_subvectors = soft_quantized_subvectors - (soft_quantized_subvectors - discrete_quantized_subvectors).detach()
            
            # Reshape to [batch_size, input_dim]
            quantized = quantized_subvectors.reshape(batch_size, -1)
            
        else:
            # During inference, use discrete quantization only
            indices = torch.argmax(dot_products, dim=2)
            subvec_indices = torch.arange(self.num_subvectors, device=x.device).view(1, -1).repeat(batch_size, 1)
            selected_vectors = self.codebooks[subvec_indices, indices, :]
            quantized = selected_vectors.reshape(batch_size, -1)
            
        return quantized
    
    def save_model(self, output_dir):
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        model_path = output_path / "pq_head_best.pt"
        state = {
            'input_dim': self.input_dim,
            'num_subvectors': self.num_subvectors,
            'code_size': self.code_size,
            'use_pq': self.use_pq,
            'codebooks': self.codebooks.data,
            'use_attention': self.use_attention
        }
        
        if self.use_attention:
            state['attention_state_dict'] = self.bias_module.state_dict()
            state['attention_hidden_dim'] = self.bias_module.hidden_dim
            state['num_attention_heads'] = self.bias_module.num_heads
        else:
            state['linear_state_dict'] = self.bias_module.state_dict()
            state['linear_hidden_dim'] = self.bias_module.hidden_dim
            
        torch.save(state, model_path)

        config = {
            "input_dim": self.input_dim,
            "num_subvectors": self.num_subvectors,
            "code_size": self.code_size,
            "use_pq": self.use_pq,
            "use_attention": self.use_attention
        }
        
        if self.use_attention:
            config["attention_hidden_dim"] = self.bias_module.hidden_dim
            config["num_attention_heads"] = self.bias_module.num_heads
            print(f"保存模型：使用注意力机制计算偏置，隐藏维度：{self.bias_module.hidden_dim}，注意力头数：{self.bias_module.num_heads}")
        else:
            config["linear_hidden_dim"] = self.bias_module.hidden_dim
            print(f"保存模型：使用线性层计算偏置，隐藏维度：{self.bias_module.hidden_dim}")
            
        config_path = output_path / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
            
        print(f"模型已保存到: {model_path}")
        
    @classmethod
    def load_model(cls, path, device):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
        try:
            state = torch.load(path, map_location=device)
            
            # 创建模型实例
            use_attention = state.get('use_attention', True)
            model = cls(
                input_dim=state['input_dim'],
                num_subvectors=state['num_subvectors'],
                code_size=state['code_size'],
                use_pq=state.get('use_pq', True),
                use_attention=use_attention
            ).to(device)
            
            # 加载码本权重
            model.codebooks.data.copy_(state['codebooks'])
            
            # 加载模块参数
            if use_attention:
                if 'attention_hidden_dim' in state:
                    model.bias_module = MultiHeadSelfAttention(
                        model.subvector_dim,
                        hidden_dim=state['attention_hidden_dim'],
                        num_heads=state['num_attention_heads']
                    ).to(device)
                    print(f"加载模型：使用注意力机制计算偏置，隐藏维度：{state['attention_hidden_dim']}，注意力头数：{state['num_attention_heads']}")
                if 'attention_state_dict' in state:
                    model.bias_module.load_state_dict(state['attention_state_dict'])
            else:
                if 'linear_hidden_dim' in state:
                    model.bias_module = LinearBias(
                        model.subvector_dim,
                        hidden_dim=state['linear_hidden_dim']
                    ).to(device)
                    print(f"加载模型：使用线性层计算偏置，隐藏维度：{state['linear_hidden_dim']}")
                if 'linear_state_dict' in state:
                    model.bias_module.load_state_dict(state['linear_state_dict'])
            
            return model
            
        except FileNotFoundError:
            raise FileNotFoundError(f"找不到模型文件: {path}")
        except Exception as e:
            raise RuntimeError(f"加载模型失败: {str(e)}") 