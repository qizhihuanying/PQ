import json
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

class PQHead(nn.Module):
    def __init__(self, input_dim=768, num_subvectors=128, code_size=32, use_pq=True):
        super().__init__()
        self.input_dim = input_dim
        self.num_subvectors = num_subvectors
        self.code_size = code_size
        self.use_pq = use_pq

        assert input_dim % num_subvectors == 0, f"Input dimension {input_dim} must be divisible by number of subvectors {num_subvectors}"
        self.subvector_dim = input_dim // num_subvectors
        self.codebooks = nn.Parameter(torch.randn(num_subvectors, code_size, self.subvector_dim))
        nn.init.normal_(self.codebooks, mean=0.0, std=0.01)

    def forward(self, x):
        batch_size = x.size(0)

        if not self.use_pq:
            if not x.requires_grad:
                x = x.detach().clone().requires_grad_(True)
            return x

        subvectors = x.reshape(batch_size, self.num_subvectors, self.subvector_dim)
        
        # Compute dot products [batch_size, num_subvectors, code_size]
        dot_products = torch.sum(subvectors.unsqueeze(2) * self.codebooks.unsqueeze(0), dim=-1)
        
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
        
        model_path = output_path / "pq_head_full.pt"
        state = {
            'input_dim': self.input_dim,
            'num_subvectors': self.num_subvectors,
            'code_size': self.code_size,
            'use_pq': self.use_pq,
            'codebooks': self.codebooks.data
        }
        torch.save(state, model_path)

        config = {
            "input_dim": self.input_dim,
            "num_subvectors": self.num_subvectors,
            "code_size": self.code_size,
            "use_pq": self.use_pq
        }
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
            model = cls(
                input_dim=state['input_dim'],
                num_subvectors=state['num_subvectors'],
                code_size=state['code_size'],
                use_pq=state.get('use_pq', True)
            ).to(device)
            
            # 加载码本权重
            model.codebooks.data.copy_(state['codebooks'])
            
            return model
            
        except FileNotFoundError:
            raise FileNotFoundError(f"找不到模型文件: {path}")
        except Exception as e:
            raise RuntimeError(f"加载模型失败: {str(e)}") 