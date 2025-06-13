# run.py

import torch
from model import PointPerceiver
from utils import create_variable_length_batch

if __name__ == "__main__":
    
    # 模型配置
    config = {
        'num_points': 32,
        'input_dim': 2,
        'patch_size': 4,
        'embed_dim': 384,
        'query_dim': 512,
        'num_queries': 64,
        'num_layers': 6,
        'num_heads': 8,
        'ff_dim': 1024,
        'dropout': 0.1,
        'use_rope': True,
        'max_seq_len': 10000,
        'output_dim': 256
    }
    
    model = PointPerceiver(**config)
    
    # 创建可变长度的示例数据
    seq1 = torch.randn(12, 32, 2)
    seq2 = torch.randn(16, 32, 2)
    seq3 = torch.randn(8, 32, 2)
    
    padded_batch, lengths = create_variable_length_batch([seq1, seq2, seq3])
    
    # 前向传播
    with torch.no_grad():
        output = model(padded_batch, lengths)
        print(f"输入形状: {padded_batch.shape}")
        print(f"序列长度: {lengths}")
        print(f"Patch 长度: {lengths // config['patch_size']}")
        print(f"输出形状: {output.shape}")
        print(f"输出 tokens: {output.size(1)}")
