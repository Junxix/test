# run.py

import torch
from model import PointPerceiver
from utils import create_variable_length_batch

if __name__ == "__main__":
    
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
    
    seq1 = torch.randn(8, 32, 2)
    # seq2 = torch.randn(16, 32, 2)
    seq3 = torch.randn(8, 32, 2)
    
    # padded_batch, lengths = create_variable_length_batch([seq1, seq2, seq3])
    padded_batch, lengths = create_variable_length_batch([seq1, seq3])

    with torch.no_grad():
        output = model(padded_batch, lengths)
        print(f"padded_batch.shape: {padded_batch.shape}")
        print(f"lengths: {lengths}")
        print(f"lengths // config['patch_size']: {lengths // config['patch_size']}")
        print(f"output.shape: {output.shape}")
        print(f"output.size(1): {output.size(1)}")
