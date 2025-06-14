# utils.py

import torch
from typing import List

def create_variable_length_batch(sequences: List[torch.Tensor], pad_value: float = 0.0):
    batch_size = len(sequences)
    max_seq_len = max(seq.size(0) for seq in sequences)
    num_points = sequences[0].size(1)
    input_dim = sequences[0].size(2)
    
    padded_batch = torch.full(
        (batch_size, max_seq_len, num_points, input_dim),
        pad_value,
        dtype=sequences[0].dtype,
        device=sequences[0].device
    )
    
    lengths = torch.zeros(batch_size, dtype=torch.long, device=sequences[0].device)
    
    for i, seq in enumerate(sequences):
        seq_len = seq.size(0)
        padded_batch[i, :seq_len] = seq
        lengths[i] = seq_len
        
    return padded_batch, lengths