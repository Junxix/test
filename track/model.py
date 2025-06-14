# model.py

import torch
import torch.nn as nn
from track.layers import PointPatchEmbedding, PerceiverBlock

class PerceiverEncoder(nn.Module):
    def __init__(
        self,
        num_points=32,
        input_dim=2,
        patch_size=4,
        embed_dim=256,
        query_dim=512,
        num_queries=64,
        num_layers=6,
        num_heads=8,
        ff_dim=1024,
        dropout=0.1,
        use_rope=True,
        max_seq_len=2000
    ):
        super().__init__()
        self.use_rope = use_rope
        
        self.point_patch_embed = PointPatchEmbedding(
            num_points=num_points, patch_size=patch_size, in_dim=input_dim, embed_dim=embed_dim
        )
        
        self.queries = nn.Parameter(torch.randn(1, num_queries, query_dim))
        nn.init.xavier_uniform_(self.queries)
        
        self.blocks = nn.ModuleList([
            PerceiverBlock(
                query_dim, embed_dim, num_heads, ff_dim, dropout, 
                use_rope=use_rope, max_seq_len=max_seq_len // patch_size
            ) for _ in range(num_layers)
        ])
        
        self.final_norm = nn.LayerNorm(query_dim)
        
    def create_input_positions(self, num_patches, num_points):
        if not self.use_rope: return None
        patch_positions = torch.arange(num_patches)
        return patch_positions.repeat_interleave(num_points)
        
    def forward(self, points, lengths=None):
        batch_size = points.size(0)
        
        print(points.shape, lengths)
        patches, patch_lengths = self.point_patch_embed(points, lengths)
        print(patches.shape, patch_lengths)
        num_patches, num_points = patches.size(1), patches.size(2)
        
        mask = None
        if patch_lengths is not None:
            mask = torch.arange(num_patches, device=points.device)[None, :] < patch_lengths[:, None]
            mask = mask.unsqueeze(2).expand(-1, -1, num_points).contiguous().view(batch_size, -1)

        input_positions = self.create_input_positions(num_patches, num_points)
        if input_positions is not None:
            input_positions = input_positions.to(points.device)
        
        embedded = patches.reshape(batch_size, -1, patches.size(-1))  # (batch_size, num_patches * num_points, embed_dim)
        
        queries = self.queries.expand(batch_size, -1, -1)
        
        for block in self.blocks:
            queries = block(queries, embedded, mask, input_positions)
        
        return self.final_norm(queries)

class PointPerceiver(nn.Module):
    def __init__(self, output_dim=None, **kwargs):
        super().__init__()
        self.encoder = PerceiverEncoder(**kwargs)
        if output_dim is not None:
            self.output_proj = nn.Linear(kwargs['query_dim'], output_dim)
        else:
            self.output_proj = None
            
    def forward(self, points, lengths=None):
        encoded = self.encoder(points, lengths)
        if self.output_proj is not None:
            encoded = self.output_proj(encoded)
        return encoded