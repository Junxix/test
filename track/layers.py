# layers.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import math

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=10000):
        super().__init__()
        self.dim = dim
        assert dim % 2 == 0
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.cached_seq_len = None
        self.cached_cos = None
        self.cached_sin = None
        
    def forward(self, seq_len, device):
        if seq_len != self.cached_seq_len or self.cached_cos is None:
            positions = torch.arange(seq_len, dtype=torch.float, device=device)
            freqs = torch.einsum('i,j->ij', positions, self.inv_freq)
            self.cached_cos = torch.cos(freqs)
            self.cached_sin = torch.sin(freqs)
            self.cached_seq_len = seq_len
        return self.cached_cos, self.cached_sin

class PointPatchEmbedding(nn.Module):
    def __init__(self, num_points, patch_size=4, in_dim=2, embed_dim=256):
        super().__init__()
        self.patch_size = patch_size
        self.conv = nn.Conv1d(in_dim, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True)
        
    def forward(self, points, lengths=None):
        batch_size, seq_len, num_points, _ = points.shape
        patch_size = self.patch_size
        
        if lengths is not None:
            processed_points = []
            updated_lengths = []
            for i in range(batch_size):
                actual_len = lengths[i].item()
                batch_points = points[i, :actual_len] 
                if actual_len % patch_size != 0:
                    pad_len = patch_size - (actual_len % patch_size)
                    padding = batch_points[-1:].repeat(pad_len, 1, 1)
                    batch_points = torch.cat([batch_points, padding], dim=0)
                
                processed_points.append(batch_points)
                updated_lengths.append(batch_points.size(0))
            
            max_padded_len = max(len(bp) for bp in processed_points)
            final_points = torch.zeros(batch_size, max_padded_len, num_points, points.size(-1), 
                                     dtype=points.dtype, device=points.device)
            
            for i, batch_points in enumerate(processed_points):
                final_points[i, :batch_points.size(0)] = batch_points
            
            points = final_points
            lengths = torch.tensor(updated_lengths, dtype=torch.long, device=points.device)
        else:
            if points.size(1) % patch_size != 0:
                pad_len = patch_size - (points.size(1) % patch_size)
                padding = points[:, -1:].repeat(1, pad_len, 1, 1)
                points = torch.cat([points, padding], dim=1)
        
        points_reshaped = rearrange(points, 'b t n c -> (b n) c t')
        patches = self.conv(points_reshaped)
        num_patches = patches.size(-1)
        patches = rearrange(patches, '(b n) c t -> b t n c', b=batch_size, n=num_points)
        if lengths is not None:
            patch_lengths = lengths // patch_size
        else:
            patch_lengths = torch.full((batch_size,), num_patches, dtype=torch.long, device=points.device)
        return patches, patch_lengths

class MultiHeadAttentionWithRoPE(nn.Module):
    def __init__(self, query_dim, key_dim, num_heads=8, dropout=0.1, use_rope=False, max_seq_len=10000):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        self.use_rope = use_rope
        assert query_dim % num_heads == 0
        self.q_linear, self.k_linear, self.v_linear = nn.Linear(query_dim, query_dim), nn.Linear(key_dim, query_dim), nn.Linear(key_dim, query_dim)
        self.out_linear = nn.Linear(query_dim, query_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
        if self.use_rope: self.rope = RotaryPositionalEmbedding(self.head_dim, max_seq_len)
    
    def apply_rope_to_qk(self, q, k, positions_q=None, positions_k=None):
        if not self.use_rope: return q, k
        seq_len_q, seq_len_k, head_dim = q.shape[2], k.shape[2], self.head_dim
        cos_q, sin_q = self.rope.forward(seq_len_q, q.device)
        cos_k, sin_k = self.rope.forward(seq_len_k, k.device)
        if positions_q is not None: cos_q, sin_q = cos_q[positions_q], sin_q[positions_q]
        if positions_k is not None: cos_k, sin_k = cos_k[positions_k], sin_k[positions_k]
        cos_q, sin_q, cos_k, sin_k = map(lambda t: t.unsqueeze(0).unsqueeze(0), (cos_q, sin_q, cos_k, sin_k))
        q1, q2 = q[..., :head_dim//2], q[..., head_dim//2:]
        q_rotated = torch.cat([q1 * cos_q - q2 * sin_q, q1 * sin_q + q2 * cos_q], dim=-1)
        k1, k2 = k[..., :head_dim//2], k[..., head_dim//2:]
        k_rotated = torch.cat([k1 * cos_k - k2 * sin_k, k1 * sin_k + k2 * cos_k], dim=-1)
        return q_rotated, k_rotated
        
    def forward(self, query, key, value, mask=None, query_positions=None, key_positions=None):
        bs, q_len, k_len = query.size(0), query.size(1), key.size(1)
        Q = self.q_linear(query).view(bs, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_linear(key).view(bs, k_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_linear(value).view(bs, k_len, self.num_heads, self.head_dim).transpose(1, 2)
        Q, K = self.apply_rope_to_qk(Q, K, query_positions, key_positions)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        if mask is not None:
            mask = mask.unsqueeze(1).expand(-1, self.num_heads, q_len, k_len)
            scores = scores.masked_fill(mask == 0, -1e9)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        out = torch.matmul(attn_weights, V).transpose(1, 2).contiguous().view(bs, q_len, self.num_heads * self.head_dim)
        return self.out_linear(out)


class PerceiverBlock(nn.Module):
    def __init__(self, query_dim, key_dim, num_heads=8, ff_dim=1024, dropout=0.1, use_rope=False, max_seq_len=10000):
        super().__init__()
        
        self.norm_cross = nn.LayerNorm(query_dim)
        self.cross_attn = MultiHeadAttentionWithRoPE(
            query_dim, key_dim, num_heads, dropout, use_rope=use_rope, max_seq_len=max_seq_len
        )
        
        self.norm_self = nn.LayerNorm(query_dim)
        self.self_attn = MultiHeadAttentionWithRoPE(
            query_dim, query_dim, num_heads, dropout, use_rope=False 
        )
        
        self.norm_ffn = nn.LayerNorm(query_dim)
        self.ffn = nn.Sequential(
            nn.Linear(query_dim, ff_dim),
            nn.GELU(), 
            nn.Dropout(dropout),
            nn.Linear(ff_dim, query_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, queries, inputs, input_mask=None, input_positions=None):
        if input_mask is not None and input_mask.dim() == 2:
            input_mask = input_mask.unsqueeze(1) 

        cross_attn_out = self.cross_attn(
            self.norm_cross(queries), 
            inputs, 
            inputs, 
            mask=input_mask, 
            key_positions=input_positions
        )
        queries = queries + cross_attn_out 
        self_attn_out = self.self_attn(
            self.norm_self(queries), 
            self.norm_self(queries),
            self.norm_self(queries)
        )
        queries = queries + self_attn_out 

        ffn_out = self.ffn(self.norm_ffn(queries)) 
        queries = queries + ffn_out 
        
        return queries


# # layers.py

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from einops import rearrange, repeat
# import math

# class RotaryPositionalEmbedding(nn.Module):
#     def __init__(self, dim, max_seq_len=10000):
#         super().__init__()
#         self.dim = dim
#         assert dim % 2 == 0
#         inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
#         self.register_buffer('inv_freq', inv_freq)
#         self.cached_seq_len = None
#         self.cached_cos = None
#         self.cached_sin = None
        
#     def forward(self, seq_len, device):
#         if seq_len != self.cached_seq_len or self.cached_cos is None:
#             positions = torch.arange(seq_len, dtype=torch.float, device=device)
#             freqs = torch.einsum('i,j->ij', positions, self.inv_freq)
#             self.cached_cos = torch.cos(freqs)
#             self.cached_sin = torch.sin(freqs)
#             self.cached_seq_len = seq_len
#         return self.cached_cos, self.cached_sin

# class PointPatchEmbedding(nn.Module):
#     def __init__(self, num_points, patch_size=4, in_dim=2, embed_dim=256):
#         super().__init__()
#         self.conv = nn.Conv1d(in_dim, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True)
        
#     def forward(self, points, lengths=None):
#         batch_size, seq_len, num_points, _ = points.shape
#         patch_size = self.conv.kernel_size[0]
#         if lengths is not None:
#             effective_lengths = (lengths // patch_size) * patch_size
#             max_effective_len = effective_lengths.max().item()
#             if max_effective_len < seq_len:
#                 points = points[:, :max_effective_len]
#         if points.size(1) % patch_size != 0:
#             pad_len = patch_size - (points.size(1) % patch_size)
#             padding = points[:, -1:].repeat(1, pad_len, 1, 1)
#             points = torch.cat([points, padding], dim=1)
#         points_reshaped = rearrange(points, 'b t n c -> (b n) c t')
#         patches = self.conv(points_reshaped)
#         num_patches = patches.size(-1)
#         patches = rearrange(patches, '(b n) c t -> b t n c', b=batch_size, n=num_points)
#         if lengths is not None:
#             patch_lengths = (lengths // patch_size) * patch_size // patch_size
#         else:
#             patch_lengths = torch.full((batch_size,), num_patches, dtype=torch.long, device=points.device)
#         return patches, patch_lengths

# class MultiHeadAttentionWithRoPE(nn.Module):
#     def __init__(self, query_dim, key_dim, num_heads=8, dropout=0.1, use_rope=False, max_seq_len=10000):
#         super().__init__()
#         self.num_heads = num_heads
#         self.head_dim = query_dim // num_heads
#         self.use_rope = use_rope
#         assert query_dim % num_heads == 0
#         self.q_linear, self.k_linear, self.v_linear = nn.Linear(query_dim, query_dim), nn.Linear(key_dim, query_dim), nn.Linear(key_dim, query_dim)
#         self.out_linear = nn.Linear(query_dim, query_dim)
#         self.dropout = nn.Dropout(dropout)
#         self.scale = math.sqrt(self.head_dim)
#         if self.use_rope: self.rope = RotaryPositionalEmbedding(self.head_dim, max_seq_len)
    
#     def apply_rope_to_qk(self, q, k, positions_q=None, positions_k=None):
#         if not self.use_rope: return q, k
#         seq_len_q, seq_len_k, head_dim = q.shape[2], k.shape[2], self.head_dim
#         cos_q, sin_q = self.rope.forward(seq_len_q, q.device)
#         cos_k, sin_k = self.rope.forward(seq_len_k, k.device)
#         if positions_q is not None: cos_q, sin_q = cos_q[positions_q], sin_q[positions_q]
#         if positions_k is not None: cos_k, sin_k = cos_k[positions_k], sin_k[positions_k]
#         cos_q, sin_q, cos_k, sin_k = map(lambda t: t.unsqueeze(0).unsqueeze(0), (cos_q, sin_q, cos_k, sin_k))
#         q1, q2 = q[..., :head_dim//2], q[..., head_dim//2:]
#         q_rotated = torch.cat([q1 * cos_q - q2 * sin_q, q1 * sin_q + q2 * cos_q], dim=-1)
#         k1, k2 = k[..., :head_dim//2], k[..., head_dim//2:]
#         k_rotated = torch.cat([k1 * cos_k - k2 * sin_k, k1 * sin_k + k2 * cos_k], dim=-1)
#         return q_rotated, k_rotated
        
#     def forward(self, query, key, value, mask=None, query_positions=None, key_positions=None):
#         bs, q_len, k_len = query.size(0), query.size(1), key.size(1)
#         Q = self.q_linear(query).view(bs, q_len, self.num_heads, self.head_dim).transpose(1, 2)
#         K = self.k_linear(key).view(bs, k_len, self.num_heads, self.head_dim).transpose(1, 2)
#         V = self.v_linear(value).view(bs, k_len, self.num_heads, self.head_dim).transpose(1, 2)
#         Q, K = self.apply_rope_to_qk(Q, K, query_positions, key_positions)
#         scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
#         if mask is not None:
#             mask = mask.unsqueeze(1).expand(-1, self.num_heads, q_len, k_len)
#             scores = scores.masked_fill(mask == 0, -1e9)
#         attn_weights = F.softmax(scores, dim=-1)
#         attn_weights = self.dropout(attn_weights)
#         out = torch.matmul(attn_weights, V).transpose(1, 2).contiguous().view(bs, q_len, self.num_heads * self.head_dim)
#         return self.out_linear(out)


# class PerceiverBlock(nn.Module):
#     def __init__(self, query_dim, key_dim, num_heads=8, ff_dim=1024, dropout=0.1, use_rope=False, max_seq_len=10000):
#         super().__init__()
        
#         self.norm_cross = nn.LayerNorm(query_dim)
#         self.cross_attn = MultiHeadAttentionWithRoPE(
#             query_dim, key_dim, num_heads, dropout, use_rope=use_rope, max_seq_len=max_seq_len
#         )
        
#         self.norm_self = nn.LayerNorm(query_dim)
#         self.self_attn = MultiHeadAttentionWithRoPE(
#             query_dim, query_dim, num_heads, dropout, use_rope=False 
#         )
        
#         self.norm_ffn = nn.LayerNorm(query_dim)
#         self.ffn = nn.Sequential(
#             nn.Linear(query_dim, ff_dim),
#             nn.GELU(), 
#             nn.Dropout(dropout),
#             nn.Linear(ff_dim, query_dim),
#             nn.Dropout(dropout)
#         )
        
#     def forward(self, queries, inputs, input_mask=None, input_positions=None):
#         if input_mask is not None and input_mask.dim() == 2:
#             input_mask = input_mask.unsqueeze(1) 

#         cross_attn_out = self.cross_attn(
#             self.norm_cross(queries), 
#             inputs, 
#             inputs, 
#             mask=input_mask, 
#             key_positions=input_positions
#         )
#         queries = queries + cross_attn_out 
#         self_attn_out = self.self_attn(
#             self.norm_self(queries), 
#             self.norm_self(queries),
#             self.norm_self(queries)
#         )
#         queries = queries + self_attn_out 

#         ffn_out = self.ffn(self.norm_ffn(queries)) 
#         queries = queries + ffn_out 
        
#         return queries
