import torch
from torch import nn
from policy.minkowski.resnet import ResNet14


class Sparse3DEncoder(torch.nn.Module):
    def __init__(self, input_dim = 6, output_dim = 512):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.cloud_encoder = ResNet14(in_channels=input_dim, out_channels=output_dim, conv1_kernel_size=3, dilations=(1,1,1,1), bn_momentum=0.02)
        self.position_embedding = SparsePositionalEncoding(output_dim)

    def forward(self, sinput, max_num_token=100, batch_size=24):
        ''' max_num_token: maximum token number for each point cloud, which can be adjusted depending on the scene density.
                           100 for voxel_size=0.005 in our experiments
        '''
        soutput = self.cloud_encoder(sinput)
        feats_batch, coords_batch = soutput.F, soutput.C
        feats_list = []
        coords_list = []
        for i in range(batch_size):
            mask = (coords_batch[:,0] == i)
            feats_list.append(feats_batch[mask])
            coords_list.append(coords_batch[mask])
        pos_list = self.position_embedding(coords_list)

        tokens = torch.zeros([batch_size, max_num_token, self.output_dim], dtype=feats_batch.dtype, device=feats_batch.device)
        pos_emb = torch.zeros([batch_size, max_num_token, self.output_dim], dtype=feats_batch.dtype, device=feats_batch.device)
        token_padding_mask = torch.ones([batch_size, max_num_token], dtype=torch.bool, device=feats_batch.device)
        for i, (feats, pos) in enumerate(zip(feats_list, pos_list)):
            num_token = min(max_num_token, len(feats))
            tokens[i,:num_token] = feats[:num_token]
            pos_emb[i,:num_token] = pos[:num_token]
            token_padding_mask[i,:num_token] = False
        
        return tokens, pos_emb, token_padding_mask



class HistoryActionEncoder(torch.nn.Module):

    def __init__(self, action_dim=10, output_dim=512, num_history=5):
        super().__init__()
        self.action_dim = action_dim
        self.output_dim = output_dim
        self.num_history = num_history
        
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, output_dim // 2),
            nn.ReLU(),
            nn.Linear(output_dim // 2, output_dim),
            nn.LayerNorm(output_dim)
        )
        
        self.temporal_pos_embedding = nn.Parameter(
            torch.randn(1, num_history, output_dim)
        )
    
    def forward(self, history_relative_actions):
        if history_relative_actions is None:
            return None
            
        hist_action_tokens = self.action_encoder(history_relative_actions)
        hist_seq_len = hist_action_tokens.size(1)
        
        hist_action_tokens = hist_action_tokens + self.temporal_pos_embedding[:, -hist_seq_len:]
        
        return hist_action_tokens


class Enhanced3DEncoder(torch.nn.Module):
    def __init__(self, input_dim=6, output_dim=512, action_dim=10, num_history=5):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.action_dim = action_dim
        self.num_history = num_history

        self.cloud_encoder = Sparse3DEncoder(input_dim, output_dim)
        self.history_action_encoder = HistoryActionEncoder(action_dim, output_dim, num_history)

    def forward(self, sinput, history_relative_actions=None, max_num_token=100, batch_size=24):
        cloud_tokens, cloud_pos_emb, cloud_padding_mask = self.cloud_encoder(
            sinput, max_num_token, batch_size
        )
        
        if history_relative_actions is not None:
            hist_action_tokens = self.history_action_encoder(history_relative_actions)
            hist_seq_len = hist_action_tokens.size(1)
            
            total_seq_len = hist_seq_len + max_num_token
            
            all_tokens = torch.zeros([batch_size, total_seq_len, self.output_dim],
                                dtype=cloud_tokens.dtype, device=cloud_tokens.device)
            all_pos_emb = torch.zeros([batch_size, total_seq_len, self.output_dim],
                                    dtype=cloud_tokens.dtype, device=cloud_tokens.device)
            all_padding_mask = torch.ones([batch_size, total_seq_len],
                                        dtype=torch.bool, device=cloud_tokens.device)
            
            all_tokens[:, :hist_seq_len] = hist_action_tokens
            all_pos_emb[:, :hist_seq_len] = 0  
            all_padding_mask[:, :hist_seq_len] = False
            
            all_tokens[:, hist_seq_len:] = cloud_tokens
            all_pos_emb[:, hist_seq_len:] = cloud_pos_emb
            all_padding_mask[:, hist_seq_len:] = cloud_padding_mask
            
            return all_tokens, all_pos_emb, all_padding_mask
        else:
            return cloud_tokens, cloud_pos_emb, cloud_padding_mask


class SparsePositionalEncoding(nn.Module):
    """
    Sparse positional encoding for point tokens, similar to the standard version
    """
    def __init__(self, num_pos_feats=64, temperature=10000, max_pos=800):
        super().__init__()
        ''' max_pos: position range will be [-max_pos/2, max_pos/2) along X/Y/Z-axis.
                     remeber to keep this value fixed in your training and evaluation.
                     800 for voxel_size=0.005 in our experiments.
        '''
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.max_pos = max_pos
        self.origin_pos = max_pos // 2
        self._init_position_vector()

    def _init_position_vector(self):
        x_steps = y_steps = self.num_pos_feats // 3
        z_steps = self.num_pos_feats - x_steps - y_steps
        xyz_embed = torch.arange(self.max_pos, dtype=torch.float32)[:,None]

        x_dim_t = torch.arange(x_steps, dtype=torch.float32)
        y_dim_t = torch.arange(y_steps, dtype=torch.float32)
        z_dim_t = torch.arange(z_steps, dtype=torch.float32)
        x_dim_t = self.temperature ** (2 * (x_dim_t // 2) / x_steps)
        y_dim_t = self.temperature ** (2 * (y_dim_t // 2) / y_steps)
        z_dim_t = self.temperature ** (2 * (z_dim_t // 2) / z_steps)

        pos_x_vector = xyz_embed / x_dim_t
        pos_y_vector = xyz_embed / y_dim_t
        pos_z_vector = xyz_embed / z_dim_t
        self.pos_x_vector = torch.stack([pos_x_vector[:,0::2].sin(), pos_x_vector[:,1::2].cos()], dim=2).flatten(1)
        self.pos_y_vector = torch.stack([pos_y_vector[:,0::2].sin(), pos_y_vector[:,1::2].cos()], dim=2).flatten(1)
        self.pos_z_vector = torch.stack([pos_z_vector[:,0::2].sin(), pos_z_vector[:,1::2].cos()], dim=2).flatten(1)

    def forward(self, coords_list):
        pos_list = []
        for coords in coords_list:
            coords = (coords[:,1:4] + self.origin_pos).long()
            coords[:,0] = torch.clamp(coords[:,0], 0, self.max_pos-1)
            coords[:,1] = torch.clamp(coords[:,1], 0, self.max_pos-1)
            coords[:,2] = torch.clamp(coords[:,2], 0, self.max_pos-1)
            pos_x = self.pos_x_vector.to(coords.device)[coords[:,0]]
            pos_y = self.pos_y_vector.to(coords.device)[coords[:,1]]
            pos_z = self.pos_z_vector.to(coords.device)[coords[:,2]]
            pos = torch.cat([pos_x, pos_y, pos_z], dim=1)
            pos_list.append(pos)
        return pos_list