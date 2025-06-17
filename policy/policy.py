import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms

from policy.transformer import Transformer
from policy.diffusion import DiffusionUNetPolicy
from policy.tokenizer import Enhanced3DEncoder, Sparse3DEncoder
from track.model import PointPerceiver 

class RISE(nn.Module):
    def __init__(
        self, 
        num_action = 20,
        num_history = 5,
        input_dim = 6,
        obs_feature_dim = 512, 
        action_dim = 10, 
        hidden_dim = 512,
        nheads = 8, 
        num_encoder_layers = 4, 
        num_decoder_layers = 1, 
        dim_feedforward = 2048, 
        dropout = 0.1,
        use_relative_action=True,
        gripper_perceiver_config=None,  
        selected_perceiver_config=None 
    ):
        super().__init__()
        num_obs = 1
        self.num_history = num_history
        self.use_relative_action = use_relative_action

        if use_relative_action:
            self.sparse_encoder = Enhanced3DEncoder(input_dim, obs_feature_dim, action_dim, num_history=num_history)
        else:
            self.sparse_encoder = Sparse3DEncoder(input_dim, obs_feature_dim)

        self.transformer = Transformer(hidden_dim, nheads, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout)
        self.action_decoder = DiffusionUNetPolicy(action_dim, num_action, num_obs, obs_feature_dim)
        self.readout_embed = nn.Embedding(1, hidden_dim)

        self.use_dual_perceiver = gripper_perceiver_config is not None and selected_perceiver_config is not None
        
        # 0: pointcloud token, 1: gripper_perceiver token, 2: selected_perceiver token
        self.type_embedding = nn.Embedding(3, hidden_dim)
        
        if self.use_dual_perceiver:
            self.gripper_perceiver = PointPerceiver(**gripper_perceiver_config)
            self.selected_perceiver = PointPerceiver(**selected_perceiver_config)
            
            total_tokens = gripper_perceiver_config['num_queries'] + selected_perceiver_config['num_queries']
            
            gripper_output_dim = gripper_perceiver_config['output_dim']
            if gripper_output_dim is None:
                gripper_output_dim = gripper_perceiver_config['query_dim']
            
            selected_output_dim = selected_perceiver_config['output_dim'] 
            if selected_output_dim is None:
                selected_output_dim = selected_perceiver_config['query_dim']
                
            assert gripper_output_dim == selected_output_dim, \
                f"Gripper and selected perceiver output dims must match: {gripper_output_dim} vs {selected_output_dim}"
            
            self.perceiver_fusion_layer = nn.Linear(gripper_output_dim, hidden_dim)
            
            print(f"Dual Perceiver initialized:")
            print(f"  - Gripper Perceiver: {gripper_perceiver_config['num_queries']} queries, output_dim={gripper_output_dim}")
            print(f"  - Selected Perceiver: {selected_perceiver_config['num_queries']} queries, output_dim={selected_output_dim}") 
            print(f"  - Total tokens: {total_tokens}")
        else:
            self.gripper_perceiver = None
            self.selected_perceiver = None

    def forward(self, cloud, actions = None, relative_actions=None, gripper_tracks=None, selected_tracks=None, 
                gripper_track_lengths=None, selected_track_lengths=None, batch_size = 24):
        
        if self.use_relative_action and relative_actions is not None:
            src, pos, src_padding_mask = self.sparse_encoder(cloud, relative_actions, batch_size=batch_size)
        else:
            src, pos, src_padding_mask = self.sparse_encoder(cloud, batch_size=batch_size)

        point_cloud_type_embed = self.type_embedding(torch.zeros(batch_size, src.size(1), dtype=torch.long, device=src.device))
        pos = pos + point_cloud_type_embed

        if self.use_dual_perceiver and gripper_tracks is not None and selected_tracks is not None:
            gripper_tokens = self.gripper_perceiver(gripper_tracks, lengths=gripper_track_lengths)
            selected_tokens = self.selected_perceiver(selected_tracks, lengths=selected_track_lengths)
            
            # gripper_tokens: (batch_size, num_queries, query_dim)
            # selected_tokens: (batch_size, num_queries, query_dim)
            gripper_tokens = self.perceiver_fusion_layer(gripper_tokens)
            selected_tokens = self.perceiver_fusion_layer(selected_tokens)
            
            gripper_type_embed = self.type_embedding(torch.ones(batch_size, gripper_tokens.size(1), dtype=torch.long, device=src.device))
            gripper_pos = torch.zeros_like(gripper_tokens) + gripper_type_embed
            
            selected_type_embed = self.type_embedding(torch.full((batch_size, selected_tokens.size(1)), 2, dtype=torch.long, device=src.device))
            selected_pos = torch.zeros_like(selected_tokens) + selected_type_embed
            
            perceiver_tokens = torch.cat([gripper_tokens, selected_tokens], dim=1)  # (batch_size, 8, hidden_dim)
            perceiver_pos = torch.cat([gripper_pos, selected_pos], dim=1)
            
            src = torch.cat([src, perceiver_tokens], dim=1)
            pos = torch.cat([pos, perceiver_pos], dim=1)

            perceiver_padding_mask = torch.zeros(
                (batch_size, perceiver_tokens.size(1)),
                dtype=torch.bool, device=src.device
            )
            src_padding_mask = torch.cat([src_padding_mask, perceiver_padding_mask], dim=1)

        elif self.use_dual_perceiver and gripper_tracks is not None:
            gripper_tokens = self.gripper_perceiver(gripper_tracks, lengths=gripper_track_lengths)
            gripper_tokens = self.perceiver_fusion_layer(gripper_tokens)

            gripper_type_embed = self.type_embedding(torch.ones(batch_size, gripper_tokens.size(1), dtype=torch.long, device=src.device))
            gripper_pos = torch.zeros_like(gripper_tokens) + gripper_type_embed

            src = torch.cat([src, gripper_tokens], dim=1)

            pos = torch.cat([pos, gripper_pos], dim=1)

            gripper_padding_mask = torch.zeros(
                (batch_size, gripper_tokens.size(1)),
                dtype=torch.bool, device=src.device
            )
            src_padding_mask = torch.cat([src_padding_mask, gripper_padding_mask], dim=1)

        readout = self.transformer(src, src_padding_mask, self.readout_embed.weight, pos)[-1]
        readout = readout[:, 0]
        if actions is not None:
            loss = self.action_decoder.compute_loss(readout, actions)
            return loss
        else:
            with torch.no_grad():
                action_pred = self.action_decoder.predict_action(readout)
            return action_pred