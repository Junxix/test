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
        perceiver_config=None 
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

        if perceiver_config:
            self.perceiver_encoder = PointPerceiver(**perceiver_config)
            # todo
            self.perceiver_fusion_layer = nn.Linear(perceiver_config['query_dim'], hidden_dim)
        else:
            self.perceiver_encoder = None


    def forward(self, cloud, actions = None, relative_actions=None, point_tracks=None, track_lengths=None, batch_size = 24):
        if self.use_relative_action and relative_actions is not None:
            src, pos, src_padding_mask = self.sparse_encoder(cloud, relative_actions, batch_size=batch_size)
        else:
            src, pos, src_padding_mask = self.sparse_encoder(cloud, batch_size=batch_size)

        if self.perceiver_encoder is not None and point_tracks is not None:
            perceiver_tokens = self.perceiver_encoder(point_tracks, lengths=track_lengths)
            perceiver_tokens = self.perceiver_fusion_layer(perceiver_tokens)

            src = torch.cat([src, perceiver_tokens], dim=1)

            perceiver_pos = torch.zeros_like(perceiver_tokens)
            pos = torch.cat([pos, perceiver_pos], dim=1)

            perceiver_padding_mask = torch.zeros(
                (batch_size, perceiver_tokens.size(1)),
                dtype=torch.bool, device=src.device
            )
            src_padding_mask = torch.cat([src_padding_mask, perceiver_padding_mask], dim=1)

        readout = self.transformer(src, src_padding_mask, self.readout_embed.weight, pos)[-1]
        readout = readout[:, 0]
        if actions is not None:
            loss = self.action_decoder.compute_loss(readout, actions)
            return loss
        else:
            with torch.no_grad():
                action_pred = self.action_decoder.predict_action(readout)
            return action_pred
