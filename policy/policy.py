import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms


from policy.transformer import Transformer
from policy.diffusion import DiffusionUNetPolicy
from policy.tokenizer import Enhanced3DEncoder, Sparse3DEncoder

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
        use_relative_action=True
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

    def forward(self, cloud, actions = None, relative_actions=None, batch_size = 24):
        if self.use_relative_action and relative_actions is not None:
            src, pos, src_padding_mask = self.sparse_encoder(cloud, relative_actions, batch_size=batch_size)
        else:
            src, pos, src_padding_mask = self.sparse_encoder(cloud, batch_size=batch_size)

        readout = self.transformer(src, src_padding_mask, self.readout_embed.weight, pos)[-1]
        readout = readout[:, 0]
        if actions is not None:
            loss = self.action_decoder.compute_loss(readout, actions)
            return loss
        else:
            with torch.no_grad():
                action_pred = self.action_decoder.predict_action(readout)
            return action_pred
