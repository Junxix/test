import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from policy.transformer import Transformer
from policy.diffusion import DiffusionUNetPolicy
from policy.tokenizer import Enhanced3DEncoder, Sparse3DEncoder
from track.model import PointPerceiver 

class DINOv2SemanticExtractor(nn.Module):
    def __init__(self, hidden_dim=512, dinov2_model_name='dinov2_vitb14'):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.dinov2 = torch.hub.load('/home/jingjing/.cache/torch/hub/facebookresearch_dinov2_main',  dinov2_model_name, source='local',  force_reload=False)
        self.dinov2.eval()
        
        for param in self.dinov2.parameters():
            param.requires_grad = False
            
        # (vitb14: 768, vits14: 384, vitl14: 1024, vitg14: 1536)
        dinov2_dim = self.dinov2.embed_dim
        
        self.feature_projection = nn.Linear(dinov2_dim, hidden_dim)
        
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)), 
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def extract_semantic_features(self, rgb_images):
        """
        Args:
            rgb_images: (batch_size, 3, H, W)
        Returns:
            features: (batch_size, hidden_dim, h, w)
        """
        batch_size = rgb_images.size(0)
        
        processed_images = torch.stack([self.preprocess(img) for img in rgb_images])
        
        with torch.no_grad():
            features = self.dinov2.forward_features(processed_images)
            patch_features = features['x_norm_patchtokens']  # (batch_size, num_patches, dinov2_dim)
            
        # (DINOv2 ViT-B/14: 224/14 = 16)
        patch_size = 14
        grid_size = 224 // patch_size  # 16
        
        patch_features = patch_features.view(batch_size, grid_size, grid_size, -1)
        patch_features = patch_features.permute(0, 3, 1, 2)  # (batch_size, dinov2_dim, 16, 16)
        
        patch_features = F.interpolate(patch_features, size=(56, 56), mode='bilinear', align_corners=False)
        
        batch_size, dinov2_dim, h, w = patch_features.shape
        patch_features = patch_features.permute(0, 2, 3, 1).contiguous()  # (batch_size, h, w, dinov2_dim)
        patch_features = patch_features.view(-1, dinov2_dim)  # (batch_size * h * w, dinov2_dim)
        
        projected_features = self.feature_projection(patch_features)  # (batch_size * h * w, hidden_dim)
        projected_features = projected_features.view(batch_size, h, w, self.hidden_dim)
        projected_features = projected_features.permute(0, 3, 1, 2)  # (batch_size, hidden_dim, h, w)
        
        return projected_features
    
    def sample_features_at_positions(self, semantic_features, normalized_positions, img_height=720, img_width=1280):
        """
        Args:
            semantic_features: (batch_size, hidden_dim, h, w) 
            normalized_positions: (batch_size, seq_len, num_points, 2) [0,1]
            img_height, img_width: 
        Returns:
            sampled_features: (batch_size, seq_len, num_points, hidden_dim)
        """
        batch_size, hidden_dim, feat_h, feat_w = semantic_features.shape
        batch_size_pos, seq_len, num_points, _ = normalized_positions.shape
        
        assert batch_size == batch_size_pos, "Batch size mismatch"
        
        # normalized_positions: [0,1] -> feature map coordinates
        x_coords = normalized_positions[..., 0] * (feat_w - 1)  # (batch_size, seq_len, num_points)
        y_coords = normalized_positions[..., 1] * (feat_h - 1)  # (batch_size, seq_len, num_points)
        
        # 重塑坐标以便进行grid_sample
        # grid_sample需要坐标范围在[-1,1]
        x_coords_norm = (x_coords / (feat_w - 1)) * 2 - 1
        y_coords_norm = (y_coords / (feat_h - 1)) * 2 - 1
        
        # 创建采样网格
        grid = torch.stack([x_coords_norm, y_coords_norm], dim=-1)  # (batch_size, seq_len, num_points, 2)
        
        # 为了使用grid_sample，需要重塑维度
        total_samples = seq_len * num_points
        grid_reshaped = grid.view(batch_size, 1, total_samples, 2)  # (batch_size, 1, total_samples, 2)
        
        sampled_features = F.grid_sample(
            semantic_features, 
            grid_reshaped, 
            mode='bilinear', 
            padding_mode='border',
            align_corners=True
        )  # (batch_size, hidden_dim, 1, total_samples)
        
        sampled_features = sampled_features.squeeze(2)  # (batch_size, hidden_dim, total_samples)
        sampled_features = sampled_features.permute(0, 2, 1)  # (batch_size, total_samples, hidden_dim)
        sampled_features = sampled_features.view(batch_size, seq_len, num_points, hidden_dim)
        
        return sampled_features


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
        selected_perceiver_config=None,
        use_dinov2_semantic_pos=True,
        dinov2_model_name='dinov2_vitb14'
    ):
        super().__init__()
        num_obs = 1
        self.num_history = num_history
        self.use_relative_action = use_relative_action
        self.use_dinov2_semantic_pos = use_dinov2_semantic_pos

        if use_relative_action:
            self.sparse_encoder = Enhanced3DEncoder(input_dim, obs_feature_dim, action_dim, num_history=num_history)
        else:
            self.sparse_encoder = Sparse3DEncoder(input_dim, obs_feature_dim)

        self.transformer = Transformer(hidden_dim, nheads, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout)
        self.action_decoder = DiffusionUNetPolicy(action_dim, num_action, num_obs, obs_feature_dim)
        self.readout_embed = nn.Embedding(1, hidden_dim)

        self.use_dual_perceiver = gripper_perceiver_config is not None and selected_perceiver_config is not None
        
        if self.use_dinov2_semantic_pos:
            self.semantic_extractor = DINOv2SemanticExtractor(hidden_dim, dinov2_model_name)
            print("use DINOv2 semantic position encoding")
        else:
            self.type_embedding = nn.Embedding(3, hidden_dim)
            print("use type embedding for position encoding")
        
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

    def forward(self, cloud, actions=None, relative_actions=None, gripper_tracks=None, selected_tracks=None, 
                gripper_track_lengths=None, selected_track_lengths=None, rgb_images=None, batch_size=24):
        
        if self.use_relative_action and relative_actions is not None:
            src, pos, src_padding_mask = self.sparse_encoder(cloud, relative_actions, batch_size=batch_size)
        else:
            src, pos, src_padding_mask = self.sparse_encoder(cloud, batch_size=batch_size)

        if self.use_dinov2_semantic_pos:
            pass  
        else:
            point_cloud_type_embed = self.type_embedding(torch.zeros(batch_size, src.size(1), dtype=torch.long, device=src.device))
            pos = pos + point_cloud_type_embed

        if self.use_dual_perceiver and gripper_tracks is not None and selected_tracks is not None:
            # gripper_tokens: (batch_size, num_points, num_queries, query_dim)
            gripper_tokens = self.gripper_perceiver(gripper_tracks, lengths=gripper_track_lengths)
            selected_tokens = self.selected_perceiver(selected_tracks, lengths=selected_track_lengths)
            # print(f"Gripper tokens shape: {gripper_tokens.shape}, Selected tokens shape: {selected_tokens.shape}")
            if len(gripper_tokens.shape) == 4:  # (batch_size, num_points, num_queries_per_point, dim)
                batch_size_g, num_points_g, queries_per_point_g, dim_g = gripper_tokens.shape
                gripper_tokens = gripper_tokens.view(batch_size_g, num_points_g * queries_per_point_g, dim_g)
            
            if len(selected_tokens.shape) == 4:  # (batch_size, num_points, num_queries_per_point, dim)
                batch_size_s, num_points_s, queries_per_point_s, dim_s = selected_tokens.shape
                selected_tokens = selected_tokens.view(batch_size_s, num_points_s * queries_per_point_s, dim_s)
            
            # print(f"Gripper tokens reshaped to: {gripper_tokens.shape}, Selected tokens reshaped to: {selected_tokens.shape}")
            # (batch_size, total_tokens, dim)
            gripper_tokens = self.perceiver_fusion_layer(gripper_tokens)
            selected_tokens = self.perceiver_fusion_layer(selected_tokens)
            
            if self.use_dinov2_semantic_pos and rgb_images is not None:
                semantic_features = self.semantic_extractor.extract_semantic_features(rgb_images)
                

                current_gripper_coords = gripper_tracks[:, -1:, :, :]  # (batch_size, 1, 2, 2)
                current_selected_coords = selected_tracks[:, -1:, :, :]  # (batch_size, 1, 4, 2)
                
                gripper_semantic_features = self.semantic_extractor.sample_features_at_positions(
                    semantic_features, current_gripper_coords
                )  # (batch_size, 1, 2, hidden_dim)
                
                selected_semantic_features = self.semantic_extractor.sample_features_at_positions(
                    semantic_features, current_selected_coords
                )  # (batch_size, 1, 4, hidden_dim)

                # print(f"Gripper semantic features shape: {gripper_semantic_features.shape}, Selected semantic features shape: {selected_semantic_features.shape}")
                
                gripper_semantic_features = gripper_semantic_features.squeeze(1)  # (batch_size, 2, hidden_dim)
                selected_semantic_features = selected_semantic_features.squeeze(1)  # (batch_size, 4, hidden_dim)
                # print(f"Gripper semantic features after squeeze: {gripper_semantic_features.shape}, Selected semantic features after squeeze: {selected_semantic_features.shape}")
                
                gripper_semantic_pos_expanded = gripper_semantic_features.unsqueeze(2).repeat(
                    1, 1, queries_per_point_g, 1
                )  # (batch_size, 2, queries_per_point, hidden_dim)
                gripper_pos = gripper_semantic_pos_expanded.view(
                    batch_size, num_points_g * queries_per_point_g, self.transformer.d_model
                )  # (batch_size, total_gripper_queries, hidden_dim)
                
                selected_semantic_pos_expanded = selected_semantic_features.unsqueeze(2).repeat(
                    1, 1, queries_per_point_s, 1
                )  # (batch_size, 4, queries_per_point, hidden_dim)
                selected_pos = selected_semantic_pos_expanded.view(
                    batch_size, num_points_s * queries_per_point_s, self.transformer.d_model
                )  # (batch_size, total_selected_queries, hidden_dim)
                

                
            else:
                # 使用传统的type embedding
                gripper_type_embed = self.type_embedding(torch.ones(batch_size, gripper_tokens.size(1), dtype=torch.long, device=src.device))
                gripper_pos = torch.zeros_like(gripper_tokens) + gripper_type_embed
                
                selected_type_embed = self.type_embedding(torch.full((batch_size, selected_tokens.size(1)), 2, dtype=torch.long, device=src.device))
                selected_pos = torch.zeros_like(selected_tokens) + selected_type_embed
            
            perceiver_tokens = torch.cat([gripper_tokens, selected_tokens], dim=1)
            perceiver_pos = torch.cat([gripper_pos, selected_pos], dim=1)
            
            src = torch.cat([src, perceiver_tokens], dim=1)
            pos = torch.cat([pos, perceiver_pos], dim=1)

            perceiver_padding_mask = torch.zeros(
                (batch_size, perceiver_tokens.size(1)),
                dtype=torch.bool, device=src.device
            )
            src_padding_mask = torch.cat([src_padding_mask, perceiver_padding_mask], dim=1)

        elif self.use_dual_perceiver and gripper_tracks is not None:
            raise NotImplementedError("Selected tracks must be provided when using dual perceiver.")
            # gripper_tokens = self.gripper_perceiver(gripper_tracks, lengths=gripper_track_lengths)
            
            # if len(gripper_tokens.shape) == 4:  # (batch_size, num_points, num_queries, dim)
            #     batch_size_g, num_points_g, num_queries_g, dim_g = gripper_tokens.shape
            #     gripper_tokens = gripper_tokens.view(batch_size_g, num_points_g * num_queries_g, dim_g)
            
            # gripper_tokens = self.perceiver_fusion_layer(gripper_tokens)

            # if self.use_dinov2_semantic_pos and rgb_images is not None:
            #     semantic_features = self.semantic_extractor.extract_semantic_features(rgb_images)
            #     gripper_semantic_pos = self.semantic_extractor.sample_features_at_positions(
            #         semantic_features, gripper_tracks
            #     )
                
            #     gripper_queries_per_point = num_queries_g // num_points_g
            #     gripper_semantic_pos_expanded = gripper_semantic_pos.unsqueeze(3).repeat(
            #         1, 1, 1, gripper_queries_per_point, 1
            #     ).view(batch_size, -1, self.transformer.d_model)
                
            #     gripper_pos = gripper_semantic_pos_expanded[:, :gripper_tokens.size(1), :]
            # else:
            #     gripper_type_embed = self.type_embedding(torch.ones(batch_size, gripper_tokens.size(1), dtype=torch.long, device=src.device))
            #     gripper_pos = torch.zeros_like(gripper_tokens) + gripper_type_embed

            # src = torch.cat([src, gripper_tokens], dim=1)
            # pos = torch.cat([pos, gripper_pos], dim=1)

            # gripper_padding_mask = torch.zeros(
            #     (batch_size, gripper_tokens.size(1)),
            #     dtype=torch.bool, device=src.device
            # )
            # src_padding_mask = torch.cat([src_padding_mask, gripper_padding_mask], dim=1)

        readout = self.transformer(src, src_padding_mask, self.readout_embed.weight, pos)[-1]
        readout = readout[:, 0]
        if actions is not None:
            loss = self.action_decoder.compute_loss(readout, actions)
            return loss
        else:
            with torch.no_grad():
                action_pred = self.action_decoder.predict_action(readout)
            return action_pred