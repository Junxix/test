import torch
import torch.nn as nn
from transformers import ViTModel
from torch.nn import functional as F
import torchvision.transforms as transforms

from policy.transformer import Transformer
from policy.diffusion import DiffusionUNetPolicy
from policy.tokenizer import Enhanced3DEncoder, Sparse3DEncoder
from track.model import PointPerceiver 

class DINOFeatureExtractor(nn.Module):
    """DINO特征提取器"""
    
    def __init__(self, hidden_dim=512, model_name='facebook/dino-vitb16'):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # 加载预训练的DINO模型
        self.dino_model = ViTModel.from_pretrained(model_name)
        # 冻结DINO参数（可选）
        for param in self.dino_model.parameters():
            param.requires_grad = False
            
        # 特征投影层
        self.feature_proj = nn.Linear(self.dino_model.config.hidden_size, hidden_dim)
        
        # 图像预处理
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def extract_patch_features(self, rgb_images, coords):
        """
        从图像指定坐标提取DINO patch特征
        
        Args:
            rgb_images: [B, C, H, W] RGB图像
            coords: [B, N, 2] N个点的归一化坐标 (x, y)
        
        Returns:
            features: [B, N, hidden_dim] 每个点的DINO特征
        """
        batch_size, num_points = coords.shape[:2]
        
        # 预处理图像
        if rgb_images.dtype == torch.uint8:
            rgb_images = rgb_images.float() / 255.0
        processed_images = self.preprocess(rgb_images)
        
        # 提取DINO特征
        with torch.no_grad():
            outputs = self.dino_model(processed_images)
            # 使用patch embeddings，去掉CLS token
            patch_features = outputs.last_hidden_state[:, 1:]  # [B, 196, hidden_size]
        
        # 将坐标映射到patch grid
        # DINO ViT-B/16的patch grid是14x14=196
        patch_h, patch_w = 14, 14
        
        # coords是归一化坐标[0,1]，映射到patch indices
        patch_x = (coords[..., 0] * (patch_w - 1)).long().clamp(0, patch_w - 1)
        patch_y = (coords[..., 1] * (patch_h - 1)).long().clamp(0, patch_h - 1)
        patch_indices = patch_y * patch_w + patch_x  # [B, N]
        
        # 提取对应patch的特征
        point_features = []
        for b in range(batch_size):
            batch_patch_features = patch_features[b]  # [196, hidden_size]
            batch_indices = patch_indices[b]  # [N]
            selected_features = batch_patch_features[batch_indices]  # [N, hidden_size]
            point_features.append(selected_features)
        
        point_features = torch.stack(point_features, dim=0)  # [B, N, hidden_size]
        
        # 投影到目标维度
        point_features = self.feature_proj(point_features)  # [B, N, hidden_dim]
        
        return point_features


class DINOPositionEmbedding(nn.Module):
    """将DINO特征作为位置编码的实现"""
    
    def __init__(self, hidden_dim=512, tokens_per_point=4):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.tokens_per_point = tokens_per_point
        
        self.dino_extractor = DINOFeatureExtractor(hidden_dim)
        
        # 为每个点的多个tokens生成不同的位置编码变换
        self.position_diversifier = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Dropout(0.1)
            ) for _ in range(tokens_per_point)
        ])
        
        # token位置区分嵌入
        self.token_position_embed = nn.Parameter(
            torch.randn(tokens_per_point, hidden_dim) * 0.02
        )
        
    def _create_diverse_position_embeddings(self, point_features):
        """
        为每个点创建多个具有多样性的位置编码
        
        Args:
            point_features: [B, N, hidden_dim] N个点的DINO特征
        Returns:
            diverse_positions: [B, N * tokens_per_point, hidden_dim]
        """
        batch_size, num_points, hidden_dim = point_features.shape
        
        all_positions = []
        
        for point_idx in range(num_points):
            point_feature = point_features[:, point_idx]  # [B, hidden_dim]
            point_positions = []
            
            for token_idx in range(self.tokens_per_point):
                # 通过不同的变换网络生成不同的位置编码
                transformed_position = self.position_diversifier[token_idx](point_feature)
                
                # 添加token位置嵌入
                position_with_embed = transformed_position + self.token_position_embed[token_idx]
                
                point_positions.append(position_with_embed)
            
            # 堆叠当前点的所有位置编码
            point_positions = torch.stack(point_positions, dim=1)  # [B, tokens_per_point, hidden_dim]
            all_positions.append(point_positions)
        
        # 拼接所有点的位置编码
        diverse_positions = torch.cat(all_positions, dim=1)  # [B, N * tokens_per_point, hidden_dim]
        
        return diverse_positions
        
    def forward(self, rgb_images, gripper_coords, selected_coords):
        """
        生成DINO位置编码
        
        Args:
            rgb_images: [B, C, H, W] RGB图像
            gripper_coords: [B, 2, 2] 2个gripper点的坐标 (左爪, 右爪)
            selected_coords: [B, 4, 2] 4个selected点的坐标
        
        Returns:
            gripper_pos_embed: [B, 8, hidden_dim] gripper点的位置编码
            selected_pos_embed: [B, 16, hidden_dim] selected点的位置编码
        """
        
        # 1. 提取每个gripper点的DINO特征
        gripper_features = self.dino_extractor.extract_patch_features(
            rgb_images, gripper_coords  # [B, 2, 2]
        )  # 输出: [B, 2, hidden_dim]
        
        # 2. 为gripper点生成多样化的位置编码
        gripper_pos_embed = self._create_diverse_position_embeddings(gripper_features)
        
        # 3. 提取每个selected点的DINO特征
        selected_features = self.dino_extractor.extract_patch_features(
            rgb_images, selected_coords  # [B, 4, 2]
        )  # 输出: [B, 4, hidden_dim]
        
        # 4. 为selected点生成多样化的位置编码
        selected_pos_embed = self._create_diverse_position_embeddings(selected_features)
        
        return gripper_pos_embed, selected_pos_embed


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
        use_dino_position=True
    ):
        super().__init__()
        num_obs = 1
        self.num_history = num_history
        self.use_relative_action = use_relative_action
        self.use_dino_position = use_dino_position

        if use_relative_action:
            self.sparse_encoder = Enhanced3DEncoder(input_dim, obs_feature_dim, action_dim, num_history=num_history)
        else:
            self.sparse_encoder = Sparse3DEncoder(input_dim, obs_feature_dim)

        self.transformer = Transformer(hidden_dim, nheads, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout)
        self.action_decoder = DiffusionUNetPolicy(action_dim, num_action, num_obs, obs_feature_dim)
        self.readout_embed = nn.Embedding(1, hidden_dim)

        self.use_dual_perceiver = gripper_perceiver_config is not None and selected_perceiver_config is not None
        
        # 类型嵌入：0=point cloud, 1=gripper_perceiver, 2=selected_perceiver
        self.type_embedding = nn.Embedding(3, hidden_dim)
        
        if self.use_dual_perceiver:
            self.gripper_perceiver = PointPerceiver(**gripper_perceiver_config)
            self.selected_perceiver = PointPerceiver(**selected_perceiver_config)
            
            gripper_output_dim = gripper_perceiver_config.get('output_dim') or gripper_perceiver_config['query_dim']
            selected_output_dim = selected_perceiver_config.get('output_dim') or selected_perceiver_config['query_dim']
                
            assert gripper_output_dim == selected_output_dim, \
                f"Gripper and selected perceiver output dims must match: {gripper_output_dim} vs {selected_output_dim}"
            
            self.perceiver_fusion_layer = nn.Linear(gripper_output_dim, hidden_dim)
            
            total_tokens = gripper_perceiver_config['num_queries'] + selected_perceiver_config['num_queries']
            
            print(f"Dual Perceiver initialized:")
            print(f"  - Gripper Perceiver: {gripper_perceiver_config['num_queries']} queries, output_dim={gripper_output_dim}")
            print(f"  - Selected Perceiver: {selected_perceiver_config['num_queries']} queries, output_dim={selected_output_dim}") 
            print(f"  - Total tokens: {total_tokens}")
        else:
            self.gripper_perceiver = None
            self.selected_perceiver = None

        # DINO位置编码器
        if self.use_dino_position:
            self.dino_position_encoder = DINOPositionEmbedding(hidden_dim, tokens_per_point=4)

    def forward(self, cloud, actions=None, relative_actions=None, gripper_tracks=None, selected_tracks=None, 
                gripper_track_lengths=None, selected_track_lengths=None, batch_size=24,
                rgb_images=None, gripper_coords=None, selected_coords=None):
        
        # 1. 处理点云特征
        if self.use_relative_action and relative_actions is not None:
            src, pos, src_padding_mask = self.sparse_encoder(cloud, relative_actions, batch_size=batch_size)
        else:
            src, pos, src_padding_mask = self.sparse_encoder(cloud, batch_size=batch_size)

        # 为点云tokens添加类型嵌入
        point_cloud_type_embed = self.type_embedding(torch.zeros(batch_size, src.size(1), dtype=torch.long, device=src.device))
        pos = pos + point_cloud_type_embed

        # 2. 处理PointPerceiver tokens + DINO position embedding
        if self.use_dual_perceiver and gripper_tracks is not None and selected_tracks is not None:
            # 2.1 通过PointPerceiver生成content tokens
            gripper_tokens = self.gripper_perceiver(gripper_tracks, lengths=gripper_track_lengths)
            selected_tokens = self.selected_perceiver(selected_tracks, lengths=selected_track_lengths)
            
            gripper_tokens = self.perceiver_fusion_layer(gripper_tokens)
            selected_tokens = self.perceiver_fusion_layer(selected_tokens)
            
            # 2.2 生成DINO位置编码（如果有DINO数据）
            if self.use_dino_position and rgb_images is not None and gripper_coords is not None and selected_coords is not None:
                dino_gripper_pos, dino_selected_pos = self.dino_position_encoder(
                    rgb_images, gripper_coords, selected_coords
                )
                
                # 检查并调整维度匹配
                gripper_tokens_per_point = dino_gripper_pos.size(1) // gripper_coords.size(1)  # 应该是4
                selected_tokens_per_point = dino_selected_pos.size(1) // selected_coords.size(1)  # 应该是4
                
                # 扩展tokens以匹配DINO位置编码的数量
                gripper_tokens_expanded = gripper_tokens.repeat_interleave(gripper_tokens_per_point, dim=1)
                selected_tokens_expanded = selected_tokens.repeat_interleave(selected_tokens_per_point, dim=1)
                
                perceiver_tokens = torch.cat([gripper_tokens_expanded, selected_tokens_expanded], dim=1)
                dino_pos = torch.cat([dino_gripper_pos, dino_selected_pos], dim=1)

            else:
                # 没有DINO数据，使用传统type embedding
                gripper_type_embed = self.type_embedding(torch.ones(batch_size, gripper_tokens.size(1), dtype=torch.long, device=src.device))
                gripper_pos = torch.zeros_like(gripper_tokens) + gripper_type_embed
                
                selected_type_embed = self.type_embedding(torch.full((batch_size, selected_tokens.size(1)), 2, dtype=torch.long, device=src.device))
                selected_pos = torch.zeros_like(selected_tokens) + selected_type_embed
                
                perceiver_tokens = torch.cat([gripper_tokens, selected_tokens], dim=1)
                dino_pos = torch.cat([gripper_pos, selected_pos], dim=1)
            
            # 2.3 添加到主序列
            src = torch.cat([src, perceiver_tokens], dim=1)
            pos = torch.cat([pos, dino_pos], dim=1)

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