import os
import json
import torch
import argparse
import numpy as np
import torch.nn as nn
import MinkowskiEngine as ME
import matplotlib.pyplot as plt
import torch.distributed as dist

from tqdm import tqdm
from copy import deepcopy
from easydict import EasyDict as edict
from diffusers.optimization import get_cosine_schedule_with_warmup

from policy import RISE
from dataset.realworld import RealWorldDataset, collate_fn
from utils.training import set_seed, plot_history, sync_loss


default_args = edict({
    "data_path": "data/collect_pens",
    "aug": False,
    "aug_jitter": False,
    "num_action": 20,
    "num_history": 5,
    "voxel_size": 0.005,
    "obs_feature_dim": 512,
    "hidden_dim": 512,
    "nheads": 8,
    "num_encoder_layers": 4,
    "num_decoder_layers": 1,
    "dim_feedforward": 2048,
    "dropout": 0.1,
    "ckpt_dir": "logs/collect_pens",
    "resume_ckpt": None,
    "resume_epoch": -1,
    "lr": 3e-4,
    "batch_size": 240,
    "num_epochs": 1000,
    "save_epochs": 50,
    "num_workers": 24,
    "seed": 233,
    "vis_data": False,
    "num_targets": 2 
})


def train(args_override):
    # load default arguments
    args = deepcopy(default_args)
    for key, value in args_override.items():
        args[key] = value

    # prepare distributed training
    torch.multiprocessing.set_sharing_strategy('file_system')
    WORLD_SIZE = int(os.environ['WORLD_SIZE'])
    RANK = int(os.environ['RANK'])
    LOCAL_RANK = int(os.environ['LOCAL_RANK'])
    os.environ['NCCL_P2P_DISABLE'] = '1'
    dist.init_process_group(backend = 'nccl', init_method = 'env://', world_size = WORLD_SIZE, rank = RANK)

    # set up device
    set_seed(args.seed)
    torch.cuda.set_device(LOCAL_RANK)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # dataset & dataloader
    if RANK == 0: print("Loading dataset ...")
    dataset = RealWorldDataset(
        path = args.data_path,
        split = 'train',
        num_obs = 1,
        num_action = args.num_action,
        num_history = args.num_history,
        voxel_size = args.voxel_size,
        aug = args.aug,
        aug_jitter = args.aug_jitter, 
        with_cloud = False,
        vis = args.vis_data,
        num_targets = args.num_targets 
    )
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, 
        num_replicas = WORLD_SIZE, 
        rank = RANK, 
        shuffle = True
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size = args.batch_size // WORLD_SIZE,
        num_workers = args.num_workers,
        collate_fn = collate_fn,
        sampler = sampler,
        drop_last = True
    )

    gripper_perceiver_config = {
        'num_points': 2,      # gripper左右两个点
        'input_dim': 2,       # 2D
        'patch_size': 4,      
        'embed_dim': 256,
        'query_dim': 512,
        'num_queries': 4,     
        'num_layers': 4,      
        'num_heads': 8,
        'ff_dim': 1024,
        'dropout': 0.1,
        'use_rope': True,
        'output_dim': None    
    }
    
    selected_perceiver_config = {
        'num_points': 4,      # 始终是4个点：1个target时从target_1选4个点，2个target时各选2个点
        'input_dim': 2,       # 2D
        'patch_size': 4,      
        'embed_dim': 256,
        'query_dim': 512,
        'num_queries': 4,    
        'num_layers': 4,   
        'num_heads': 8,
        'ff_dim': 1024,
        'dropout': 0.1,
        'use_rope': True,
        'output_dim': None    
    }

    if RANK == 0: 
        print(f"Loading policy with {args.num_targets} targets...")
        if args.num_targets == 1:
            print("Selected perceiver will process 4 points from target_1")
        else:
            print("Selected perceiver will process 4 points (2 from target_1 + 2 from target_2)")

    # 新增：从命令行参数控制是否使用DINOv2语义位置嵌入
    use_dinov2_semantic_pos = getattr(args, 'use_dinov2_semantic_pos', True)
    dinov2_model_name = getattr(args, 'dinov2_model_name', 'dinov2_vitb14')

    policy = RISE(
        num_action = args.num_action,
        num_history = args.num_history,
        input_dim = 6,
        obs_feature_dim = args.obs_feature_dim,
        action_dim = 10,
        hidden_dim = args.hidden_dim,
        nheads = args.nheads,
        num_encoder_layers = args.num_encoder_layers,
        num_decoder_layers = args.num_decoder_layers,
        use_relative_action = False,
        dropout = args.dropout,
        gripper_perceiver_config = gripper_perceiver_config,  
        selected_perceiver_config = selected_perceiver_config,
        use_dinov2_semantic_pos = use_dinov2_semantic_pos,  # 新增
        dinov2_model_name = dinov2_model_name  # 新增
    ).to(device)
    
    if RANK == 0:
        n_parameters = sum(p.numel() for p in policy.parameters() if p.requires_grad)
        print("Number of parameters: {:.2f}M".format(n_parameters / 1e6))
        if use_dinov2_semantic_pos:
            print(f"Using DINOv2 semantic position embedding with model: {dinov2_model_name}")
        else:
            print("Using traditional type embedding")
            
    policy = nn.parallel.DistributedDataParallel(
        policy, 
        device_ids = [LOCAL_RANK], 
        output_device = LOCAL_RANK, 
        find_unused_parameters = True
    )

    # load checkpoint
    if args.resume_ckpt is not None:
        policy.module.load_state_dict(torch.load(args.resume_ckpt, map_location = device), strict = False)
        if RANK == 0:
            print("Checkpoint {} loaded.".format(args.resume_ckpt))

    # ckpt path
    if RANK == 0 and not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)
    
    # optimizer and lr scheduler
    if RANK == 0: print("Loading optimizer and scheduler ...")
    optimizer = torch.optim.AdamW(policy.parameters(), lr = args.lr, betas = [0.95, 0.999], weight_decay = 1e-6)

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer = optimizer,
        num_warmup_steps = 2000,
        num_training_steps = len(dataloader) * args.num_epochs
    )
    lr_scheduler.last_epoch = len(dataloader) * (args.resume_epoch + 1) - 1

    # training
    train_history = []

    policy.train()
    for epoch in range(args.resume_epoch + 1, args.num_epochs):
        if RANK == 0: print("Epoch {}".format(epoch)) 
        sampler.set_epoch(epoch)
        optimizer.zero_grad()
        num_steps = len(dataloader)
        pbar = tqdm(dataloader) if RANK == 0 else dataloader
        avg_loss = 0

        for data in pbar:
            # cloud data processing
            cloud_coords = data['input_coords_list']
            cloud_feats = data['input_feats_list']
            action_data = data['action_normalized']
            relative_action_data = data.get('relative_action_normalized', None)
            gripper_tracks_data = data.get('gripper_tracks', None) 
            selected_tracks_data = data.get('selected_tracks', None)
            gripper_track_lengths_data = data.get('gripper_track_lengths', None) 
            selected_track_lengths_data = data.get('selected_track_lengths', None)
            rgb_images_data = data.get('rgb_image', None)  # 新增：获取RGB图像

            cloud_feats, cloud_coords, action_data = cloud_feats.to(device), cloud_coords.to(device), action_data.to(device)
            if relative_action_data is not None:
                relative_action_data = relative_action_data.to(device)
            if gripper_tracks_data is not None:
                gripper_tracks_data = gripper_tracks_data.to(device)
            if selected_tracks_data is not None:
                selected_tracks_data = selected_tracks_data.to(device)
            if gripper_track_lengths_data is not None:
                gripper_track_lengths_data = gripper_track_lengths_data.to(device)
            if selected_track_lengths_data is not None:
                selected_track_lengths_data = selected_track_lengths_data.to(device)
            if rgb_images_data is not None:  # 新增：处理RGB图像
                rgb_images_data = rgb_images_data.to(device)
            
            cloud_data = ME.SparseTensor(cloud_feats, cloud_coords)
            # forward
            loss = policy(
                cloud=cloud_data, 
                actions=action_data, 
                relative_actions=relative_action_data, 
                gripper_tracks=gripper_tracks_data,      
                selected_tracks=selected_tracks_data,    
                gripper_track_lengths=gripper_track_lengths_data,
                selected_track_lengths=selected_track_lengths_data,
                rgb_images=rgb_images_data,  # 新增：传递RGB图像
                batch_size=action_data.shape[0]
            )
            
            # backward
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()
            avg_loss += loss.item()

        avg_loss = avg_loss / num_steps
        sync_loss(avg_loss, device)
        train_history.append(avg_loss)

        if RANK == 0:
            print("Train loss: {:.6f}".format(avg_loss))
            if (epoch + 1) % args.save_epochs == 0:
                torch.save(
                    policy.module.state_dict(),
                    os.path.join(args.ckpt_dir, "policy_epoch_{}_seed_{}.ckpt".format(epoch + 1, args.seed))
                )
                plot_history(train_history, epoch, args.ckpt_dir, args.seed)

    if RANK == 0:
        torch.save(
            policy.module.state_dict(),
            os.path.join(args.ckpt_dir, "policy_last.ckpt")
        )
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', action = 'store', type = str, help = 'data path', required = True)
    parser.add_argument('--aug', action = 'store_true', help = 'whether to add 3D data augmentation')
    parser.add_argument('--aug_jitter', action = 'store_true', help = 'whether to add color jitter augmentation')
    parser.add_argument('--num_action', action = 'store', type = int, help = 'number of action steps', required = False, default = 20)
    parser.add_argument('--num_history', action = 'store', type = int, help = 'number of history actions', required = False, default = 5)
    parser.add_argument('--voxel_size', action = 'store', type = float, help = 'voxel size', required = False, default = 0.005)
    parser.add_argument('--obs_feature_dim', action = 'store', type = int, help = 'observation feature dimension', required = False, default = 512)
    parser.add_argument('--hidden_dim', action = 'store', type = int, help = 'hidden dimension', required = False, default = 512)
    parser.add_argument('--nheads', action = 'store', type = int, help = 'number of heads', required = False, default = 8)
    parser.add_argument('--num_encoder_layers', action = 'store', type = int, help = 'number of encoder layers', required = False, default = 4)
    parser.add_argument('--num_decoder_layers', action = 'store', type = int, help = 'number of decoder layers', required = False, default = 1)
    parser.add_argument('--dim_feedforward', action = 'store', type = int, help = 'feedforward dimension', required = False, default = 2048)
    parser.add_argument('--dropout', action = 'store', type = float, help = 'dropout ratio', required = False, default = 0.1)
    parser.add_argument('--ckpt_dir', action = 'store', type = str, help = 'checkpoint directory', required = True)
    parser.add_argument('--resume_ckpt', action = 'store', type = str, help = 'resume checkpoint file', required = False, default = None)
    parser.add_argument('--resume_epoch', action = 'store', type = int, help = 'resume from which epoch', required = False, default = -1)
    parser.add_argument('--lr', action = 'store', type = float, help = 'learning rate', required = False, default = 3e-4)
    parser.add_argument('--batch_size', action = 'store', type = int, help = 'batch size', required = False, default = 240)
    parser.add_argument('--num_epochs', action = 'store', type = int, help = 'training epochs', required = False, default = 1000)
    parser.add_argument('--save_epochs', action = 'store', type = int, help = 'saving epochs', required = False, default = 50)
    parser.add_argument('--num_workers', action = 'store', type = int, help = 'number of workers', required = False, default = 24)
    parser.add_argument('--seed', action = 'store', type = int, help = 'seed', required = False, default = 233)
    parser.add_argument('--vis_data', action = 'store_true', help = 'whether to visualize the input data and ground truth actions.')
    parser.add_argument('--num_targets', action = 'store', type = int, help = 'number of targets to use (1 or 2)', required = False, default = 2)
    
    # 新增参数
    parser.add_argument('--use_dinov2_semantic_pos', action = 'store_true', help = 'whether to use DINOv2 semantic position embedding')
    parser.add_argument('--dinov2_model_name', action = 'store', type = str, help = 'DINOv2 model name', 
                        choices=['dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14', 'dinov2_vitg14'], 
                        default = 'dinov2_vitb14')

    train(vars(parser.parse_args()))