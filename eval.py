import os
import time
import json
import torch
import argparse
import numpy as np
import open3d as o3d
import torch.nn as nn
import MinkowskiEngine as ME
import matplotlib.pyplot as plt
import torch.distributed as dist

from tqdm import tqdm
from copy import deepcopy
from easydict import EasyDict as edict
from diffusers.optimization import get_cosine_schedule_with_warmup

from policy import RISE
from eval_agent import Agent
from utils.constants import *
from utils.training import set_seed
from dataset.projector import Projector
from utils.ensemble import EnsembleBuffer
from utils.transformation import rotation_transform


default_args = edict({
    "ckpt": None,
    "calib": "calib/",
    "num_action": 20,
    "num_history": 5,  
    "num_inference_step": 20,
    "voxel_size": 0.005,
    "obs_feature_dim": 512,
    "hidden_dim": 512,
    "nheads": 8,
    "num_encoder_layers": 4,
    "num_decoder_layers": 1,
    "dim_feedforward": 2048,
    "dropout": 0.1,
    "max_steps": 300,
    "seed": 233,
    "vis": False,
    "discretize_rotation": True,
    "ensemble_mode": "act"
})


def create_point_cloud(colors, depths, cam_intrinsics, voxel_size = 0.005):
    """
    color, depth => point cloud
    """
    h, w = depths.shape
    fx, fy = cam_intrinsics[0, 0], cam_intrinsics[1, 1]
    cx, cy = cam_intrinsics[0, 2], cam_intrinsics[1, 2]

    colors = o3d.geometry.Image(colors.astype(np.uint8))
    depths = o3d.geometry.Image(depths.astype(np.float32))

    camera_intrinsics = o3d.camera.PinholeCameraIntrinsic(
        width = w, height = h, fx = fx, fy = fy, cx = cx, cy = cy
    )
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        colors, depths, depth_scale = 1.0, convert_rgb_to_intensity = False
    )
    cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, camera_intrinsics)
    cloud = cloud.voxel_down_sample(voxel_size)
    points = np.array(cloud.points).astype(np.float32)
    colors = np.array(cloud.colors).astype(np.float32)

    x_mask = ((points[:, 0] >= WORKSPACE_MIN[0]) & (points[:, 0] <= WORKSPACE_MAX[0]))
    y_mask = ((points[:, 1] >= WORKSPACE_MIN[1]) & (points[:, 1] <= WORKSPACE_MAX[1]))
    z_mask = ((points[:, 2] >= WORKSPACE_MIN[2]) & (points[:, 2] <= WORKSPACE_MAX[2]))
    mask = (x_mask & y_mask & z_mask)
    points = points[mask]
    colors = colors[mask]
    # imagenet normalization
    colors = (colors - IMG_MEAN) / IMG_STD
    # final cloud
    cloud_final = np.concatenate([points, colors], axis = -1).astype(np.float32)
    return cloud_final

def create_batch(coords, feats):
    """
    coords, feats => batch coords, batch feats (batch size = 1)
    """
    coords_batch = [coords]
    feats_batch = [feats]
    coords_batch, feats_batch = ME.utils.sparse_collate(coords_batch, feats_batch)
    return coords_batch, feats_batch

def create_input(colors, depths, cam_intrinsics, voxel_size = 0.005):
    """
    colors, depths => batch coords, batch feats
    """
    cloud = create_point_cloud(colors, depths, cam_intrinsics, voxel_size = voxel_size)
    coords = np.ascontiguousarray(cloud[:, :3] / voxel_size, dtype = np.int32)
    coords_batch, feats_batch = create_batch(coords, cloud)
    return coords_batch, feats_batch, cloud

def unnormalize_action(action):
    action[..., :3] = (action[..., :3] + 1) / 2.0 * (TRANS_MAX - TRANS_MIN) + TRANS_MIN
    action[..., -1] = (action[..., -1] + 1) / 2.0 * MAX_GRIPPER_WIDTH
    return action

def normalize_relative_action(relative_action):
    relative_action = relative_action.copy()
    relative_action[:, :3] = np.clip(relative_action[:, :3], -REL_TRANS_MAX, REL_TRANS_MAX)
    relative_action[:, :3] = relative_action[:, :3] / REL_TRANS_MAX
    
    relative_action[:, -1] = np.clip(relative_action[:, -1], -REL_GRIPPER_MAX, REL_GRIPPER_MAX)
    relative_action[:, -1] = relative_action[:, -1] / REL_GRIPPER_MAX
    
    return relative_action

def rot_diff(rot1, rot2):
    rot1_mat = rotation_transform(
        rot1,
        from_rep = "rotation_6d",
        to_rep = "matrix"
    )
    rot2_mat = rotation_transform(
        rot2,
        from_rep = "rotation_6d",
        to_rep = "matrix"
    )
    diff = rot1_mat @ rot2_mat.T
    diff = np.diag(diff).sum()
    diff = min(max((diff - 1) / 2.0, -1), 1)
    return np.arccos(diff)

def discretize_rotation(rot_begin, rot_end, rot_step_size = np.pi / 16):
    n_step = int(rot_diff(rot_begin, rot_end) // rot_step_size) + 1
    rot_steps = []
    for i in range(n_step):
        rot_i = rot_begin * (n_step - 1 - i) / n_step + rot_end * (i + 1) / n_step
        rot_steps.append(rot_i)
    return rot_steps

def compute_history_relative_actions(action_history, num_history):
    if len(action_history) < 2:
        padding = np.zeros((num_history, 10))
        padding[:, 3:9] = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]) 
        return padding
    
    relative_actions = []
    for i in range(1, len(action_history)):
        rel_action = np.zeros(10)
        
        rel_action[:3] = action_history[i][:3] - action_history[i-1][:3]
        rel_action[3:9] = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        rel_action[9] = action_history[i][9] - action_history[i-1][9]
        
        relative_actions.append(rel_action)
    
    relative_actions = np.array(relative_actions)
    
    if len(relative_actions) < num_history:
        padding_count = num_history - len(relative_actions)
        padding = np.zeros((padding_count, 10))
        padding[:, 3:9] = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        relative_actions = np.concatenate([padding, relative_actions], axis=0)
    elif len(relative_actions) > num_history:
        relative_actions = relative_actions[-num_history:]
    
    return relative_actions


def evaluate(args_override):
    # load default arguments
    args = deepcopy(default_args)
    for key, value in args_override.items():
        args[key] = value

    # set up device
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # policy
    print("Loading policy ...")
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
        dropout = args.dropout,
        use_relative_action=True
    ).to(device)
    n_parameters = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print("Number of parameters: {:.2f}M".format(n_parameters / 1e6))

    # load checkpoint
    assert args.ckpt is not None, "Please provide the checkpoint to evaluate."
    policy.load_state_dict(torch.load(args.ckpt, map_location = device), strict = False)
    print("Checkpoint {} loaded.".format(args.ckpt))

    # evaluation
    agent = Agent(
        robot_ip = "192.168.2.100",
        pc_ip = "192.168.2.35",
        gripper_port = "/dev/ttyUSB0",
        camera_serial = "750612070851"
    )
    projector = Projector(args.calib)
    ensemble_buffer = EnsembleBuffer(mode = args.ensemble_mode)
    
    if args.discretize_rotation:
        last_rot = np.array(agent.ready_rot_6d, dtype = np.float32)
    with torch.inference_mode():
        policy.eval()
        prev_width = None
        executed_actions = []
        for t in range(args.max_steps):
            if t % args.num_inference_step == 0:
                # pre-process inputs
                colors, depths = agent.get_observation()
                coords, feats, cloud = create_input(
                    colors,
                    depths,
                    cam_intrinsics = agent.intrinsics,
                    voxel_size = args.voxel_size
                )
                feats, coords = feats.to(device), coords.to(device)
                cloud_data = ME.SparseTensor(feats, coords)
                
                history_relative = compute_history_relative_actions(executed_actions, args.num_history)
                history_relative_normalized = normalize_relative_action(history_relative)
                history_relative_input = torch.from_numpy(history_relative_normalized).unsqueeze(0).to(device).float()
                
                # predict
                pred_raw_action = policy(
                    cloud_data, 
                    actions=None, 
                    relative_actions=history_relative_input, 
                    batch_size=1
                ).squeeze(0).cpu().numpy()
                
                action = unnormalize_action(pred_raw_action)
                
                # visualization
                if args.vis:
                    import open3d as o3d
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(cloud[:, :3])
                    pcd.colors = o3d.utility.Vector3dVector(cloud[:, 3:] * IMG_STD + IMG_MEAN)
                    tcp_vis_list = []
                    for raw_tcp in action:
                        tcp_vis = o3d.geometry.TriangleMesh.create_sphere(0.01).translate(raw_tcp[:3])
                        tcp_vis_list.append(tcp_vis)
                    o3d.visualization.draw_geometries([pcd, *tcp_vis_list])
                
                action_width = action[..., -1]
                action_camera = np.concatenate([action[..., :-1], action_width[..., np.newaxis]], axis = -1)
                
                ensemble_buffer.add_action(action_camera, t)
                
            step_action_camera = ensemble_buffer.get_action()
            if step_action_camera is None:  
                continue

            executed_action = step_action_camera
            executed_actions.append(executed_action.copy())
            if len(executed_actions) > args.num_history + 1:
                executed_actions.pop(0)

            step_tcp_camera = step_action_camera[:-1]
            step_width = step_action_camera[-1]
            
            # project action to base coordinate
            step_tcp_base = projector.project_tcp_to_base_coord(
                step_tcp_camera, 
                cam = agent.camera_serial, 
                rotation_rep = "rotation_6d"
            )
            
            # safety insurance
            step_tcp_base[..., :3] = np.clip(
                step_tcp_base[..., :3], 
                SAFE_WORKSPACE_MIN + SAFE_EPS, 
                SAFE_WORKSPACE_MAX - SAFE_EPS
            )
            
            # send tcp pose to robot
            if args.discretize_rotation:
                rot_steps = discretize_rotation(last_rot, step_tcp_base[3:], np.pi / 16)
                last_rot = step_tcp_base[3:]
                for rot in rot_steps:
                    step_tcp_base[3:] = rot
                    agent.set_tcp_pose(
                        step_tcp_base, 
                        rotation_rep = "rotation_6d",
                        blocking = True
                    )
            else:
                agent.set_tcp_pose(
                    step_tcp_base,
                    rotation_rep = "rotation_6d",
                    blocking = True
                )
            
            # send gripper width to gripper (thresholding to avoid repeating sending signals to gripper)
            if prev_width is None or abs(prev_width - step_width) > GRIPPER_THRESHOLD:
                agent.set_gripper_width(step_width, blocking = True)
                prev_width = step_width
    
    agent.stop()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', action = 'store', type = str, help = 'checkpoint path', required = True)
    parser.add_argument('--calib', action = 'store', type = str, help = 'calibration path', required = True)
    parser.add_argument('--num_action', action = 'store', type = int, help = 'number of action steps', required = False, default = 20)
    parser.add_argument('--num_history', action = 'store', type = int, help = 'number of history relative actions', required = False, default = 5)
    parser.add_argument('--num_inference_step', action = 'store', type = int, help = 'number of inference query steps', required = False, default = 20)
    parser.add_argument('--voxel_size', action = 'store', type = float, help = 'voxel size', required = False, default = 0.005)
    parser.add_argument('--obs_feature_dim', action = 'store', type = int, help = 'observation feature dimension', required = False, default = 512)
    parser.add_argument('--hidden_dim', action = 'store', type = int, help = 'hidden dimension', required = False, default = 512)
    parser.add_argument('--nheads', action = 'store', type = int, help = 'number of heads', required = False, default = 8)
    parser.add_argument('--num_encoder_layers', action = 'store', type = int, help = 'number of encoder layers', required = False, default = 4)
    parser.add_argument('--num_decoder_layers', action = 'store', type = int, help = 'number of decoder layers', required = False, default = 1)
    parser.add_argument('--dim_feedforward', action = 'store', type = int, help = 'feedforward dimension', required = False, default = 2048)
    parser.add_argument('--dropout', action = 'store', type = float, help = 'dropout ratio', required = False, default = 0.1)
    parser.add_argument('--max_steps', action = 'store', type = int, help = 'max steps for evaluation', required = False, default = 300)
    parser.add_argument('--seed', action = 'store', type = int, help = 'seed', required = False, default = 233)
    parser.add_argument('--vis', action = 'store_true', help = 'add visualization during evaluation')
    parser.add_argument('--discretize_rotation', action = 'store_true', help = 'whether to discretize rotation process.')
    parser.add_argument('--ensemble_mode', action = 'store', type = str, help = 'temporal ensemble mode', required = False, default = 'new')

    evaluate(vars(parser.parse_args()))