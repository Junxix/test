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
from utils.constants import (
    IMG_MEAN, IMG_STD, TRANS_MIN, TRANS_MAX, MAX_GRIPPER_WIDTH,
    WORKSPACE_MIN, WORKSPACE_MAX, SAFE_WORKSPACE_MIN, SAFE_WORKSPACE_MAX,
    SAFE_EPS, GRIPPER_THRESHOLD
)
from dataset.constants import REL_TRANS_MAX, REL_GRIPPER_MAX
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
    "ensemble_mode": "act",
    "use_relative_action": False,  
    "use_perceiver": True 
})


class TCPToImageConverter:
    
    def __init__(self, calib_dir, camera_serial="750612070851"):
        self.camera_serial = camera_serial
        self.img_width = 1280
        self.img_height = 720
        
        tcp_file = os.path.join(calib_dir, 'tcp.npy')
        extrinsics_file = os.path.join(calib_dir, 'extrinsics.npy')
        intrinsics_file = os.path.join(calib_dir, 'intrinsics.npy')
        
        self.tcp_calib = np.load(tcp_file)
        self.extrinsics = np.load(extrinsics_file, allow_pickle=True).item()
        self.intrinsics = np.load(intrinsics_file, allow_pickle=True).item()
        
        self._build_transformation_matrices()
        
    def _quaternion_to_rotation_matrix(self, quaternion):
        qw, qx, qy, qz = quaternion
        R = np.array([
            [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
            [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
            [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]
        ])
        return R

    def _create_transformation_matrix(self, position, quaternion):
        R = self._quaternion_to_rotation_matrix(quaternion)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = position
        return T

    def _build_transformation_matrices(self):
        position = self.tcp_calib[:3]
        quaternion = self.tcp_calib[3:]
        
        M_end_to_base = self._create_transformation_matrix(position, quaternion)
        
        M_cam0433_to_end = np.array([[0, -1, 0, 0],
                                    [1, 0, 0, 0.077],
                                    [0, 0, 1, 0.2665],
                                    [0, 0, 0, 1]])
        
        M_cam0433_to_A = self.extrinsics['043322070878'][0]
        M_cam7506_to_A = self.extrinsics[self.camera_serial][0]
        
        self.M_cam_to_base = M_cam7506_to_A @ np.linalg.inv(M_cam0433_to_A) @ M_cam0433_to_end @ M_end_to_base
        
        self.camera_matrix = self.intrinsics[self.camera_serial]
        
    def tcp_to_normalized_coords(self, tcp_position):
        results = {}
        
        offsets = {'left': -0.019, 'right': 0.019}
        
        for side, offset in offsets.items():
            point = tcp_position.copy()
            point[1] += offset
            
            object_point_world = np.append(point, 1).reshape(-1, 1)
            
            object_point_camera = self.M_cam_to_base @ object_point_world
            
            object_point_pixel = self.camera_matrix @ object_point_camera
            object_point_pixel /= object_point_pixel[2]  
            
            pixel_x = object_point_pixel[0, 0]
            pixel_y = object_point_pixel[1, 0]
            
            normalized_x = pixel_x / self.img_width
            normalized_y = pixel_y / self.img_height
            
            results[side] = np.array([normalized_x, normalized_y])
            
        return results


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

def create_dummy_point_tracks(batch_size=1, num_history=5, num_points=2):
    track_length = num_history + 1
    dummy_tracks = torch.zeros(batch_size, track_length, num_points, 2)
    return dummy_tracks

def visualize_point_tracks(image, track_array, timestep):
    import cv2
    
    img_height, img_width = image.shape[:2]
    
    vis_image = image.copy()
    
    pixel_tracks = track_array.copy()
    pixel_tracks[:, :, 0] *= img_width   
    pixel_tracks[:, :, 1] *= img_height
    pixel_tracks = pixel_tracks.astype(int)
    
    colors_bgr = {
        'left': (255, 0, 0),   
        'right': (0, 0, 255)  
    }
    
    for gripper_idx, (gripper_name, color) in enumerate(zip(['left', 'right'], colors_bgr.values())):
        gripper_track = pixel_tracks[:, gripper_idx, :]  # shape: (seq_len, 2)
        
        # 绘制轨迹线（渐变透明度效果）
        for i in range(1, len(gripper_track)):
            pt1 = tuple(gripper_track[i-1])
            pt2 = tuple(gripper_track[i])
            
            # 渐变透明度：越新的线条越不透明
            alpha = 0.3 + 0.7 * (i / len(gripper_track))
            thickness = max(2, int(3 * alpha))
            
            cv2.line(vis_image, pt1, pt2, color, thickness=thickness)
        
        # 绘制轨迹点
        for i, point in enumerate(gripper_track):
            # 确保坐标在图像范围内
            x, y = point
            x = max(0, min(x, img_width - 1))
            y = max(0, min(y, img_height - 1))
            point = (x, y)
            
            # 调整点的大小，最新的点更大
            if i == len(gripper_track) - 1:  # 最新点
                radius = 8
                thickness = -1  # 填充
                # 添加白色边框
                cv2.circle(vis_image, point, radius + 2, (255, 255, 255), 2)
            else:
                alpha = 0.5 + 0.5 * (i / len(gripper_track))
                radius = max(2, int(4 * alpha))
                thickness = 2 if i < len(gripper_track) - 1 else -1
            
            cv2.circle(vis_image, point, radius, color, thickness)
        
        if len(gripper_track) > 0:
            label_pos = gripper_track[-1]  # 最新点位置
            label_x = max(0, min(label_pos[0] + 15, img_width - 100))
            label_y = max(20, min(label_pos[1] - 10, img_height - 10))
            
            cv2.putText(vis_image, gripper_name.upper(), (label_x, label_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    text_lines = [
        f"Step: {timestep}",
        f"Track Length: {len(track_array)}",
        f"Left: Blue | Right: Red"
    ]
    
    for i, text in enumerate(text_lines):
        y_pos = 30 + i * 25
        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(vis_image, (5, y_pos - text_height - 5), 
                     (text_width + 15, y_pos + 5), (0, 0, 0), -1)
        cv2.putText(vis_image, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    os.makedirs('./eval_visualization/', exist_ok=True)
    save_path = f'./eval_visualization/tracks_step_{timestep:04d}.jpg'
    cv2.imwrite(save_path, vis_image)
    print(f"Saved track visualization to {save_path}")
    
    try:
        cv2.imshow('Gripper Point Tracks', vis_image)
        cv2.waitKey(1)  # 非阻塞显示
    except:
        pass  # 在无GUI环境中忽略显示错误


def evaluate(args_override):
    # load default arguments
    args = deepcopy(default_args)
    for key, value in args_override.items():
        args[key] = value

    # set up device
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    perceiver_config = None
    if args.use_perceiver:
        perceiver_config = {
            'num_points': 2,      
            'input_dim': 2,       
            'patch_size': 4,      
            'embed_dim': 256,
            'query_dim': 512,
            'num_queries': 64,    
            'num_layers': 4,
            'num_heads': 8,
            'ff_dim': 1024,
            'dropout': 0.1,
            'use_rope': True,
            'output_dim': None    
        }

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
        use_relative_action = args.use_relative_action,
        perceiver_config = perceiver_config
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
    
    tcp_converter = TCPToImageConverter(args.calib, agent.camera_serial)
    
    gripper_tracks_history = []  
    
    if args.discretize_rotation:
        last_rot = np.array(agent.ready_rot_6d, dtype = np.float32)
    
    with torch.inference_mode():
        policy.eval()
        prev_width = None
        executed_actions = []
        
        for t in range(args.max_steps):
            robot_tcp = agent.robot.get_tcp_pose()
            tcp_position = robot_tcp[:3]
            
            normalized_coords = tcp_converter.tcp_to_normalized_coords(tcp_position)
            
            current_track = np.array([
                normalized_coords['left'],   # shape: (2,)
                normalized_coords['right']   # shape: (2,)
            ])  # shape: (2, 2)
            
            gripper_tracks_history.append(current_track)
            
            
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
                
                relative_actions = None
                point_tracks = None
                track_lengths = None
                
                if args.use_relative_action:
                    history_relative = compute_history_relative_actions(executed_actions, args.num_history)
                    history_relative_normalized = normalize_relative_action(history_relative)
                    relative_actions = torch.from_numpy(history_relative_normalized).unsqueeze(0).to(device).float()
                
                if args.use_perceiver:
                    if len(gripper_tracks_history) > 0:
                        track_array = np.array(gripper_tracks_history)  # shape: (seq_len, 2, 2)
                        
                        point_tracks = torch.from_numpy(track_array).unsqueeze(0).to(device).float()  # shape: (1, seq_len, 2, 2)
                        track_lengths = torch.tensor([len(gripper_tracks_history)], dtype=torch.long, device=device)
                        colors_bgr = cv2.cvtColor(colors, cv2.COLOR_RGB2BGR)
                        track_array_vis = np.array(gripper_tracks_history)
                        visualize_point_tracks(colors_bgr, track_array_vis, t)
      
                        print(f"Using real point tracks: shape={point_tracks.shape}, length={track_lengths.item()}")
                    else:
                        point_tracks = create_dummy_point_tracks(1, args.num_history).to(device)
                        track_lengths = torch.tensor([args.num_history + 1], dtype=torch.long, device=device)
                        print("Using dummy point tracks")
                
                # predict
                pred_raw_action = policy(
                    cloud=cloud_data, 
                    actions=None, 
                    relative_actions=relative_actions,
                    point_tracks=point_tracks,
                    track_lengths=track_lengths,
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
    parser.add_argument('--use_relative_action', action = 'store_true', help = 'whether to use relative action (should match training config)')
    parser.add_argument('--use_perceiver', action = 'store_true', help = 'whether to use perceiver (should match training config)')

    evaluate(vars(parser.parse_args()))