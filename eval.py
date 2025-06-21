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
import cv2

from tqdm import tqdm
from copy import deepcopy
from easydict import EasyDict as edict
from diffusers.optimization import get_cosine_schedule_with_warmup

import sys
sys.path.append('/home/ubuntu/git/jingjing/testspace/RISE_add_cotracker/co-tracker')
from cotracker.utils.visualizer import Visualizer
from cotracker.predictor import CoTrackerOnlinePredictor
COTRACKER_AVAILABLE = True


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
    "use_dual_perceiver": True,
    "cotracker_checkpoint": "/home/ubuntu/git/jingjing/testspace/RISE_add_cotracker/scaled_online.pth",
    "use_dinov2_semantic_pos": True,  # 添加DINOv2支持
    "dinov2_model_name": "dinov2_vitb14"  # 添加DINOv2模型名称
})


class PointSelector:
    def __init__(self):
        self.points = []
        self.image = None
        self.original_image = None
        
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(self.points) < 4:
            self.points.append([x, y])
            print(f"choose {len(self.points)}: ({x}, {y})")
            cv2.circle(self.image, (x, y), 8, (0, 255, 0), -1)
            cv2.putText(self.image, str(len(self.points)), (x+10, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow('choose 4 points', self.image)
    
    def select_points(self, image):
        self.original_image = image.copy()
        self.image = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2BGR)
        self.points = []
        
        cv2.imshow('choose 4 points', self.image)
        cv2.setMouseCallback('choose 4 points', self.mouse_callback)
        
        print("please select 4 points on the image")
        print("after selecting 4 points, press any key to continue, press r to reset")
        
        while len(self.points) < 4:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('r'):  
                self.points = []
                self.image = cv2.cvtColor(self.original_image.copy(), cv2.COLOR_RGB2BGR)
                cv2.imshow('choose 4 points', self.image)
                print("already reset, please select 4 points again")
        
        print("already selected 4 points, press any key to continue")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        return self.points


class TCPToImageConverter:
    
    def __init__(self, calib_dir, camera_serial="750612070851", cotracker_checkpoint=None):
        self.camera_serial = camera_serial
        self.img_width = 1280
        self.img_height = 720
        
        self.cotracker = None
        self.cotracker_initialized = False
        self.selected_points_pixel = None
        
        self.point_tracks_history = [] 
        self.last_cotracker_update_frame = -1  

        self.cotracker = CoTrackerOnlinePredictor(checkpoint=cotracker_checkpoint)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cotracker = self.cotracker.to(device)
        self.cotracker_step = self.cotracker.step  

        self.point_selector = PointSelector()
        
        self.frame_buffer = []
        self.current_frame_idx = 0

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
    
    def initialize_tracking(self, first_frame):
        if not self.cotracker:
            raise RuntimeError("no cotracker available, please check the checkpoint path")
            return False
            
        print("initializing CoTracker, please select 4 points on the first frame")
        selected_points = self.point_selector.select_points(first_frame)
        
        if len(selected_points) != 4:
            raise ValueError("must select exactly 4 points")
        
        self.selected_points_pixel = np.array(selected_points)  # shape: (4, 2)
        

        queries = []
        for point in selected_points:
            queries.append([0, point[0], point[1]])  # frame 0

        device = next(self.cotracker.parameters()).device
        self.cotracker_queries = torch.tensor([queries], dtype=torch.float32, device=device)
        print(f"CoTracker queries shape: {self.cotracker_queries.shape}")
        
        frame_tensor = torch.from_numpy(first_frame).permute(2, 0, 1).unsqueeze(0).float().to(device)
        frame_tensor = frame_tensor / 255.0  # [0,1]
        
        self.current_frame_idx = 0
        self.frame_buffer = [frame_tensor]
        
        normalized_points = self.selected_points_pixel.copy()
        normalized_points = normalized_points.astype(np.float64) 
        normalized_points[:, 0] /= self.img_width
        normalized_points[:, 1] /= self.img_height
        self.point_tracks_history = [normalized_points]
        self.last_cotracker_update_frame = 0

        self.cotracker_initialized = True
        return True
    
    def update_tracking(self, new_frame):
        device = next(self.cotracker.parameters()).device
        
        frame_tensor = torch.from_numpy(new_frame).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
        self.frame_buffer.append(frame_tensor)
        self.current_frame_idx += 1
        

        should_run_cotracker = False
        required_frames = self.cotracker_step * 2
        if self.current_frame_idx == 1 or (len(self.frame_buffer) >= required_frames and self.current_frame_idx % self.cotracker.step == 0):
            should_run_cotracker = True
        
        cotracker_ran = False
        
        if should_run_cotracker:
            video_chunk_frames = self.frame_buffer[-required_frames:]
            video_chunk = torch.cat(video_chunk_frames, dim=0).unsqueeze(0)  # (1, T, C, H, W)
            
            is_first_step = (self.current_frame_idx == 1)
        
            with torch.no_grad():
                if is_first_step:
                    result = self.cotracker(
                        video_chunk,
                        is_first_step=True,
                        queries=self.cotracker_queries,
                        grid_size=0,
                    )
                    if result == (None, None):
                        return self._get_current_track(), False
                else:
                    pred_tracks, pred_visibility = self.cotracker(
                        video_chunk,
                        is_first_step=False,
                        grid_size=0,
                    )
                    
                    if pred_tracks is not None:
                        cotracker_ran = self._update_track_history_with_cotracker_result(pred_tracks)
                        return self._get_current_track(), cotracker_ran
        
        return self._get_current_track(), cotracker_ran
    
    def get_gripper_coordinates_for_frame(self, tcp_position, gripper_width, frame_idx):
        results = {}

        offsets = {'left': -1 * gripper_width/2, 'right': gripper_width/2}
        
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
    
    def _update_track_history_with_cotracker_result(self, pred_tracks):
        """
        pred_tracks: shape (1, T, num_points, 2) - T
        """
        tracks_np = pred_tracks[0].cpu().numpy()  # (T, num_points, 2)
        T, num_points, _ = tracks_np.shape
        
        normalized_tracks = tracks_np.copy()
        normalized_tracks[:, :, 0] /= self.img_width
        normalized_tracks[:, :, 1] /= self.img_height
        
        start_frame = self.current_frame_idx - T + 1
        end_frame = self.current_frame_idx
        
        while len(self.point_tracks_history) <= end_frame:
            if len(self.point_tracks_history) > 0:
                self.point_tracks_history.append(self.point_tracks_history[-1].copy())
            else:
                raise RuntimeError("No track history available to fill gaps.")
        
        for i, frame_idx in enumerate(range(start_frame, end_frame + 1)):
            if frame_idx >= 0: 
                self.point_tracks_history[frame_idx] = normalized_tracks[i]
        
        self.last_cotracker_update_frame = self.current_frame_idx
        
        print(f"Updated track history from frame {start_frame} to {end_frame}")
        
        return True
    
    def _get_current_track(self):
        if self.current_frame_idx < len(self.point_tracks_history):
            return self.point_tracks_history[self.current_frame_idx]
        elif len(self.point_tracks_history) > 0:
            return self.point_tracks_history[-1]
        else:
            raise RuntimeError("No track history available")

    def tcp_to_normalized_coords(self, tcp_position, gripper_width, current_frame=None):
        """
        Args:
            tcp_position: TCP position in 3D
            current_frame: current RGB frame for tracking update
            
        Returns:
            tuple: (results_dict, cotracker_ran)
                results_dict: contains 'left', 'right' gripper points and 4 tracked points
                cotracker_ran: boolean indicating if CoTracker was executed this frame
        """
        results = {}
        
        gripper_coords = self.get_gripper_coordinates_for_frame(tcp_position, gripper_width, self.current_frame_idx)
        results.update(gripper_coords)

        cotracker_ran = False
        
        if current_frame is not None and self.cotracker_initialized:
            tracked_points, cotracker_ran = self.update_tracking(current_frame)
            if tracked_points is not None:
                for i in range(4):
                    results[f'tracked_{i}'] = tracked_points[i]
            else:
                raise RuntimeError("Tracking failed, no valid points returned.")
        else:
            raise RuntimeError("Tracking not initialized or no current frame provided.")
            
        return results, cotracker_ran
    
    def get_complete_track_history(self):
        if len(self.point_tracks_history) == 0:
            return []
        
        return self.point_tracks_history


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

def create_dummy_point_tracks(batch_size=1, num_history=5, num_points=6):
    track_length = num_history + 1
    dummy_tracks = torch.zeros(batch_size, track_length, num_points, 2)
    return dummy_tracks

class TrackHistoryManager:

    def __init__(self, tcp_converter):
        self.tcp_converter = tcp_converter
        self.tcp_history = []  
        self.gripper_tracks_history = []  
        
    def add_current_step(self, tcp_position, normalized_coords):
        self.tcp_history.append(tcp_position.copy())
        
        current_track = np.array([
            normalized_coords['left'],       # shape: (2,)
            normalized_coords['right'],      # shape: (2,)
            normalized_coords['tracked_0'],  # shape: (2,)
            normalized_coords['tracked_1'],  # shape: (2,)
            normalized_coords['tracked_2'],  # shape: (2,)
            normalized_coords['tracked_3']   # shape: (2,)
        ])  # shape: (6, 2)
        
        while len(self.gripper_tracks_history) <= len(self.tcp_history) - 1:
            self.gripper_tracks_history.append(np.zeros((6, 2)))
        
        current_frame_idx = len(self.tcp_history) - 1
        self.gripper_tracks_history[current_frame_idx] = current_track
        
    def update_with_cotracker_result(self):
        if len(self.tcp_history) == 0:
            return
            
        cotracker_history = self.tcp_converter.point_tracks_history
        
        print(f"Updating CoTracker points. TCP history length: {len(self.tcp_history)}, CoTracker history length: {len(cotracker_history)}")
        
        for frame_idx in range(len(self.gripper_tracks_history)):
            if frame_idx < len(cotracker_history):
                cotracker_points = cotracker_history[frame_idx]
                
                if frame_idx < len(self.gripper_tracks_history):
                    self.gripper_tracks_history[frame_idx][2:6] = cotracker_points
            
        print(f"Updated CoTracker points in gripper tracks history. History length: {len(self.gripper_tracks_history)}")
    
    def get_track_history(self):
        return self.gripper_tracks_history


def visualize_point_tracks(image, track_array, timestep):
    import cv2
    
    img_height, img_width = image.shape[:2]
    
    vis_image = image.copy()
    
    pixel_tracks = track_array.copy()
    pixel_tracks[:, :, 0] *= img_width   
    pixel_tracks[:, :, 1] *= img_height
    pixel_tracks = pixel_tracks.astype(int)
    
    colors_bgr = [
        (255, 0, 0),   # 左爪
        (0, 0, 255),   # 右爪
        (0, 255, 0),   # 追踪点1
        (0, 255, 128), # 追踪点2
        (0, 128, 255), # 追踪点3
        (128, 255, 0)  # 追踪点4
    ]
    
    point_names = ['left', 'right', 'track1', 'track2', 'track3', 'track4']
    
    for point_idx, (point_name, color) in enumerate(zip(point_names, colors_bgr)):
        if point_idx >= track_array.shape[1]:
            continue
        point_track = pixel_tracks[:, point_idx, :]  
        
        for i in range(1, len(point_track)):
            pt1 = tuple(point_track[i-1])
            pt2 = tuple(point_track[i])
            alpha = 0.3 + 0.7 * (i / len(point_track))
            thickness = max(1, int(2 * alpha))
            cv2.line(vis_image, pt1, pt2, color, thickness=thickness)
        
        for i, point in enumerate(point_track):
            x, y = point
            x = max(0, min(x, img_width - 1))
            y = max(0, min(y, img_height - 1))
            point = (x, y)
            if i == len(point_track) - 1: 
                radius = 6
                thickness = -1
                cv2.circle(vis_image, point, radius + 1, (255, 255, 255), 1)
            else:
                alpha = 0.5 + 0.5 * (i / len(point_track))
                radius = max(2, int(3 * alpha))
                thickness = 1
            
            cv2.circle(vis_image, point, radius, color, thickness)
        
        if len(point_track) > 0:
            label_pos = point_track[-1]
            label_x = max(0, min(label_pos[0] + 10, img_width - 50))
            label_y = max(15, min(label_pos[1] - 5, img_height - 10))
            
            cv2.putText(vis_image, point_name.upper(), (label_x, label_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    text_lines = [
        f"Step: {timestep}",
        f"Track Length: {len(track_array)}",
        f"Points: L|R|CoTracker1-4"
    ]
    
    for i, text in enumerate(text_lines):
        y_pos = 25 + i * 20
        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(vis_image, (5, y_pos - text_height - 3), 
                     (text_width + 10, y_pos + 3), (0, 0, 0), -1)
        cv2.putText(vis_image, text, (8, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    os.makedirs('./eval_visualization/', exist_ok=True)
    save_path = f'./eval_visualization/tracks_step_{timestep:04d}.jpg'
    cv2.imwrite(save_path, vis_image)
    

def evaluate(args_override):
    # load default arguments
    args = deepcopy(default_args)
    for key, value in args_override.items():
        args[key] = value

    # set up device
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    gripper_perceiver_config = None
    selected_perceiver_config = None
    if args.use_dual_perceiver:
        gripper_perceiver_config = {
            'num_points': 2,     
            'input_dim': 2,       
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
            'num_points': 4,      
            'input_dim': 2,       
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
        gripper_perceiver_config = gripper_perceiver_config, 
        selected_perceiver_config = selected_perceiver_config,
        use_dinov2_semantic_pos = args.use_dinov2_semantic_pos,  
        dinov2_model_name = args.dinov2_model_name 
    ).to(device)
    n_parameters = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print("Number of parameters: {:.2f}M".format(n_parameters / 1e6))
    
    if args.use_dinov2_semantic_pos:
        print(f"Using DINOv2 semantic position embedding with model: {args.dinov2_model_name}")
    else:
        print("Using traditional type embedding")

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
    
    tcp_converter = TCPToImageConverter(
        args.calib, 
        agent.camera_serial,
        cotracker_checkpoint=args.cotracker_checkpoint
    )
    
    track_manager = TrackHistoryManager(tcp_converter)
    
    first_frame_initialized = False
    
    if args.discretize_rotation:
        last_rot = np.array(agent.ready_rot_6d, dtype = np.float32)
    
    with torch.inference_mode():
        policy.eval()
        prev_width = None
        executed_actions = []
        
        for t in range(args.max_steps):
            robot_tcp = agent.robot.get_tcp_pose()
            gripper_width = agent.get_gripper_width()
            tcp_position = robot_tcp[:3]
            
            colors, depths = agent.get_observation()
            
            if not first_frame_initialized:
                success = tcp_converter.initialize_tracking(colors)
                if not success:
                    raise RuntimeError("CoTracker initialization failed, please check the checkpoint path.")
                first_frame_initialized = True
            
            normalized_coords, cotracker_ran = tcp_converter.tcp_to_normalized_coords(
                tcp_position, 
                gripper_width,
                current_frame=colors
            )

            track_manager.add_current_step(tcp_position, normalized_coords)
            
            if cotracker_ran:
                track_manager.update_with_cotracker_result()
            
            gripper_tracks_history = track_manager.get_track_history()
            

            if t % args.num_inference_step == 0:
                # pre-process inputs
                coords, feats, cloud = create_input(
                    colors,
                    depths,
                    cam_intrinsics = agent.intrinsics,
                    voxel_size = args.voxel_size
                )
                feats, coords = feats.to(device), coords.to(device)
                cloud_data = ME.SparseTensor(feats, coords)
                
                rgb_images = None
                if args.use_dinov2_semantic_pos:
                    rgb_tensor = torch.from_numpy(colors).permute(2, 0, 1).unsqueeze(0).to(device).float() / 255.0
                    rgb_images = rgb_tensor
                
                relative_actions = None
                point_tracks = None
                track_lengths = None
                
                if args.use_relative_action:
                    history_relative = compute_history_relative_actions(executed_actions, args.num_history)
                    history_relative_normalized = normalize_relative_action(history_relative)
                    relative_actions = torch.from_numpy(history_relative_normalized).unsqueeze(0).to(device).float()
                
                if args.use_dual_perceiver:
                    if len(gripper_tracks_history) > 0:
                        track_array = np.array(gripper_tracks_history)  # shape: (seq_len, 6, 2)
                        
                        gripper_track_array = track_array[:, :2, :]  # 
                        selected_track_array = track_array[:, 2:, :]  # 
                        
                        gripper_tracks = torch.from_numpy(gripper_track_array).unsqueeze(0).to(device).float()
                        selected_tracks = torch.from_numpy(selected_track_array).unsqueeze(0).to(device).float()
                        gripper_track_lengths = torch.tensor([len(gripper_tracks_history)], dtype=torch.long, device=device)
                        selected_track_lengths = torch.tensor([len(gripper_tracks_history)], dtype=torch.long, device=device)
                        
                        colors_bgr = cv2.cvtColor(colors, cv2.COLOR_RGB2BGR)
                        track_array_vis = np.array(gripper_tracks_history)
                        visualize_point_tracks(colors_bgr, track_array_vis, t)
      
                        print(f"Using CoTracker dual tracks: gripper={gripper_tracks.shape}, selected={selected_tracks.shape}")
                    else:
                        gripper_tracks = torch.zeros(1, args.num_history + 1, 2, 2).to(device)
                        selected_tracks = torch.zeros(1, args.num_history + 1, 4, 2).to(device)
                        gripper_track_lengths = torch.tensor([args.num_history + 1], dtype=torch.long, device=device)
                        selected_track_lengths = torch.tensor([args.num_history + 1], dtype=torch.long, device=device)
                        print("Using dummy dual tracks")
                else:
                    gripper_tracks = None
                    selected_tracks = None
                    gripper_track_lengths = None
                    selected_track_lengths = None
                
                # predict
                pred_raw_action = policy(
                    cloud=cloud_data, 
                    actions=None, 
                    relative_actions=relative_actions,
                    gripper_tracks=gripper_tracks,
                    selected_tracks=selected_tracks,
                    gripper_track_lengths=gripper_track_lengths,
                    selected_track_lengths=selected_track_lengths,
                    rgb_images=rgb_images,
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
    cv2.destroyAllWindows()


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
    parser.add_argument('--use_dual_perceiver', action = 'store_true', help = 'whether to use dual perceiver (should match training config)')
    parser.add_argument('--cotracker_checkpoint', action = 'store', type = str, help = 'path to CoTracker checkpoint', required = False, default = "/home/ubuntu/git/jingjing/testspace/RISE_add_cotracker/scaled_online.pth")
    
    parser.add_argument('--use_dinov2_semantic_pos', action = 'store_true', help = 'whether to use DINOv2 semantic position embedding')
    parser.add_argument('--dinov2_model_name', action = 'store', type = str, help = 'DINOv2 model name', 
                        choices=['dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14', 'dinov2_vitg14'], 
                        default = 'dinov2_vitb14')

    evaluate(vars(parser.parse_args()))