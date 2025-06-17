import os
import json
import torch
import numpy as np
import open3d as o3d
import MinkowskiEngine as ME
import torchvision.transforms as T
import collections.abc as container_abcs
from typing import List

from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset

from dataset.constants import *
from dataset.projector import Projector
from utils.transformation import rot_trans_mat, apply_mat_to_pose, apply_mat_to_pcd, xyz_rot_transform


def create_variable_length_batch(sequences: List[torch.Tensor], pad_value: float = 0.0):
    """处理变长序列的batch化"""
    batch_size = len(sequences)
    max_seq_len = max(seq.size(0) for seq in sequences)
    num_points = sequences[0].size(1)
    input_dim = sequences[0].size(2)
    
    padded_batch = torch.full(
        (batch_size, max_seq_len, num_points, input_dim),
        pad_value,
        dtype=sequences[0].dtype,
        device=sequences[0].device
    )
    
    lengths = torch.zeros(batch_size, dtype=torch.long, device=sequences[0].device)
    
    for i, seq in enumerate(sequences):
        seq_len = seq.size(0)
        padded_batch[i, :seq_len] = seq
        lengths[i] = seq_len
        
    return padded_batch, lengths


class RealWorldDataset(Dataset):
    """
    Real-world Dataset.
    """
    def __init__(
        self, 
        path, 
        split = 'train', 
        num_obs = 1,
        num_action = 20, 
        num_history = 5, 
        voxel_size = 0.005,
        cam_ids = ['750612070851'],
        aug = False,
        aug_trans_min = [-0.2, -0.2, -0.2],
        aug_trans_max = [0.2, 0.2, 0.2],
        aug_rot_min = [-30, -30, -30],
        aug_rot_max = [30, 30, 30],
        aug_jitter = False,
        aug_jitter_params = [0.4, 0.4, 0.2, 0.1],
        aug_jitter_prob = 0.2,
        with_cloud = False,
        vis = False
    ):
        assert split in ['train', 'val', 'all']

        self.path = path
        self.split = split
        self.data_path = os.path.join(path, split)
        self.calib_path = os.path.join(path, "calib")
        self.num_obs = num_obs
        self.num_action = num_action
        self.num_history = num_history
        self.voxel_size = voxel_size
        self.aug = aug
        self.aug_trans_min = np.array(aug_trans_min)
        self.aug_trans_max = np.array(aug_trans_max)
        self.aug_rot_min = np.array(aug_rot_min)
        self.aug_rot_max = np.array(aug_rot_max)
        self.aug_jitter = aug_jitter
        self.aug_jitter_params = np.array(aug_jitter_params)
        self.aug_jitter_prob = aug_jitter_prob
        self.with_cloud = with_cloud
        self.vis = vis
        
        self.all_demos = sorted(os.listdir(self.data_path))
        self.num_demos = len(self.all_demos)

        self.data_paths = []
        self.cam_ids = []
        self.calib_timestamp = []
        self.obs_frame_ids = []
        self.action_frame_ids = []
        self.history_frame_ids = []
        self.track_indices = []
        self.track_cache = {}
        self.projectors = {}
        
        for i in range(self.num_demos):
            demo_path = os.path.join(self.data_path, self.all_demos[i])
            for cam_id in cam_ids:
                # path
                cam_path = os.path.join(demo_path, "cam_{}".format(cam_id))
                if not os.path.exists(cam_path):
                    continue
                # metadata
                with open(os.path.join(demo_path, "metadata.json"), "r") as f:
                    meta = json.load(f)
                # get frame ids
                frame_ids = [
                    int(os.path.splitext(x)[0]) 
                    for x in sorted(os.listdir(os.path.join(cam_path, "color"))) 
                    if int(os.path.splitext(x)[0]) <= meta["finish_time"]
                ]
                # get calib timestamps
                with open(os.path.join(demo_path, "timestamp.txt"), "r") as f:
                    calib_timestamp = f.readline().rstrip()
                # get samples according to num_obs and num_action
                obs_frame_ids_list = []
                action_frame_ids_list = []
                history_frame_ids_list = []
                track_indices_list = []

                for cur_idx in range(len(frame_ids) - 1):
                    obs_pad_before = max(0, num_obs - cur_idx - 1)
                    action_pad_after = max(0, num_action - (len(frame_ids) - 1 - cur_idx))
                    frame_begin = max(0, cur_idx - num_obs + 1)
                    frame_end = min(len(frame_ids), cur_idx + num_action + 1)
                    obs_frame_ids = frame_ids[:1] * obs_pad_before + frame_ids[frame_begin: cur_idx + 1]
                    action_frame_ids = frame_ids[cur_idx + 1: frame_end] + frame_ids[-1:] * action_pad_after
                    
                    history_begin = max(0, cur_idx - self.num_history)
                    actual_history_frames = frame_ids[history_begin:cur_idx + 1]  
                    
                    if len(actual_history_frames) < self.num_history + 1:
                        padding_count = self.num_history + 1 - len(actual_history_frames)
                        history_frame_ids = [frame_ids[0]] * padding_count + actual_history_frames
                    else:
                        history_frame_ids = actual_history_frames
                    
                    assert len(history_frame_ids) == self.num_history + 1, \
                        f"Expected {self.num_history + 1} history frames, got {len(history_frame_ids)}"
                    
                    obs_frame_ids_list.append(obs_frame_ids)
                    action_frame_ids_list.append(action_frame_ids)
                    history_frame_ids_list.append(history_frame_ids)
                    track_indices_list.append(cur_idx)

                self.data_paths += [demo_path] * len(obs_frame_ids_list)
                self.cam_ids += [cam_id] * len(obs_frame_ids_list)
                self.calib_timestamp += [calib_timestamp] * len(obs_frame_ids_list)
                self.obs_frame_ids += obs_frame_ids_list
                self.action_frame_ids += action_frame_ids_list
                self.history_frame_ids += history_frame_ids_list
                self.track_indices += track_indices_list
        
    def __len__(self):
        return len(self.obs_frame_ids)

    def _augmentation(self, clouds, tcps):
        translation_offsets = np.random.rand(3) * (self.aug_trans_max - self.aug_trans_min) + self.aug_trans_min
        rotation_angles = np.random.rand(3) * (self.aug_rot_max - self.aug_rot_min) + self.aug_rot_min
        rotation_angles = rotation_angles / 180 * np.pi  # tranform from degree to radius
        aug_mat = rot_trans_mat(translation_offsets, rotation_angles)
        center = clouds[-1][..., :3].mean(axis = 0)

        for i in range(len(clouds)):
            clouds[i][..., :3] -= center
            clouds[i] = apply_mat_to_pcd(clouds[i], aug_mat)
            clouds[i][..., :3] += center

        tcps[..., :3] -= center
        tcps = apply_mat_to_pose(tcps, aug_mat, rotation_rep = "quaternion")
        tcps[..., :3] += center

        return clouds, tcps

    def _normalize_tcp(self, tcp_list):
        ''' tcp_list: [T, 3(trans) + 6(rot) + 1(width)]'''
        tcp_list[:, :3] = (tcp_list[:, :3] - TRANS_MIN) / (TRANS_MAX - TRANS_MIN) * 2 - 1
        tcp_list[:, -1] = tcp_list[:, -1] / MAX_GRIPPER_WIDTH * 2 - 1
        return tcp_list

    def load_point_cloud(self, colors, depths, cam_id):
        h, w = depths.shape
        fx, fy = INTRINSICS[cam_id][0, 0], INTRINSICS[cam_id][1, 1]
        cx, cy = INTRINSICS[cam_id][0, 2], INTRINSICS[cam_id][1, 2]
        scale = 1000. if 'f' not in cam_id else 4000.
        colors = o3d.geometry.Image(colors.astype(np.uint8))
        depths = o3d.geometry.Image(depths.astype(np.float32))
        camera_intrinsics = o3d.camera.PinholeCameraIntrinsic(
            width = w, height = h, fx = fx, fy = fy, cx = cx, cy = cy
        )
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            colors, depths, scale, convert_rgb_to_intensity = False
        )
        cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, camera_intrinsics)
        cloud = cloud.voxel_down_sample(self.voxel_size)
        points = np.array(cloud.points)
        colors = np.array(cloud.colors)
        return points.astype(np.float32), colors.astype(np.float32)

    def _compute_relative_actions(self, action_tcps):
        if len(action_tcps) <= 1:
            return np.zeros((0, action_tcps.shape[1]) if len(action_tcps) > 0 else (0, 7))
        
        relative_actions = []
        for i in range(1, len(action_tcps)):
            rel_action = np.zeros_like(action_tcps[i])
            if np.array_equal(action_tcps[i], action_tcps[i-1]):
                rel_action[:3] = np.array([0.0, 0.0, 0.0])
            else:
                rel_action[:3] = action_tcps[i, :3] - action_tcps[i-1, :3]
            
            rel_action[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
            relative_actions.append(rel_action)
        
        return np.array(relative_actions) 

    def _compute_relative_gripper_actions(self, action_grippers):
        if len(action_grippers) <= 1:
            return np.zeros(0)
        
        relative_grippers = []
        for i in range(1, len(action_grippers)):
            if action_grippers[i] == action_grippers[i-1]:
                relative_grippers.append(0.0)
            else:
                relative_grippers.append(action_grippers[i] - action_grippers[i-1])
        
        return np.array(relative_grippers)  

    def _normalize_relative_tcp(self, relative_tcp_list):
        relative_tcp_list[:, :3] = np.clip(relative_tcp_list[:, :3], -REL_TRANS_MAX, REL_TRANS_MAX)
        relative_tcp_list[:, :3] = relative_tcp_list[:, :3] / REL_TRANS_MAX
        
        relative_tcp_list[:, -1] = np.clip(relative_tcp_list[:, -1], -REL_GRIPPER_MAX, REL_GRIPPER_MAX)
        relative_tcp_list[:, -1] = relative_tcp_list[:, -1] / REL_GRIPPER_MAX
        
        return relative_tcp_list

    def __getitem__(self, index):
        data_path = self.data_paths[index]
        cam_id = self.cam_ids[index]
        calib_timestamp = self.calib_timestamp[index]
        obs_frame_ids = self.obs_frame_ids[index]
        action_frame_ids = self.action_frame_ids[index]
        history_frame_ids = self.history_frame_ids[index] 
        track_idx = self.track_indices[index]

        # directories
        color_dir = os.path.join(data_path, "cam_{}".format(cam_id), 'color')
        depth_dir = os.path.join(data_path, "cam_{}".format(cam_id), 'depth')
        tcp_dir = os.path.join(data_path, "cam_{}".format(cam_id), 'tcp')
        gripper_dir = os.path.join(data_path, "cam_{}".format(cam_id), 'gripper_command')

        # load camera projector by calib timestamp
        timestamp_path = os.path.join(data_path, 'timestamp.txt')
        with open(timestamp_path, 'r') as f:
            timestamp = f.readline().rstrip()
        if timestamp not in self.projectors:
            # create projector cache
            self.projectors[timestamp] = Projector(os.path.join(self.calib_path, timestamp))
        projector = self.projectors[timestamp]

        # create color jitter
        if self.split == 'train' and self.aug_jitter:
            jitter = T.ColorJitter(
                brightness = self.aug_jitter_params[0],
                contrast = self.aug_jitter_params[1],
                saturation = self.aug_jitter_params[2],
                hue = self.aug_jitter_params[3]
            )
            jitter = T.RandomApply([jitter], p = self.aug_jitter_prob)

        # load colors and depths
        colors_list = []
        depths_list = []
        for frame_id in obs_frame_ids:
            colors = Image.open(os.path.join(color_dir, "{}.png".format(frame_id)))
            if self.split == 'train' and self.aug_jitter:
                colors = jitter(colors)
            colors_list.append(colors)
            depths_list.append(
                np.array(Image.open(os.path.join(depth_dir, "{}.png".format(frame_id))), dtype = np.float32)
            )
        colors_list = np.stack(colors_list, axis = 0)
        depths_list = np.stack(depths_list, axis = 0)

        # point clouds
        clouds = []
        for i, frame_id in enumerate(obs_frame_ids):
            points, colors = self.load_point_cloud(colors_list[i], depths_list[i], cam_id)
            x_mask = ((points[:, 0] >= WORKSPACE_MIN[0]) & (points[:, 0] <= WORKSPACE_MAX[0]))
            y_mask = ((points[:, 1] >= WORKSPACE_MIN[1]) & (points[:, 1] <= WORKSPACE_MAX[1]))
            z_mask = ((points[:, 2] >= WORKSPACE_MIN[2]) & (points[:, 2] <= WORKSPACE_MAX[2]))
            mask = (x_mask & y_mask & z_mask)
            points = points[mask]
            colors = colors[mask]
            # apply imagenet normalization
            colors = (colors - IMG_MEAN) / IMG_STD
            cloud = np.concatenate([points, colors], axis = -1)
            clouds.append(cloud)

        track_path = os.path.join(data_path, "cam_{}".format(cam_id), "gripper_tracks", "normalized_coords_left.npy")
        track_path_right = os.path.join(data_path, "cam_{}".format(cam_id), "gripper_tracks", "normalized_coords_right.npy")
        pred_tracks_path = os.path.join(data_path, "cam_{}".format(cam_id), "tracks", "pred_tracks.npy")

        cache_key = f"{track_path}_{pred_tracks_path}_{cam_id}"

        if cache_key in self.track_cache:
            all_tracks = self.track_cache[cache_key]
        elif os.path.exists(track_path) and os.path.exists(pred_tracks_path):
            left_tracks = np.load(track_path)      # shape: (seq_len, 2)
            right_tracks = np.load(track_path_right)  # shape: (seq_len, 2)
            
            pred_tracks = np.load(pred_tracks_path)  # shape: (1, 379, 118, 2)
            pred_tracks = pred_tracks[0]  # (379, 118, 2)
            
            pred_tracks_normalized = pred_tracks.copy()
            pred_tracks_normalized[:, :, 0] /= 1280.0  
            pred_tracks_normalized[:, :, 1] /= 720.0   
            
            if pred_tracks_normalized.shape[1] >= 4:
                selected_indices = np.random.choice(pred_tracks_normalized.shape[1], 4, replace=False)
                selected_tracks = pred_tracks_normalized[:, selected_indices, :]  
            else:
                selected_indices = np.arange(pred_tracks_normalized.shape[1])
                selected_tracks = pred_tracks_normalized[:, selected_indices, :]
                while selected_tracks.shape[1] < 4:
                    last_point = selected_tracks[:, -1:, :]
                    selected_tracks = np.concatenate([selected_tracks, last_point], axis=1)
                selected_tracks = selected_tracks[:, :4, :] 
            
            min_len = min(len(left_tracks), len(right_tracks), len(selected_tracks))
            left_tracks = left_tracks[:min_len]
            right_tracks = right_tracks[:min_len]  
            selected_tracks = selected_tracks[:min_len]
            
            gripper_tracks = np.stack([left_tracks, right_tracks], axis=1)  # shape: (min_len, 2, 2)
            
            self.track_cache[cache_key] = {
                'gripper_tracks': gripper_tracks,
                'selected_tracks': selected_tracks
            }
            
        # elif os.path.exists(track_path):
        #     left_tracks = np.load(track_path)
        #     right_tracks = np.load(track_path_right)
        #     gripper_tracks = np.stack([left_tracks, right_tracks], axis=1)  # shape: (seq_len, 2, 2)
        #     self.track_cache[cache_key] = {
        #         'gripper_tracks': gripper_tracks,
        #         'selected_tracks': None
        #     }
        else:
            raise FileNotFoundError(f"Track file not found: {track_path} or {pred_tracks_path}")
            all_tracks = None

        if cache_key in self.track_cache:
            track_data = self.track_cache[cache_key]
            gripper_track_slice = track_data['gripper_tracks'][:track_idx + 1] if track_data['gripper_tracks'] is not None else None
            selected_track_slice = track_data['selected_tracks'][:track_idx + 1] if track_data['selected_tracks'] is not None else None
        else:
            raise FileNotFoundError(f"Track file not found: {track_path}")  

        # actions
        action_tcps = []
        action_grippers = []
        for frame_id in action_frame_ids:
            tcp = np.load(os.path.join(tcp_dir, "{}.npy".format(frame_id)))[:7].astype(np.float32)
            projected_tcp = projector.project_tcp_to_camera_coord(tcp, cam_id)
            gripper_width = decode_gripper_width(np.load(os.path.join(gripper_dir, "{}.npy".format(frame_id)))[0])
            action_tcps.append(projected_tcp)
            action_grippers.append(gripper_width)
        action_tcps = np.stack(action_tcps)
        action_grippers = np.stack(action_grippers)

        history_tcps = []
        history_grippers = []
        for frame_id in history_frame_ids:
            tcp = np.load(os.path.join(tcp_dir, "{}.npy".format(frame_id)))[:7].astype(np.float32)
            projected_tcp = projector.project_tcp_to_camera_coord(tcp, cam_id)
            gripper_width = decode_gripper_width(np.load(os.path.join(gripper_dir, "{}.npy".format(frame_id)))[0])
            history_tcps.append(projected_tcp)
            history_grippers.append(gripper_width)
        
        history_tcps = np.stack(history_tcps)  
        history_grippers = np.stack(history_grippers)  
        
        relative_action_tcps = self._compute_relative_actions(history_tcps)  
        relative_action_grippers = self._compute_relative_gripper_actions(history_grippers)  

        # point augmentations
        if self.split == 'train' and self.aug:
            clouds, action_tcps = self._augmentation(clouds, action_tcps)

        # visualization
        if self.vis:
            points = clouds[-1][..., :3]
            print("point range", points.min(axis=0), points.max(axis=0))
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors * IMG_STD + IMG_MEAN)
            traj = []
            # red box stands for the workspace range
            bbox3d_1 = o3d.geometry.AxisAlignedBoundingBox(WORKSPACE_MIN, WORKSPACE_MAX)
            bbox3d_1.color = [1, 0, 0]
            # green box stands for the translation normalization range
            bbox3d_2 = o3d.geometry.AxisAlignedBoundingBox(TRANS_MIN, TRANS_MAX)
            bbox3d_2.color = [0, 1, 0]
            action_tcps_vis = xyz_rot_transform(action_tcps, from_rep = "quaternion", to_rep = "matrix")
            for i in range(len(action_tcps_vis)):
                action = action_tcps_vis[i]
                frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.03).transform(action)
                traj.append(frame)
            o3d.visualization.draw_geometries([pcd.voxel_down_sample(self.voxel_size), bbox3d_1, bbox3d_2, *traj])
        
        # rotation transformation (to 6d)
        action_tcps = xyz_rot_transform(action_tcps, from_rep = "quaternion", to_rep = "rotation_6d")
        relative_action_tcps = xyz_rot_transform(relative_action_tcps, from_rep="quaternion", to_rep="rotation_6d")

        actions = np.concatenate((action_tcps, action_grippers[..., np.newaxis]), axis = -1)
        relative_actions = np.concatenate((relative_action_tcps, relative_action_grippers[..., np.newaxis]), axis=-1)

        assert relative_actions.shape[0] == self.num_history, \
            f"Expected {self.num_history} relative actions, got {relative_actions.shape[0]}"

        # normalization
        actions_normalized = self._normalize_tcp(actions.copy())
        relative_actions_normalized = self._normalize_relative_tcp(relative_actions.copy())

        # make voxel input
        input_coords_list = []
        input_feats_list = []
        for cloud in clouds:
            # Upd Note. Make coords contiguous.
            coords = np.ascontiguousarray(cloud[:, :3] / self.voxel_size, dtype = np.int32)
            # Upd Note. API change.
            input_coords_list.append(coords)
            input_feats_list.append(cloud.astype(np.float32))

        # convert to torch
        actions = torch.from_numpy(actions).float()
        actions_normalized = torch.from_numpy(actions_normalized).float()
        relative_actions = torch.from_numpy(relative_actions).float()
        relative_actions_normalized = torch.from_numpy(relative_actions_normalized).float()
        
        gripper_tracks_tensor = torch.from_numpy(gripper_track_slice).float() if gripper_track_slice is not None else None
        selected_tracks_tensor = torch.from_numpy(selected_track_slice).float() if selected_track_slice is not None else None

        ret_dict = {
            'input_coords_list': input_coords_list,
            'input_feats_list': input_feats_list,
            'action': actions,
            'action_normalized': actions_normalized,
            'relative_action': relative_actions, 
            'relative_action_normalized': relative_actions_normalized,
            'gripper_tracks': gripper_tracks_tensor,    #  shape: (seq_len, 2, 2)
            'selected_tracks': selected_tracks_tensor   #  shape: (seq_len, 4, 2)
        }
        
        if self.with_cloud:  # warning: this may significantly slow down the training process.
            ret_dict["clouds_list"] = clouds

        return ret_dict
        

def collate_fn(batch):
    if type(batch[0]).__module__ == 'numpy':
        return torch.stack([torch.from_numpy(b) for b in batch], 0)
    elif torch.is_tensor(batch[0]):
        return torch.stack(batch, 0)
    elif isinstance(batch[0], container_abcs.Mapping):
        ret_dict = {}
        for key in batch[0]:
            if key in TO_TENSOR_KEYS:
                if key == 'gripper_tracks':
                    gripper_tracks_list = [d[key] for d in batch if d[key] is not None]
                    if len(gripper_tracks_list) > 0:
                        lengths = [gt.size(0) for gt in gripper_tracks_list]
                        if len(set(lengths)) == 1:
                            ret_dict[key] = torch.stack(gripper_tracks_list, 0)
                            ret_dict['gripper_track_lengths'] = None
                        else:
                            padded_batch, track_lengths = create_variable_length_batch(gripper_tracks_list)
                            ret_dict[key] = padded_batch
                            ret_dict['gripper_track_lengths'] = track_lengths
                    else:
                        ret_dict[key] = None
                        ret_dict['gripper_track_lengths'] = None
                elif key == 'selected_tracks':
                    selected_tracks_list = [d[key] for d in batch if d[key] is not None]
                    if len(selected_tracks_list) > 0:
                        lengths = [st.size(0) for st in selected_tracks_list]
                        if len(set(lengths)) == 1:
                            ret_dict[key] = torch.stack(selected_tracks_list, 0)
                            ret_dict['selected_track_lengths'] = None
                        else:
                            padded_batch, track_lengths = create_variable_length_batch(selected_tracks_list)
                            ret_dict[key] = padded_batch
                            ret_dict['selected_track_lengths'] = track_lengths
                    else:
                        ret_dict[key] = None
                        ret_dict['selected_track_lengths'] = None
                else:
                    ret_dict[key] = collate_fn([d[key] for d in batch])
            else:
                ret_dict[key] = [d[key] for d in batch]
        coords_batch = ret_dict['input_coords_list']
        feats_batch = ret_dict['input_feats_list']
        coords_batch, feats_batch = ME.utils.sparse_collate(coords_batch, feats_batch)
        ret_dict['input_coords_list'] = coords_batch
        ret_dict['input_feats_list'] = feats_batch
        return ret_dict
    elif isinstance(batch[0], container_abcs.Sequence):
        return [sample for b in batch for sample in b]
    
    raise TypeError("batch must contain tensors, dicts or lists; found {}".format(type(batch[0])))


def decode_gripper_width(gripper_width):
    return gripper_width / 1000. * 0.095
