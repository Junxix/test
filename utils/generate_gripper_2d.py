import numpy as np
import cv2
import os

root_train_dir = '/data/jingjing/data/realdata_sampled/realdata_20250618/train/'
calib_root_dir = '/data/jingjing/data/realdata_sampled/realdata_20250618/calib/1749642854745'

def get_gripper_offset(gripper_info_dir, timestamp):
    """
    从gripper_info文件夹中读取对应时间戳的npy文件，计算偏移量
    
    Args:
        gripper_info_dir: gripper_info文件夹路径
        timestamp: 时间戳字符串
    
    Returns:
        offset: 计算得到的偏移量
    """
    try:
        gripper_file = os.path.join(gripper_info_dir, f'{timestamp}.npy')
        if os.path.exists(gripper_file):
            gripper_data = np.load(gripper_file)
            # 第一维/1000 * 0.095 / 2
            offset = gripper_data[0] / 1000 * 0.095 / 2
            return offset
        else:
            print(f"Warning: Gripper info file not found: {gripper_file}")
            return 0.019  # 返回默认值
    except Exception as e:
        print(f"Error reading gripper info for timestamp {timestamp}: {e}")
        return 0.019  # 返回默认值

def process_scene(scene_dir):
    cam_dir = os.path.join(scene_dir, 'cam_750612070851')
    tcp_dir = os.path.join(cam_dir, 'tcp')
    color_dir = os.path.join(cam_dir, 'color')
    gripper_info_dir = os.path.join(cam_dir, 'gripper_info')
    
    png_files = sorted([f for f in os.listdir(color_dir) if f.endswith('.png')], 
                      key=lambda x: int(x.split('.')[0]))
    
    if png_files:
        image_path = os.path.join(color_dir, png_files[0])
    else:
        print(f"No PNG files found in {color_dir}")
        return

    color_base_names = set()
    for f in os.listdir(color_dir):
        if f.endswith('.png'):
            base_name = f.split('.')[0] 
            color_base_names.add(base_name)

    # 收集轨迹点和对应的时间戳
    trajectory_data = []  # 存储 (轨迹点, 时间戳) 的列表
    if os.path.exists(tcp_dir):
        npy_files = sorted([f for f in os.listdir(tcp_dir) if f.endswith('.npy')])
        for file_name in npy_files:
            base_name = file_name.split('.')[0]
            if base_name in color_base_names:
                file_path = os.path.join(tcp_dir, file_name)
                data = np.load(file_path)
                first_three_numbers = data[:3]
                trajectory_data.append((first_three_numbers, base_name))
        
        if not trajectory_data:
            print(f"No matching trajectory data found in {tcp_dir}")
            return
    else:
        print(f"The directory {tcp_dir} does not exist.")
        return

    tcp_file = os.path.join(calib_root_dir, 'tcp.npy')
    extrinsics_file = os.path.join(calib_root_dir, 'extrinsics.npy')
    intrinsics_file = os.path.join(calib_root_dir, 'intrinsics.npy')
    
    tcp = np.load(tcp_file)
    extrinsics = np.load(extrinsics_file, allow_pickle=True).item()
    intrinsics = np.load(intrinsics_file, allow_pickle=True).item()
    
    position = tcp[:3]
    quaternion = tcp[3:]

    def quaternion_to_rotation_matrix(quaternion):
        qw, qx, qy, qz = quaternion
        R = np.array([
            [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
            [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
            [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]
        ])
        return R

    def create_transformation_matrix(position, quaternion):
        R = quaternion_to_rotation_matrix(quaternion)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = position
        return T

    M_end_to_base = create_transformation_matrix(position, quaternion)
    M_cam0433_to_end = np.array([[0, -1, 0, 0],
                                [1, 0, 0, 0.077],
                                [0, 0, 1, 0.2665],
                                [0, 0, 0, 1]])
    M_cam0433_to_A = extrinsics['043322070878'][0]
    M_cam7506_to_A = extrinsics['750612070851'][0]
    M_cam7506_to_base = M_cam7506_to_A @ np.linalg.inv(M_cam0433_to_A) @ M_cam0433_to_end @ M_end_to_base
    
    camera_matrix = intrinsics['750612070851']
    print(f"Camera matrix: {camera_matrix}")
    extrinsic_matrix = M_cam7506_to_base
    
    image_right = cv2.imread(image_path)
    image_left = cv2.imread(image_path)
    
    img_width = 1280
    img_height = 720
    
    def process_gripper_side(trajectory_data, gripper_info_dir, offset_sign, side_name, image, color):
        """
        处理gripper侧的轨迹，每个点使用对应时间戳的动态偏移量
        
        Args:
            trajectory_data: (轨迹点, 时间戳) 的列表
            gripper_info_dir: gripper_info文件夹路径
            offset_sign: 偏移量的符号 (1 for right, -1 for left)
            side_name: 侧面名称
            image: 要绘制的图像
            color: 绘制颜色
        """
        pixel_coordinates = []
        normalized_coordinates = []
        prev_point = None
        offsets_used = []
        
        for point, timestamp in trajectory_data:
            # 为每个时间戳获取对应的gripper偏移量
            if os.path.exists(gripper_info_dir):
                gripper_offset = get_gripper_offset(gripper_info_dir, timestamp)
            else:
                gripper_offset = 0.019  # 默认值
            
            offset = offset_sign * gripper_offset
            offsets_used.append(gripper_offset)
            
            # 复制点并应用偏移量
            point_with_offset = point.copy()
            point_with_offset[1] += offset
            
            object_point_world = np.append(point_with_offset, 1).reshape(-1, 1)
            object_point_camera = extrinsic_matrix @ object_point_world
            object_point_pixel = camera_matrix @ object_point_camera
            object_point_pixel /= object_point_pixel[2]  
            
            pixel_x = object_point_pixel[0, 0]
            pixel_y = object_point_pixel[1, 0]
            
            pixel_coordinates.append([pixel_x, pixel_y])
            
            normalized_x = pixel_x / img_width
            normalized_y = pixel_y / img_height
            normalized_coordinates.append([normalized_x, normalized_y])
            
            if prev_point is not None:
                cv2.line(image, (int(prev_point[0]), int(prev_point[1])), 
                        (int(pixel_x), int(pixel_y)), color, thickness=4)
            prev_point = [pixel_x, pixel_y]
        
        print(f"{side_name} gripper offsets used: min={min(offsets_used):.6f}, max={max(offsets_used):.6f}, mean={np.mean(offsets_used):.6f}")
        
        return np.array(pixel_coordinates), np.array(normalized_coordinates), offsets_used
    
    # 使用动态计算的偏移量处理左右gripper
    pixel_coords_right, norm_coords_right, offsets_right = process_gripper_side(
        trajectory_data, gripper_info_dir, 1, "Right", image_right, (0, 0, 255)
    )
    
    pixel_coords_left, norm_coords_left, offsets_left = process_gripper_side(
        trajectory_data, gripper_info_dir, -1, "Left", image_left, (255, 0, 0)
    )

    print(f"Processing scene: {scene_dir}")
    print(f"Processed {len(trajectory_data)} trajectory points")
    
    os.makedirs('./videos/2d/', exist_ok=True)
    
    gripper_tracks_dir = os.path.join(scene_dir, 'cam_750612070851/gripper_tracks/')
    os.makedirs(gripper_tracks_dir, exist_ok=True)
    
    scene_name = os.path.basename(scene_dir)
    save_path_right = os.path.join('./videos/2d/', f'marked_image_right_{scene_name}.jpg')
    save_path_left = os.path.join('./videos/2d/', f'marked_image_left_{scene_name}.jpg')
    
    cv2.imwrite(save_path_right, image_right)
    cv2.imwrite(save_path_left, image_left)
    print(f"Saved right gripper image to {save_path_right}")
    print(f"Saved left gripper image to {save_path_left}")
    
    # 可选：保存坐标数据和偏移量信息
    pixel_save_path_right = os.path.join(gripper_tracks_dir, 'pixel_coords_right.npy')
    normalized_save_path_right = os.path.join(gripper_tracks_dir, 'normalized_coords_right.npy')
    offsets_save_path_right = os.path.join(gripper_tracks_dir, 'offsets_right.npy')
    np.save(pixel_save_path_right, pixel_coords_right)
    np.save(normalized_save_path_right, norm_coords_right)
    np.save(offsets_save_path_right, np.array(offsets_right))
    print(f"Saved right pixel coordinates to {pixel_save_path_right}")
    print(f"Saved right normalized coordinates to {normalized_save_path_right}")
    print(f"Saved right offsets to {offsets_save_path_right}")
    
    pixel_save_path_left = os.path.join(gripper_tracks_dir, 'pixel_coords_left.npy')
    normalized_save_path_left = os.path.join(gripper_tracks_dir, 'normalized_coords_left.npy')
    offsets_save_path_left = os.path.join(gripper_tracks_dir, 'offsets_left.npy')
    np.save(pixel_save_path_left, pixel_coords_left)
    np.save(normalized_save_path_left, norm_coords_left)
    np.save(offsets_save_path_left, np.array(offsets_left))
    print(f"Saved left pixel coordinates to {pixel_save_path_left}")
    print(f"Saved left normalized coordinates to {normalized_save_path_left}")
    print(f"Saved left offsets to {offsets_save_path_left}")
    
    return {
        'right': {
            'pixel': pixel_coords_right, 
            'normalized': norm_coords_right,
            'offsets': offsets_right
        },
        'left': {
            'pixel': pixel_coords_left, 
            'normalized': norm_coords_left,
            'offsets': offsets_left
        }
    }

# 主处理循环
for task_dir in os.listdir(root_train_dir):
    task_path = os.path.join(root_train_dir, task_dir)
    if os.path.isdir(task_path):
        coordinates_data = process_scene(task_path)
        if coordinates_data:
            print(f"Completed processing for {task_dir}")
            print(f"Right gripper: {len(coordinates_data['right']['offsets'])} points processed")
            print(f"Left gripper: {len(coordinates_data['left']['offsets'])} points processed")
        else:
            print(f"Failed to process {task_dir}")
        print("-" * 50)