import os
import sys
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import functools
from PIL import Image
from tqdm import tqdm
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Optional

sys.path.append('/home/jingjing/workspace/sam2')  
from sam2.build_sam import build_sam2_video_predictor

from cotracker.utils.visualizer import Visualizer, read_video_from_path
from cotracker.predictor import CoTrackerPredictor

DEFAULT_DEVICE = (
    "cuda" if torch.cuda.is_available() else 
    "mps" if torch.backends.mps.is_available() else "cpu"
)

class SAM2CoTrackerIntegration:
    def __init__(self, sam2_checkpoint, sam2_config, cotracker_checkpoint=None):
        self.device = DEFAULT_DEVICE
        self.sam2_checkpoint = sam2_checkpoint
        self.sam2_config = sam2_config
        self.cotracker_checkpoint = cotracker_checkpoint
        
        self._init_sam2()
        
        self._init_cotracker()
        
        self.points = []
        self.labels = []
        self.is_accepting_clicks = True
        self.mask = None
        
        self.temp_dirs = [] 
        
    def _init_sam2(self):
        print("正在加载SAM2模型...")
        if self.device == "cuda":
            torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
        
        self.sam2_predictor = build_sam2_video_predictor(
            self.sam2_config, 
            self.sam2_checkpoint, 
            device=self.device
        )
        
    def _init_cotracker(self):
        print("正在加载CoTracker模型...")
        
        if self.cotracker_checkpoint is not None:
            window_len = 16  
            self.cotracker_model = CoTrackerPredictor(
                checkpoint=self.cotracker_checkpoint,
                offline=False,
                window_len=window_len,
            )
        else:
            self.cotracker_model = torch.hub.load("/home/jingjing/workspace/su1/co-tracker/", "cotracker3_offline", source='local')

        
        self.cotracker_model = self.cotracker_model.to(self.device)
        
    def prepare_video_data(self, video_path):
        if os.path.isfile(video_path):
            self.video = read_video_from_path(video_path)
            self.video = torch.from_numpy(self.video).permute(0, 3, 1, 2)[None].float()
        elif os.path.isdir(video_path):
            self.video = self._load_image_sequence(video_path)
        else:
            raise ValueError(f"无效路径: {video_path}")
            
        self.video = self.video.to(self.device)
        
        self._prepare_sam2_data(video_path)
        
    def _load_image_sequence(self, image_dir):
        image_files = sorted([
            f for f in os.listdir(image_dir) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ], key=lambda x: int(os.path.splitext(x)[0]) if os.path.splitext(x)[0].isdigit() else x)
        
        images = []
        for img_file in tqdm(image_files, desc="加载图像"):
            img_path = os.path.join(image_dir, img_file)
            img = np.array(Image.open(img_path))
            if len(img.shape) == 3 and img.shape[2] == 3:  
                images.append(img)
        
        video_array = np.stack(images, axis=0)  # [T, H, W, C]
        video_tensor = torch.from_numpy(video_array).permute(0, 3, 1, 2)[None].float()  # [1, T, C, H, W]
        
        return video_tensor
        
    def _prepare_sam2_data(self, video_path):
        if os.path.isfile(video_path):
            temp_dir = "./temp_frames"
            os.makedirs(temp_dir, exist_ok=True)
            self.temp_dirs.append(temp_dir)  
            self._extract_frames_from_video(video_path, temp_dir)
            self.sam2_video_dir = temp_dir
        else:
            self.sam2_video_dir = self._prepare_image_directory(video_path)
            
        self.frame_names = sorted([
            f for f in os.listdir(self.sam2_video_dir) 
            if f.lower().endswith(('.jpg', '.jpeg'))
        ], key=lambda x: int(os.path.splitext(x)[0]) if os.path.splitext(x)[0].isdigit() else x)
        
        if not self.frame_names:
            raise RuntimeError(f"在{self.sam2_video_dir}中未找到JPG文件")
        
        print(f"在{self.sam2_video_dir}中找到{len(self.frame_names)}帧")
        
        self.inference_state = self.sam2_predictor.init_state(video_path=self.sam2_video_dir)
        
    def reset_sam2_state(self):
        """重置SAM2状态，用于开始新的标注"""
        self.sam2_predictor.reset_state(self.inference_state)
        self.points = []
        self.labels = []
        self.mask = None
        
    def _prepare_image_directory(self, image_dir):
        jpg_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg'))]
        png_files = [f for f in os.listdir(image_dir) if f.lower().endswith('.png')]
        
        if jpg_files:
            print(f"找到{len(jpg_files)}个JPG文件，直接使用...")
            return image_dir
        elif png_files:
            print(f"找到{len(png_files)}个PNG文件，正在转换为JPG...")
            return self._convert_png_to_jpg(image_dir, png_files)
        else:
            raise ValueError(f"{image_dir}不包含任何JPG或PNG文件")
    
    def _convert_png_to_jpg(self, source_dir, png_files):
        temp_jpg_dir = "./temp_jpg_frames"
        os.makedirs(temp_jpg_dir, exist_ok=True)
        self.temp_dirs.append(temp_jpg_dir) 
        
        for file in os.listdir(temp_jpg_dir):
            file_path = os.path.join(temp_jpg_dir, file)
            if os.path.isfile(file_path):
                os.unlink(file_path)
        
        def get_numeric_sort_key(filename):
            name = os.path.splitext(filename)[0]
            try:
                return int(name)
            except ValueError:
                return name
        
        png_files.sort(key=get_numeric_sort_key)
        
        success_count = 0
        png_files = png_files[:1]
        
        for png_file in tqdm(png_files, desc="PNG转JPG"):
            try:
                png_path = os.path.join(source_dir, png_file)
                
                base_name = os.path.splitext(png_file)[0]
                jpg_file = f"{base_name}.jpg"
                jpg_path = os.path.join(temp_jpg_dir, jpg_file)
                
                with Image.open(png_path) as img:
                    if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
                        background = Image.new('RGB', img.size, (255, 255, 255))
                        if img.mode == 'P':
                            img = img.convert('RGBA')
                        background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
                        img = background
                    elif img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    img.save(jpg_path, 'JPEG', quality=95)
                    success_count += 1
                    
            except Exception as e:
                print(f"转换{png_file}时出错: {e}")
                continue
        
        print(f"成功转换: {success_count}/{len(png_files)}个PNG文件")
        
        if success_count == 0:
            raise RuntimeError("没有PNG文件成功转换为JPG")
        
        converted_files = [f for f in os.listdir(temp_jpg_dir) if f.lower().endswith('.jpg')]
        
        return temp_jpg_dir
        
    def _extract_frames_from_video(self, video_path, output_dir):
        cap = cv2.VideoCapture(video_path)
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_path = os.path.join(output_dir, f"{frame_idx:06d}.jpg")
            cv2.imwrite(frame_path, frame)
            frame_idx += 1
            
        cap.release()
        
    def interactive_annotation(self, ann_frame_idx=0, target_id=1):
        
        print(f"\n=== 标注目标 {target_id} ===")
        print("左键点击添加正点，右键点击添加负点")
        print("按 'z' 删除最后一个点")
        print("关闭窗口完成当前目标的标注")
        
        # 重置点击状态
        self.points = []
        self.labels = []
        self.is_accepting_clicks = True
        
        plt.figure(figsize=(12, 8))
        plt.title(f"目标 {target_id} - 帧 {ann_frame_idx} 标注")
        
        frame_path = os.path.join(self.sam2_video_dir, self.frame_names[ann_frame_idx])
        frame_image = Image.open(frame_path)
        plt.imshow(frame_image)
        
        ann_obj_id = target_id
        onclick_callback = functools.partial(
            self._on_click, 
            ann_frame_idx=ann_frame_idx, 
            ann_obj_id=ann_obj_id
        )
        onkey_callback = functools.partial(
            self._on_key, 
            ann_frame_idx=ann_frame_idx, 
            ann_obj_id=ann_obj_id
        )
        
        canvas = plt.gcf().canvas
        canvas.mpl_connect('button_press_event', onclick_callback)
        canvas.mpl_connect('key_press_event', onkey_callback)
        
        plt.show()
        
        return self.mask
        
    def _on_click(self, event, ann_frame_idx, ann_obj_id):
        if not self.is_accepting_clicks or event.xdata is None or event.ydata is None:
            return
            
        x, y = int(event.xdata), int(event.ydata)
        self.is_accepting_clicks = False
        
        if event.button in [1, 3]:
            label = 1 if event.button == 1 else 0
            self.points.append([x, y])
            self.labels.append(label)
            print(f"添加点 ({x}, {y}) 标签: {label}")
            
            self._update_sam2_prediction(ann_frame_idx, ann_obj_id)
            
        self.is_accepting_clicks = True
        
    def _on_key(self, event, ann_frame_idx, ann_obj_id):
        if event.key == 'z' and len(self.points) > 0:
            removed_point = self.points.pop()
            removed_label = self.labels.pop()
            print(f"删除点: {removed_point} 标签: {removed_label}")
            
            if len(self.points) > 0:
                self._update_sam2_prediction(ann_frame_idx, ann_obj_id)
            else:
                self._clear_display()
                
    def _update_sam2_prediction(self, ann_frame_idx, ann_obj_id):
        if len(self.points) == 0:
            return
            
        _, out_obj_ids, out_mask_logits = self.sam2_predictor.add_new_points_or_box(
            inference_state=self.inference_state,
            frame_idx=ann_frame_idx,
            obj_id=ann_obj_id,
            points=np.array(self.points),
            labels=np.array(self.labels)
        )
        
        self._clear_display()
        
        for i, out_obj_id in enumerate(out_obj_ids):
            mask = (out_mask_logits[i] > 0.0).cpu().numpy()
            self.mask = mask  
            self._show_points(np.array(self.points), np.array(self.labels))
            self._show_mask(mask, obj_id=out_obj_id)
            
    def _clear_display(self):
        ax = plt.gca()
        images = ax.images
        if len(images) > 1:
            for img in images[1:]:
                img.remove()
        for collection in ax.collections:
            collection.remove()
        plt.draw()
        
    def _show_points(self, coords, labels, marker_size=200):
        ax = plt.gca()
        pos_points = coords[labels == 1]
        neg_points = coords[labels == 0]
        
        if len(pos_points) > 0:
            ax.scatter(pos_points[:, 0], pos_points[:, 1], 
                      color='green', marker='*', s=marker_size, 
                      edgecolor='white', linewidth=1.25)
        if len(neg_points) > 0:
            ax.scatter(neg_points[:, 0], neg_points[:, 1], 
                      color='red', marker='*', s=marker_size, 
                      edgecolor='white', linewidth=1.25)
                      
    def _show_mask(self, mask, obj_id=None):
        ax = plt.gca()
        
        if obj_id is not None:
            color = np.array([*plt.get_cmap("tab10")(obj_id)[:3], 0.6])
        else:
            color = np.array([1, 0, 0, 0.6])  
            
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image, alpha=0.6)
        plt.draw()
        
    def track_with_cotracker(self, grid_size=10, grid_query_frame=0, 
                           backward_tracking=False, use_mask=True):

        segm_mask = None
        if use_mask and self.mask is not None:
            if len(self.mask.shape) == 3:
                mask_2d = self.mask[0] 
            else:
                mask_2d = self.mask
                
            segm_mask = torch.from_numpy(mask_2d.astype(np.uint8))[None, None]
            segm_mask = segm_mask.to(self.device)
        
        if use_mask and segm_mask is not None:
            pred_tracks, pred_visibility = self.cotracker_model(
                self.video,
                grid_size=grid_size,
                grid_query_frame=grid_query_frame,
                backward_tracking=backward_tracking,
                segm_mask=segm_mask
            )
        else:
            pred_tracks, pred_visibility = self.cotracker_model(
                self.video,
                grid_size=grid_size,
                grid_query_frame=grid_query_frame,
                backward_tracking=backward_tracking
            )
            
        return pred_tracks, pred_visibility
        
    def visualize_tracking_results(self, pred_tracks, pred_visibility, 
                                 output_dir="./tracking_results", 
                                 grid_query_frame=0, backward_tracking=False):
        
        os.makedirs(output_dir, exist_ok=True)
        
        vis = Visualizer(save_dir=output_dir, pad_value=120, linewidth=3)
        vis.visualize(
            self.video,
            pred_tracks,
            pred_visibility,
            query_frame=0 if backward_tracking else grid_query_frame,
        )
        
    def save_mask(self, output_path):
        if self.mask is not None:
            if len(self.mask.shape) == 3:
                mask_to_save = self.mask[0]
            else:
                mask_to_save = self.mask
                
            mask_uint8 = (mask_to_save * 255).astype(np.uint8)
            Image.fromarray(mask_uint8).save(output_path)
            print(f"掩码已保存到: {output_path}")
        else:
            print("没有掩码可保存")
    
    def cleanup_temp_files(self):
        import shutil
        for temp_dir in self.temp_dirs:
            if os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir)
                    print(f"已清理临时目录: {temp_dir}")
                except Exception as e:
                    print(f"清理临时目录{temp_dir}时出错: {e}")
        self.temp_dirs.clear()
    
    def __del__(self):
        try:
            self.cleanup_temp_files()
        except:
            pass  


def main():
    """主函数 - 支持多目标追踪"""
    
    # 获取用户输入：要处理几个目标
    while True:
        try:
            num_targets = int(input("请输入要追踪的目标数量: "))
            if num_targets > 0:
                break
            else:
                print("请输入大于0的数字")
        except ValueError:
            print("请输入有效的数字")
    
    print(f"\n将进行 {num_targets} 个目标的标注和追踪")
    
    # 配置路径
    sam2_checkpoint = "/data/jingjing/pretrained-models/sam2/checkpoints/sam2.1_hiera_large.pt"
    sam2_config = "//data/jingjing/pretrained-models/sam2/configs/sam2.1/sam2.1_hiera_l.yaml"
    
    base_path = "/data/jingjing/data/realdata_sampled/realdata_sampled_20250620/train/task_0011_user_0555_scene_0005_cfg_0001/cam_750612070851"
    video_path = os.path.join(base_path, "color")
    
    # 创建集成对象
    integrator = SAM2CoTrackerIntegration(
        sam2_checkpoint=sam2_checkpoint,
        sam2_config=sam2_config,
        cotracker_checkpoint=None 
    )
    
    # 准备视频数据（只需要一次）
    print("正在准备视频数据...")
    integrator.prepare_video_data(video_path)
    
    # 创建保存目录
    save_path = os.path.join(base_path, "tracks")
    Path(save_path).mkdir(parents=True, exist_ok=True)
    
    # 循环处理每个目标
    for target_idx in range(num_targets):
        target_id = target_idx + 1
        
        print(f"\n{'='*50}")
        print(f"开始处理目标 {target_id}/{num_targets}")
        print(f"{'='*50}")
        
        # 重置SAM2状态（为新目标准备）
        integrator.reset_sam2_state()
        
        # 交互式标注
        print(f"开始目标 {target_id} 的交互式标注...")
        mask = integrator.interactive_annotation(ann_frame_idx=0, target_id=target_id)
        
        if mask is not None:
            # 保存当前目标的掩码
            mask_path = f"./generated_mask_target_{target_id}.png"
            integrator.save_mask(mask_path)
            
            print(f"开始目标 {target_id} 的CoTracker追踪...")
            pred_tracks, pred_visibility = integrator.track_with_cotracker(
                grid_size=50, 
                grid_query_frame=0,
                backward_tracking=False,
                use_mask=True
            )

            # 转换为numpy格式
            tracks_data = pred_tracks.detach().cpu().numpy() if isinstance(pred_tracks, torch.Tensor) else pred_tracks
            visibility_data = pred_visibility.detach().cpu().numpy() if isinstance(pred_visibility, torch.Tensor) else pred_visibility

            # 保存当前目标的追踪结果
            tracks_filename = f"pred_tracks_target_{target_id}.npy"
            visibility_filename = f"pred_visibility_target_{target_id}.npy"
            
            np.save(os.path.join(save_path, tracks_filename), tracks_data)
            np.save(os.path.join(save_path, visibility_filename), visibility_data)

            print(f"目标 {target_id} 的追踪数据已保存到: {save_path}")
            print(f"  - {tracks_filename}")
            print(f"  - {visibility_filename}")

            # 可视化当前目标的追踪结果
            # vis_output_dir = f"./tracking_results_target_{target_id}"
            vis_output_dir = os.path.join(base_path, f"tracking_results_target_{target_id}")
            integrator.visualize_tracking_results(
                pred_tracks, 
                pred_visibility,
                output_dir=vis_output_dir
            )
            
            print(f"目标 {target_id} 的可视化结果保存到: {vis_output_dir}")
            
        else:
            print(f"目标 {target_id} 标注失败，跳过...")
            continue
    
    print(f"\n{'='*50}")
    print(f"所有 {num_targets} 个目标处理完成！")
    print(f"追踪数据保存在: {save_path}")
    print(f"可视化结果保存在各自的 tracking_results_target_X 文件夹中")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()