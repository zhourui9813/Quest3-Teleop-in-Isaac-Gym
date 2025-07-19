# isaac gym库存在问题，一定要先import pinocchio再import isaacgym
import pinocchio
import os
import pink
from pink.tasks import FrameTask
from trimesh.path.packing import visualize

# import matplotlib.pyplot as plt
from teleop.pygame_display import PygameDisplay
import yourdfpy
from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
import math
import numpy as np
from scipy.spatial.transform import Rotation as R
import torch

import time
from tqdm import tqdm

from teleop.teleop_utils import *
import cv2
import yaml
import click
import zarr
import os
import sys
from multiprocessing import Array, Process, shared_memory, Queue, Manager, Event, Semaphore
from teleop.TeleVision import OpenTeleVision
from teleop.Preprocessor import VuerPreprocessor
from dex_retargeting.retargeting_config import RetargetingConfig
from pathlib import Path
from pytransform3d import rotations
from teleop.constants_vuer import tip_indices
from datetime import datetime
from teleop.vuer_teleop import VuerTeleop
from teleop.simulation import left_arm_base, right_arm_base, Sim
from teleop.gesture_recognizer import GestureRecognizer, Pinch

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)

def draw_image_caption(color_img_left, color_img_right, text, position,
                       fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.0, lineType=cv2.LINE_AA, thickness=2, color=(0,0,0)):
    # 确保图像是可写的numpy数组且数据类型正确
    if color_img_left is not None:
        # 转换为连续内存布局的numpy数组
        img_left = np.ascontiguousarray(color_img_left.astype(np.uint8))
        cv2.putText(img_left, text, position,
                    fontFace=fontFace,
                    fontScale=fontScale,
                    lineType=lineType,
                    thickness=thickness,
                    color=color)
        # 将修改后的数据复制回原图像
        np.copyto(color_img_left, img_left)

    if color_img_right is not None:
        # 转换为连续内存布局的numpy数组
        img_right = np.ascontiguousarray(color_img_right.astype(np.uint8))
        cv2.putText(img_right, text, position,
                    fontFace=fontFace,
                    fontScale=fontScale,
                    lineType=lineType,
                    thickness=thickness,
                    color=color)
        # 将修改后的数据复制回原图像
        np.copyto(color_img_right, img_right)

def start_episode_recording(session_output_dir, episode_num):
    """开始新的episode录制"""
    episode_dir = os.path.join(session_output_dir, f"episode{episode_num}")
    
    # 创建episode目录
    os.makedirs(episode_dir, exist_ok=True)
    
    # 创建子目录
    rgb_dir = os.path.join(episode_dir, "RGB")
    depth_dir = os.path.join(episode_dir, "Depth")
    segment_dir = os.path.join(episode_dir, "Segment")
    
    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)
    os.makedirs(segment_dir, exist_ok=True)
    
    # 创建视频写入器
    writers = {
        'left_rgb': cv2.VideoWriter(os.path.join(rgb_dir, "left_rgb.mp4"), 
                                   cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (1280, 720)),
        'right_rgb': cv2.VideoWriter(os.path.join(rgb_dir, "right_rgb.mp4"), 
                                    cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (1280, 720)),
        'left_depth': cv2.VideoWriter(os.path.join(depth_dir, "left_depth.mp4"), 
                                     cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (1280, 720)),
        'right_depth': cv2.VideoWriter(os.path.join(depth_dir, "right_depth.mp4"), 
                                      cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (1280, 720)),
        'left_seg': cv2.VideoWriter(os.path.join(segment_dir, "left_segment.mp4"), 
                                   cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (1280, 720)),
        'right_seg': cv2.VideoWriter(os.path.join(segment_dir, "right_segment.mp4"), 
                                    cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (1280, 720))
    }
    
    print(f"Started recording episode {episode_num} in {episode_dir}")
    return writers, episode_dir

def stop_episode_recording(writers, episode_dir):
    """停止当前episode录制"""
    if writers:
        for writer in writers.values():
            writer.release()
        print(f"Stopped recording episode in {episode_dir}")
    return None, None

def set_seg_color_map(max_idx):
    seg_color_list = []
    # 为每个分割 ID 设置对应颜色
    for i in range(max_idx + 1):
        if i == 1:
            color = [0, 0, 255]  # 设置 ID 1 的颜色
        elif i == 2:
            color = [98, 193, 66]  # 设置 ID 2 的颜色
        elif i == 3:
            color = [0, 255, 0]  # 设置 ID 3 的颜色
        elif i == 4:
            color = [0, 255, 255]  # 设置 ID 4 的颜色
        else:
            color = [0, 0, 0]  # 设置其他 ID 的默认颜色，这里假设其他 ID 映射到黑色

        seg_color_list.append(color)
    return np.array(seg_color_list)

# Default display mode: "rgb", "depth", "mask"
@click.command()
@click.option('--isaac_output', '-o', default="output", required=True, help='Path to isaac gym image')
@click.option('--robot_ik_urdf', '-ik_u', default="assets/franka_pinocchio/robots/franka_panda.urdf", required=True, help='Path to pinocchio urdf')
@click.option('--assets_path', '-a_p', default="assets/", required=True, help='Path to assets')
@click.option('--init_mode', '-i_m', default="rgb", required=True, help='Initial visualization mode')
@click.option('--vis_camera', '-v_c', default=False, required=True, help='Visualization on screen')
@click.option('--vis_issac', '-v_i', default=False, required=True, help='Visualization on screen')
@click.option('--ik_dt', default=1e-2, type=float, required=True)
@click.option('--ik_thresh', default=1e-2, type=float, required=True)
@click.option('--ik_max_iterate_step', default=100, type=int, required=True)
def main(isaac_output,
         robot_ik_urdf,
         assets_path,
         init_mode,
         vis_camera,
         vis_issac,
         ik_dt,
         ik_thresh,
         ik_max_iterate_step
         ):

    teleoperator = VuerTeleop('config/inspire_hand_0_4_6.yml', assets_path)

    # urdf file used in pinocchi
    franka_pin = pinocchio.RobotWrapper.BuildFromURDF(
        filename=robot_ik_urdf,
        package_dirs=["/opt/ros/noetic/share/"],
        root_joint=None,
    )
    print(f"URDF description successfully loaded in {franka_pin}")
    franka_pin.data = pinocchio.Data(franka_pin.model)

    # Create time-based output directory structure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_output_dir = os.path.join(isaac_output, timestamp)
    
    # Create main session directory
    if not os.path.exists(session_output_dir):
        os.makedirs(session_output_dir)

    seg_color_map = set_seg_color_map(4)

    simulator = Sim(assets_path=assets_path,
                    left_arm_pose=left_arm_base,
                    right_arm_pose=right_arm_base,
                    robot_pin=franka_pin,
                    ik_dt=ik_dt,
                    ik_thresh=ik_thresh,
                    ik_max_iterate_step=ik_max_iterate_step)

    gr = GestureRecognizer()
    
    # Initialize display mode state
    display_mode = init_mode
    
    # Initialize pygame display (if enabled)
    pygame_display = None
    if vis_camera:
        pygame_display = PygameDisplay(width=2560, height=720, fps=30)
        pygame_display.initialize()
    
    # Initialize recording state
    is_recording = False
    episode_counter = 1
    current_episode_writers = None
    current_episode_dir = None

    try:
        while True:
            head_rmat, head_pos, left_pose, right_pose, left_qpos, right_qpos, ori_left_hand_mat, ori_right_hand_mat = teleoperator.step()

            left_color_image, right_color_image, left_depth_image, right_depth_image, left_mask, right_mask = \
                    simulator.step(head_rmat=head_rmat,
                                   head_pose=head_pos,
                                   left_arm_pose=left_pose,
                                   right_arm_pose=right_pose,
                                   left_hand_pose=left_qpos,
                                   right_hand_pose=right_qpos,
                                   issac_viewer=vis_issac)
            
            # Convert segmentation masks to colored images
            left_seg_colored = seg_color_map[left_mask].astype(np.uint8)
            right_seg_colored = seg_color_map[right_mask].astype(np.uint8)
            
            # Recording control using left hand gestures
            if gr.is_pinch(ori_left_hand_mat, Pinch.THUMB_INDEX) and not is_recording:
                # Start recording
                current_episode_writers, current_episode_dir = start_episode_recording(session_output_dir, episode_counter)
                is_recording = True
                print(f"Recording started for episode {episode_counter}")
                
            elif gr.is_pinch(ori_left_hand_mat, Pinch.THUMB_MIDDLE) and is_recording:
                # Stop recording
                current_episode_writers, current_episode_dir = stop_episode_recording(current_episode_writers, current_episode_dir)
                is_recording = False
                episode_counter += 1
                print(f"Recording stopped. Next episode will be {episode_counter}")
            
            # Save videos only if recording
            if is_recording and current_episode_writers:
                # Save RGB videos
                current_episode_writers['left_rgb'].write(cv2.cvtColor(left_color_image, cv2.COLOR_RGB2BGR))
                current_episode_writers['right_rgb'].write(cv2.cvtColor(right_color_image, cv2.COLOR_RGB2BGR))
                
                # Save depth videos (colored)
                current_episode_writers['left_depth'].write(cv2.cvtColor(left_depth_image, cv2.COLOR_RGB2BGR))
                current_episode_writers['right_depth'].write(cv2.cvtColor(right_depth_image, cv2.COLOR_RGB2BGR))
                
                # Save segmentation videos
                current_episode_writers['left_seg'].write(cv2.cvtColor(left_seg_colored, cv2.COLOR_RGB2BGR))
                current_episode_writers['right_seg'].write(cv2.cvtColor(right_seg_colored, cv2.COLOR_RGB2BGR))

            # Check for gesture commands to switch display mode
            if gr.is_pinch(ori_right_hand_mat, Pinch.THUMB_INDEX):
                display_mode = "depth"
            elif gr.is_pinch(ori_right_hand_mat, Pinch.THUMB_MIDDLE):
                display_mode = "mask"
            elif gr.is_pinch(ori_right_hand_mat, Pinch.THUMB_RING):
                display_mode = "rgb"
            
            # Display according to current mode
            recording_status = "RECORDING" if is_recording else "NOT RECORDING"
            visualize_text = f"Visualization Mode: {display_mode}"
            recording_text = f"Recording Status: {recording_status}"
            episode_text = f"Episode: {episode_counter}"
            
            if display_mode == "depth":
                draw_image_caption(left_depth_image, right_depth_image, visualize_text,
                                   (300, 40), color=(255, 255, 0))
                draw_image_caption(left_depth_image, right_depth_image, recording_text,
                                   (300, 80), color=(0, 191, 255))
                draw_image_caption(left_depth_image, right_depth_image, episode_text,
                                   (300, 120), color=(84, 255, 159))
                np.copyto(teleoperator.img_array, np.hstack((left_depth_image, right_depth_image)))
                if pygame_display:
                    pygame_display.display_image(left_depth_image, right_depth_image, f"Depth Mode - {display_mode}")
            elif display_mode == "mask":
                draw_image_caption(left_seg_colored, right_seg_colored, recording_text,
                                   (300, 40), color=(255, 255, 0))
                draw_image_caption(left_seg_colored, right_seg_colored, visualize_text,
                                   (300, 80), color=(0, 191, 255))
                draw_image_caption(left_seg_colored, right_seg_colored, episode_text,
                                  (300, 120), color=(84, 255, 159))
                np.copyto(teleoperator.img_array, np.hstack((left_seg_colored, right_seg_colored)))
                if pygame_display:
                    pygame_display.display_image(left_seg_colored, right_seg_colored, f"Mask Mode - {display_mode}")
            else:  # rgb mode
                draw_image_caption(left_color_image, right_color_image, recording_text,
                                   (300, 40), color=(255, 255, 0))
                draw_image_caption(left_color_image, right_color_image, visualize_text,
                                   (300, 80), color=(0, 191, 255))
                draw_image_caption(left_color_image, right_color_image, episode_text,
                                  (300, 120), color=(84, 255, 159))
                np.copyto(teleoperator.img_array, np.hstack((left_color_image, right_color_image)))
                if pygame_display:
                    pygame_display.display_image(left_color_image, right_color_image, f"RGB Mode - {display_mode}")
    except KeyboardInterrupt:
        # Stop current episode recording if active
        if is_recording and current_episode_writers:
            current_episode_writers, current_episode_dir = stop_episode_recording(current_episode_writers, current_episode_dir)
        
        # 清理pygame资源
        if pygame_display:
            pygame_display.cleanup()
        
        simulator.end()
        exit(0)

if __name__ == '__main__':
    main()