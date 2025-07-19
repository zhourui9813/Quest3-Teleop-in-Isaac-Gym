# isaac gym库存在问题，一定要先import pinocchio再import isaacgym
import pinocchio
import os
import pink
from pink.tasks import FrameTask
# import matplotlib.pyplot as plt
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

from teleop_utils import *
import cv2
import yaml
import click
import zarr
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
from human_data.constants import standard2xfylzu
from real.common.replay_buffer import ReplayBuffer


# Franka 底座位置
base_height = 0.9

def generate_sinusoidal_trajectory(num_steps=100, time_duration=10, amplitudes=None, frequencies=None, phases=None):
    """
    生成一个6维的正弦运动位姿序列。

    :param num_steps: 采样步数
    :param time_duration: 运动持续时间（秒）
    :param amplitudes: 每个维度的振幅（列表）
    :param frequencies: 每个维度的频率（列表）
    :param phases: 每个维度的初始相位（列表）
    :return: 形状为 (num_steps, 6) 的numpy数组，表示位姿序列
    """
    if amplitudes is None:
        amplitudes = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]  # 默认振幅
    if frequencies is None:
        frequencies = [0.5, 0.5, 0.5, 0.2, 0.2, 0.2]  # 默认频率 (Hz)
    if phases is None:
        phases = [0, np.pi / 2, np.pi, 0, np.pi / 2, np.pi]  # 默认相位

    t = np.linspace(0, time_duration, num_steps)  # 生成时间序列
    trajectory = np.zeros((num_steps, 6))

    for i in range(6):
        trajectory[:, i] = amplitudes[i] * np.sin(2 * np.pi * frequencies[i] * t + phases[i])

    return t, trajectory

def hand_sinusoidal_trajectory(num_steps=100, time_duration=10, amplitudes=None, frequencies=None, phases=None):


    orientation = [1.50331319, 0.05143192, -0.03968948]


    """
    生成一个6维的正弦运动位姿序列。

    :param num_steps: 采样步数
    :param time_duration: 运动持续时间（秒）
    :param amplitudes: 每个维度的振幅（列表）
    :param frequencies: 每个维度的频率（列表）
    :param phases: 每个维度的初始相位（列表）
    :return: 形状为 (num_steps, 6) 的numpy数组，表示位姿序列
    """
    if amplitudes is None:
        amplitudes = [0.2, 0.2, 0.2, 0.4, 0.4, 0.4]  # 默认振幅
    if frequencies is None:
        frequencies = [0.5, 0.5, 0.5, 0.2, 0.2, 0.2]  # 默认频率 (Hz)
    if phases is None:
        phases = [0, np.pi / 2, np.pi, 0, np.pi / 2, np.pi]  # 默认相位

    t = np.linspace(0, time_duration, num_steps)  # 生成时间序列
    trajectory = np.zeros((num_steps, 6))

    for i in range(6):
        if i==0:
            trajectory[:, i] = amplitudes[i] * np.sin(2 * np.pi * frequencies[i] * t + phases[i]) + 0.5
        elif i==1:
            trajectory[:, i] = amplitudes[i] * np.sin(2 * np.pi * frequencies[i] * t + phases[i])
        elif i==2:
            trajectory[:, i] = amplitudes[i] * np.sin(2 * np.pi * frequencies[i] * t + phases[i]) + 0.4
        else:
            trajectory[:, i] = amplitudes[i] * np.sin(2 * np.pi * frequencies[i] * t + phases[i]) + orientation[i-3]

    return t, trajectory





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


# 输入末端工具的目标姿态以及初始姿态，迭代计算IK
def pink_solve_ik(target_pose,
                  robot,
                  frame_name,
                  stop_thres,
                  dt,
                  max_iterate_step,
                  initial_joint_state=None,
                  verbose=False):
    # 若没有定义初始状态 则使用关节中间位置作为初始状态计算
    low = robot.model.lowerPositionLimit
    high = robot.model.upperPositionLimit
    if initial_joint_state is None:
        robot.q0 = (low + high) / 2
    else:
        robot.q0 = initial_joint_state

    # Task details
    ee_task = FrameTask(frame_name, [1.0, 1.0, 1.0], [1, 1, 1])

    ee_task.set_target(ndarray_to_se3(target_pose))

    configuration = pink.Configuration(robot.model, robot.data, robot.q0)
    franka_eef_pose = configuration.get_transform_frame_to_world(frame_name)
    error_norm = np.linalg.norm(ee_task.compute_error(configuration))



    nb_steps = 0
    while error_norm > stop_thres:
        dv = pink.solve_ik(
            configuration,
            tasks=[ee_task],
            dt=dt,
            damping=1e-8,
            solver="quadprog",
        )
        q_out = pinocchio.integrate(robot.model, configuration.q, dv * dt)
        configuration = pink.Configuration(robot.model, robot.data, q_out)
        pinocchio.updateFramePlacements(robot.model, robot.data)
        error_norm = np.linalg.norm(ee_task.compute_error(configuration))
        franka_eef_pose = configuration.get_transform_frame_to_world(frame_name)
        nb_steps += 1
        if nb_steps >= max_iterate_step:
            break
    if verbose:
        print(f"Desired precision is error_norm < {stop_thres}")
        print(f"Terminated after {nb_steps} steps with {error_norm = :.2}")
    return configuration.q, franka_eef_pose

class Sim:
    def __init__(self,
                 right_arm_pose,
                 robot_pin,
                 ik_dt,
                 ik_thresh,
                 ik_max_iterate_step,
                 print_freq=False,
                 ):
        self.print_freq = print_freq

        # initialize gym
        self.gym = gymapi.acquire_gym()
        self.robot_pin = robot_pin
        self.ik_dt = ik_dt
        self.ik_thresh = ik_thresh
        self.ik_max_iterate_step = ik_max_iterate_step

        # configure sim
        sim_params = gymapi.SimParams()
        sim_params.dt = 1 / 60
        sim_params.substeps = 2
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 4
        sim_params.physx.num_velocity_iterations = 1
        sim_params.physx.max_gpu_contact_pairs = 8388608
        sim_params.physx.contact_offset = 0.002
        sim_params.physx.friction_offset_threshold = 0.001
        sim_params.physx.friction_correlation_distance = 0.0005
        sim_params.physx.rest_offset = 0.0
        sim_params.physx.use_gpu = True
        sim_params.use_gpu_pipeline = False

        self.sim = self.gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
        if self.sim is None:
            print("*** Failed to create sim")
            quit()

        plane_params = gymapi.PlaneParams()
        plane_params.distance = 0.0
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

        # load table asset
        table_asset_options = gymapi.AssetOptions()
        table_asset_options.disable_gravity = True
        table_asset_options.fix_base_link = True
        table_asset = self.gym.create_box(self.sim, 3, 3, 0.1, table_asset_options)

        # load cube asset
        cube_asset_options = gymapi.AssetOptions()
        cube_asset_options.density = 10
        cube_asset = self.gym.create_box(self.sim, 0.05, 0.05, 0.05, cube_asset_options)

        # set up the env grid
        num_envs = 1
        num_per_row = int(math.sqrt(num_envs))
        env_spacing = 1.25

        # env_lower 和 env_upper 表示你要创建的环境在坐标系中的“最小点”和“最大点”
        env_lower = gymapi.Vec3(-env_spacing, 0.0, -env_spacing)
        env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)
        np.random.seed(0)
        # 这里调用 Isaac Gym 的 create_env，在物理仿真 sim 中，创建一个环境
        self.env = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)

        # table
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(0, 0, base_height)
        pose.r = gymapi.Quat(0, 0, 0, 1)
        table_handle = self.gym.create_actor(self.env, table_asset, pose, 'table', 0)
        color = gymapi.Vec3(0.5, 0.5, 1.0)
        self.gym.set_rigid_body_color(self.env, table_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)

        # cube
        pose = gymapi.Transform()  # 在 Isaac Gym 中，gymapi.Transform() 用于构造一个“位姿”对象（3D 平移 + 旋转）
        pose.p = gymapi.Vec3(0, 0, base_height + 0.5)
        pose.r = gymapi.Quat(0, 0, 0, 1)
        # actor是GymAsset的实例。函数create_actor将一个参与者添加到环境中，并返回一个参与者句柄，该句柄可用于以后与该参与者交互
        cube_handle = self.gym.create_actor(self.env, cube_asset, pose, 'cube', 0)
        color = gymapi.Vec3(1, 0.5, 0.5)
        self.gym.set_rigid_body_color(self.env, cube_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)

        # 定义urdf文件路径
        asset_root = "../assets"
        arm_asset_path = "franka_inspire_hand/franka_description/robots/franka_panda_right.urdf"
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.disable_gravity = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS  # 所有关节都使用位置控制模式
        arm_asset = self.gym.load_asset(self.sim, asset_root, arm_asset_path, asset_options)

        self.dof = self.gym.get_asset_dof_count(arm_asset)

        self.arm_handle = self.gym.create_actor(self.env, arm_asset, right_arm_pose, 'right_arm', 1, 1)
        self.gym.set_actor_dof_states(self.env, self.arm_handle, np.zeros(self.dof, gymapi.DofState.dtype),
                                      gymapi.STATE_ALL)

        arm_states = np.zeros(self.dof, dtype=gymapi.DofState.dtype)

        init_joint_position = [0.,  0.,  0.,  -1.507,  0.,  1.507,
         0.,  0.1,          0. ,         0.   ,       0.    ,      0.,
         0.   ,       0.    ,      0.     ,     0.    ,      0.    ,      0.,
         0.]


        arm_states['pos'] = init_joint_position
        self.previous_joint_position = np.array(init_joint_position[:7])
        # print(arm_states['pos'])
        self.gym.set_actor_dof_states(self.env, self.arm_handle, arm_states, gymapi.STATE_POS)


        self.set_seg_id(self.arm_handle, 1)

        # create default viewer
        self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
        if self.viewer is None:
            print("*** Failed to create viewer")
            quit()
        cam_pos = gymapi.Vec3(1, 1, 2)
        cam_target = gymapi.Vec3(0, 0, 1)
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        self.cam_lookat_offset = np.array([0, 0, 1])
        self.left_cam_offset = np.array([0, 0.0, 0])
        self.right_cam_offset = np.array([0, -0.033, 0])
        self.cam_pos = np.array([-0.6, 0, 1.6])

        self.cam_far_plane = 10
        self.cam_near_plane = 0.1

        # create left 1st preson viewer
        self.left_camera_props = gymapi.CameraProperties()
        self.left_camera_props.width = 1280
        self.left_camera_props.height = 720
        self.left_camera_props.near_plane = self.cam_near_plane
        self.left_camera_props.far_plane = self.cam_far_plane
        self.left_camera_handle = self.gym.create_camera_sensor(self.env, self.left_camera_props)
        self.gym.set_camera_location(self.left_camera_handle,
                                     self.env,
                                     gymapi.Vec3(*(self.cam_pos + self.left_cam_offset)),
                                     gymapi.Vec3(*(self.cam_pos + self.left_cam_offset + self.cam_lookat_offset)))

        # create right 1st preson viewer
        self.right_camera_props = gymapi.CameraProperties()
        self.right_camera_props.width = 1280
        self.right_camera_props.height = 720
        self.right_camera_props.near_plane = self.cam_near_plane
        self.right_camera_props.far_plane = self.cam_far_plane
        self.right_camera_handle = self.gym.create_camera_sensor(self.env, self.right_camera_props)
        self.gym.set_camera_location(self.right_camera_handle,
                                     self.env,
                                     gymapi.Vec3(*(self.cam_pos + self.right_cam_offset)),
                                     gymapi.Vec3(*(self.cam_pos + self.right_cam_offset + self.cam_lookat_offset)))



    def set_seg_id(self, actor_handler, seg_id):
        # 使用gym.get_actor_rigid_body_count获取刚体数量
        rigid_body_count = self.gym.get_actor_rigid_body_count(self.env, actor_handler)
        print(f"Actor has {rigid_body_count} rigid bodies.")

        # 遍历每个刚体并设置分割ID
        for i in range(rigid_body_count):
            self.gym.set_rigid_body_segmentation_id(self.env, actor_handler, i, seg_id)

    def depth_img_normalize(self, img):
        max_depth = -self.cam_far_plane
        min_depth = -self.cam_near_plane

        img[abs(img) > abs(max_depth)] = max_depth
        img[abs(img) < abs(min_depth)] = min_depth
        normalized_depth_image = (min_depth - img) / (min_depth - max_depth) * 255

        # 转换为 uint8 类型
        normalized_depth_image = normalized_depth_image.astype(np.uint8)
        return normalized_depth_image

    def step(self, head_pose, arm_pose_mat, hand_pose):

        if self.print_freq:
            start = time.time()

        head_rmat = R.from_rotvec(head_pose[3:]).as_matrix()
        self.cam_pos = head_pose[:3]

        # 获取hand相对于franka底座的相对姿态
        # relative_pose = get_reletive_hand_pose(arm_pose, right_arm_pose, hand_pose_type="ndarray")

        # 在pinocchio中计算右臂末端link位姿
        arm_states = np.zeros(self.dof, dtype=gymapi.DofState.dtype)
        if self.previous_joint_position is None:
            current_joint_position, eef_relative_pose = pink_solve_ik(target_pose=arm_pose_mat,
                                                                      robot=self.robot_pin,
                                                                      frame_name="hand_base_link",
                                                                      dt=self.ik_dt,
                                                                      stop_thres=self.ik_thresh,
                                                                      max_iterate_step=self.ik_max_iterate_step,
                                                                      initial_joint_state=None
                                                                      )
        else:
            current_joint_position, eef_relative_pose = pink_solve_ik(target_pose=arm_pose_mat,
                                                                      robot=self.robot_pin,
                                                                      frame_name="hand_base_link",
                                                                      dt=self.ik_dt,
                                                                      stop_thres=self.ik_thresh,
                                                                      max_iterate_step=self.ik_max_iterate_step,
                                                                      initial_joint_state=self.previous_joint_position[:7]
                                                                      )


        self.previous_joint_position = current_joint_position

        joint_position = np.concatenate([current_joint_position, hand_pose])

        arm_states['pos'] = joint_position
        # print(arm_states['pos'])
        self.gym.set_actor_dof_states(self.env, self.arm_handle, arm_states, gymapi.STATE_POS)

        # step the physics
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        self.gym.step_graphics(self.sim)
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)

        curr_lookat_offset = self.cam_lookat_offset @ head_rmat.T
        curr_left_offset = self.left_cam_offset @ head_rmat.T
        curr_right_offset = self.right_cam_offset @ head_rmat.T

        self.gym.set_camera_location(self.left_camera_handle,
                                     self.env,
                                     gymapi.Vec3(*(self.cam_pos + curr_left_offset)),
                                     gymapi.Vec3(*(self.cam_pos + curr_left_offset + curr_lookat_offset)))
        self.gym.set_camera_location(self.right_camera_handle,
                                     self.env,
                                     gymapi.Vec3(*(self.cam_pos + curr_right_offset)),
                                     gymapi.Vec3(*(self.cam_pos + curr_right_offset + curr_lookat_offset)))
        left_color_image = self.gym.get_camera_image(self.sim, self.env, self.left_camera_handle, gymapi.IMAGE_COLOR)
        right_color_image = self.gym.get_camera_image(self.sim, self.env, self.right_camera_handle, gymapi.IMAGE_COLOR)
        # RGBA convert to RGB
        left_color_image = left_color_image.reshape(left_color_image.shape[0], -1, 4)[..., :3]
        right_color_image = right_color_image.reshape(right_color_image.shape[0], -1, 4)[..., :3]

        left_depth_image = self.gym.get_camera_image(self.sim, self.env, self.left_camera_handle, gymapi.IMAGE_DEPTH)
        right_depth_image = self.gym.get_camera_image(self.sim, self.env, self.right_camera_handle, gymapi.IMAGE_DEPTH)
        left_depth_image = self.depth_img_normalize(left_depth_image)
        right_depth_image = self.depth_img_normalize(right_depth_image)

        left_mask = self.gym.get_camera_image(self.sim, self.env, self.left_camera_handle,
                                                   gymapi.IMAGE_SEGMENTATION)
        right_mask = self.gym.get_camera_image(self.sim, self.env, self.right_camera_handle,
                                                    gymapi.IMAGE_SEGMENTATION)


        self.gym.draw_viewer(self.viewer, self.sim, True)
        self.gym.sync_frame_time(self.sim)

        if self.print_freq:
            end = time.time()
            print('Frequency:', 1 / (end - start))

        return left_color_image, right_color_image, left_depth_image, right_depth_image, left_mask.astype(np.uint8), right_mask.astype(np.uint8)

    def end(self):
        self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)

@click.command()
@click.option('--robot_config', '-rc', default="../real/config/franka_inspire_atv_cam_unimanual.yaml", required=True, help='Path to robot_config')
@click.option('--human_data_zarr', '-hdz', default="../data/test_data/data_human_processed/data_human_raw_org_1280x720_640x480.zarr", required=True, help='Path to human data')
@click.option('--episode', '-e', default=0, type=int, required=True, help='episode number')
@click.option('--isaac_output', '-o', default="./robot_img", required=True, help='Path to isaac gym image')
@click.option('--video_img_folder', '-vi', default="./video_img", required=True, help='Path to original video image')
@click.option('--robot_ik_urdf', '-ik_u', default="../assets/franka_pinocchio/robots/franka_panda.urdf", required=True, help='Path to pinocchio urdf')
@click.option('--ik_dt', default=1e-2, type=float, required=True)
@click.option('--ik_thresh', default=1e-6, type=float, required=True)
@click.option('--ik_max_iterate_step', default=400, type=int, required=True)
def main(robot_config,
         human_data_zarr,
         episode,
         isaac_output,
         video_img_folder,
         robot_ik_urdf,
         ik_dt,
         ik_thresh,
         ik_max_iterate_step
         ):

    right_arm_pose = gymapi.Transform()
    right_arm_pose.p = gymapi.Vec3(-1, -0.25, base_height)
    right_arm_pose.r = gymapi.Quat(0, 0, 0, 1)

    right_arm_base_abs = gympose2matrix(right_arm_pose)

    # urdf file used in pinocchi
    franka_pin = pinocchio.RobotWrapper.BuildFromURDF(
        filename=robot_ik_urdf,
        package_dirs=["/opt/ros/noetic/share/"],
        root_joint=None,
    )
    print(f"URDF description successfully loaded in {franka_pin}")
    franka_pin.data = pinocchio.Data(franka_pin.model)

    franka_mask_path = os.path.join(isaac_output, "franka_mask")
    franka_seg_path = os.path.join(isaac_output, "franka_seg")
    franka_depth_path = os.path.join(isaac_output, "franka_depth")
    franka_color_path = os.path.join(isaac_output, "franka_color")

    if not os.path.exists(video_img_folder):
        os.makedirs(video_img_folder)

    if not os.path.exists(isaac_output):
        os.makedirs(isaac_output)
    if not os.path.exists(franka_seg_path):
        os.makedirs(franka_seg_path)
    if not os.path.exists(franka_depth_path):
        os.makedirs(franka_depth_path)
    if not os.path.exists(franka_color_path):
        os.makedirs(franka_color_path)
    if not os.path.exists(franka_mask_path):
        os.makedirs(franka_mask_path)

    replay_buffer = ReplayBuffer.create_from_path(
        zarr_path=human_data_zarr,
        mode='r')
    episode = replay_buffer.get_episode(episode)
    camera_motion_pose = episode['camera0_pose']
    human_eef_motion_trans = episode['action_robot0_eef_pos']
    human_eef_motion_rot = episode['action_robot0_eef_rot_axis_angle']
    human_eef_motion = np.concatenate([human_eef_motion_trans, human_eef_motion_rot], axis=1)
    human_hand_pose = episode['urdf_gripper0_gripper_pose']
    human_video = episode['camera0_rgb']
    T, ih, iw, ic = human_video.shape
    for i in tqdm(range(T), desc="Saving frames", ncols=100):
        frame = cv2.cvtColor(human_video[i], cv2.COLOR_BGR2RGB)
        frame_name = f"{i:05d}.jpg"
        video_img_save = os.path.join(video_img_folder, frame_name)
        cv2.imwrite(video_img_save, frame)

    with open(robot_config, 'r') as f:
        config = yaml.safe_load(f)

    cameras = config.get("cameras", [])
    first_camera = cameras[0]  # 取第一个相机的calib_cam_to_base
    calib_pose = first_camera.get("calib_cam_to_base")
    camera2base = calib_pose[0][1]  # 取六维姿态数据
    print("Camera to base calibration pose:", camera2base)

    simulator = Sim(right_arm_pose=right_arm_pose,
                    robot_pin=franka_pin,
                    ik_dt=ik_dt,
                    ik_thresh=ik_thresh,
                    ik_max_iterate_step=ik_max_iterate_step)

    camera2base = np.array(calib_pose[0][1])  # 取六维姿态数据
    camera2base[0, 3] = 0.5

    try:
        for i in range(len(camera_motion_pose)):
        # while True:
        #     i=0
            cam_calib_abs_mat = right_arm_base_abs @ camera2base
            cam_motion_pose_abs_mat = cam_calib_abs_mat @ pose_to_mat(camera_motion_pose[i])
            hand_motion_pose_abs_mat = camera2base @ pose_to_mat(human_eef_motion[i])

            cam_motion_pose_abs = mat_to_pose(cam_motion_pose_abs_mat)
            hand_motion_pose_abs = hand_motion_pose_abs_mat @ standard2xfylzu
            # right_arm_pose.p(-1, -0.25, base_height)

            hand_pose = human_hand_pose[i]

            left_color_image, right_color_image, left_depth_image, right_depth_image, left_mask, right_mask = \
                simulator.step(cam_motion_pose_abs,
                               hand_motion_pose_abs,
                               hand_pose
                               )
            isaac_img_h, isaac_img_w, _ = left_color_image.shape
            crop_w = (isaac_img_w - iw) // 2
            crop_h = (isaac_img_h - ih) // 2

            max_id = np.max(left_mask)
            color_map = set_seg_color_map(max_id)
            left_seg_image = color_map[left_mask]

            max_id = np.max(right_mask)
            color_map = set_seg_color_map(max_id)
            right_seg_image = color_map[right_mask]

            left_mask = np.stack([left_mask, left_mask, left_mask], axis=-1) * 255
            left_mask = left_mask[crop_h:crop_h+ih, crop_w:crop_w+iw]
            left_franka = cv2.bitwise_and(left_color_image[crop_h:crop_h+ih, crop_w:crop_w+iw], left_mask)

            right_mask = np.stack([right_mask, right_mask, right_mask], axis=-1) * 255
            right_mask = right_mask[crop_h:crop_h + ih, crop_w:crop_w + iw]
            right_franka = cv2.bitwise_and(right_color_image[crop_h:crop_h+ih, crop_w:crop_w+iw], right_mask)

            franka_seg_left_save = os.path.join(franka_seg_path, f"franka_seg_left_{i:05d}.jpg")
            franka_seg_right_save = os.path.join(franka_seg_path, f"franka_seg_right_{i:05d}.jpg")
            cv2.imwrite(franka_seg_left_save, left_franka)
            cv2.imwrite(franka_seg_right_save, right_franka)

            franka_mask_left_save = os.path.join(franka_mask_path, f"franka_mask_left_{i:05d}.jpg")
            franka_mask_right_save = os.path.join(franka_mask_path, f"franka_mask_right_{i:05d}.jpg")
            cv2.imwrite(franka_mask_left_save, left_mask)
            cv2.imwrite(franka_mask_right_save, right_mask)

            depth = np.hstack((left_depth_image, right_depth_image))
            color = np.hstack((left_color_image, right_color_image))
            seg = np.hstack((left_seg_image, right_seg_image))

            # franka_color_left_save = os.path.join(franka_color_path, f"franka_color_left_{i:05d}.jpg")
            # franka_color_right_save = os.path.join(franka_color_path, f"franka_color_right_{i:05d}.jpg")
            # franka_depth_left_save = os.path.join(franka_depth_path, f"franka_depth_left_{i:05d}.jpg")
            # franka_depth_right_save = os.path.join(franka_depth_path, f"franka_depth_right_{i:05d}.jpg")
            # franka_seg_left_save = os.path.join(franka_seg_path, f"franka_seg_left_{i:05d}.jpg")
            # franka_seg_right_save = os.path.join(franka_seg_path, f"franka_seg_right_{i:05d}.jpg")

            # cv2.imwrite(franka_color_left_save, left_color_image)
            # cv2.imwrite(franka_color_right_save, right_color_image)
            #
            # cv2.imwrite(franka_depth_right_save, left_depth_image)
            # cv2.imwrite(franka_depth_left_save, right_depth_image)
            #
            # cv2.imwrite(franka_seg_left_save, left_seg_image)
            # cv2.imwrite(franka_seg_right_save, right_seg_image)

            color = color[crop_h:crop_h+ih, crop_w:crop_w+iw]
            seg = np.hstack((left_franka, right_franka))
            cv2.imshow("seg_franka", seg)
            cv2.imshow("color", color)
            cv2.waitKey(1)



    except KeyboardInterrupt:
        simulator.end()
        exit(0)

if __name__ == '__main__':
    main()