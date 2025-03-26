# isaac gym库存在问题，一定要先import pinocchio再import isaacgym
import pinocchio
import os
import pink
from pink.tasks import FrameTask
import matplotlib.pyplot as plt
import yourdfpy

from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch

import math
import numpy as np
from scipy.spatial.transform import Rotation as R
import torch

from TeleVision import OpenTeleVision
from Preprocessor import VuerPreprocessor
from constants_vuer import tip_indices
from dex_retargeting.retargeting_config import RetargetingConfig
from pytransform3d import rotations

from pathlib import Path
import argparse
import time
import yaml
from multiprocessing import Array, Process, shared_memory, Queue, Manager, Event, Semaphore

from teleop.teleop_utils import euler_to_matrix
from teleop_utils import *
import cv2

# IK 参数
dt = 1e-2
stop_thres = 1e-6
max_iterate_step = 400

# Franka 底座位置
base_height = 0.9

right_arm_pose = gymapi.Transform()
right_arm_pose.p = gymapi.Vec3(-1, -0.25, base_height)
right_arm_pose.r = gymapi.Quat(0, 0, 0, 1)

right_base2world = gympose2matrix(right_arm_pose)

camera2base = np.array([-0.03745211617802127, 0.39863880614930414, 0.7757048858682832, \
                        -2.1297708863552454, 0.01951434202694724, -2.3067503645474754])

# urdf file used in pinocchio
pinocchio_urdf = os.path.join(
    os.path.dirname(__file__),
    "..",
    "assets",
    "franka_pinocchio",
    "robots",
    "franka_panda.urdf",
)

franka_pin = pinocchio.RobotWrapper.BuildFromURDF(
    filename=pinocchio_urdf,
    package_dirs=["/opt/ros/noetic/share/"],
    root_joint=None,
)
print(f"URDF description successfully loaded in {franka_pin}")

# 末端姿态连接
eef_link_trans = np.array([0, 0, 0.107])
rpy = np.array([-1.5707963, 0, -0.785398163397])
eef_link_rotation = R.from_euler('xyz', rpy)
# 转换为齐次变换矩阵
eef_link_matrix = trans_rot2matrix(eef_link_trans, eef_link_rotation)

# 末端工具和机械臂的连接，用于pinocchio计算ik
placement = ndarray_to_se3(trans_rot2matrix(eef_link_trans, eef_link_rotation))

# Frame details
joint_name = franka_pin.model.names[-1]
parent_joint = franka_pin.model.getJointId(joint_name)
parent_frame = franka_pin.model.getFrameId("panda_link7")
# placement = pinocchio.SE3.Identity()


FRAME_NAME = "ee_frame"
ee_frame = franka_pin.model.addFrame(
    pinocchio.Frame(
        FRAME_NAME,
        parent_joint,
        parent_frame,
        placement,
        pinocchio.FrameType.OP_FRAME,
    )
)
franka_pin.data = pinocchio.Data(franka_pin.model)
low = franka_pin.model.lowerPositionLimit
high = franka_pin.model.upperPositionLimit
ee_frame_id = franka_pin.model.getFrameId(FRAME_NAME)

def img_normalize(img):
    img_min = np.min(img)
    img_max = np.max(img)

    normalized_depth_image = (img - img_min) / (img_max - img_min) * 255

    # 转换为 uint8 类型
    normalized_depth_image = normalized_depth_image.astype(np.uint8)
    return normalized_depth_image





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
def pink_solve_ik(target_pose, initial_joint_state=None, robot=franka_pin, verbose=False):
    # 若没有定义初始状态 则使用关节中间位置作为初始状态计算
    if initial_joint_state is None:
        robot.q0 = (low + high) / 2
    else:
        robot.q0 = initial_joint_state

    # Task details
    ee_task = FrameTask(FRAME_NAME, [1.0, 1.0, 1.0], [1, 1, 1])

    ee_task.set_target(ndarray_to_se3(target_pose))

    configuration = pink.Configuration(robot.model, robot.data, robot.q0)
    franka_eef_pose = configuration.get_transform_frame_to_world(FRAME_NAME)
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
        franka_eef_pose = configuration.get_transform_frame_to_world(FRAME_NAME)
        nb_steps += 1
        if nb_steps >= max_iterate_step:
            break
    if verbose:
        print(f"Desired precision is error_norm < {stop_thres}")
        print(f"Terminated after {nb_steps} steps with {error_norm = :.2}")
    return configuration.q, franka_eef_pose


class VuerTeleop:
    def __init__(self, config_file_path):
        self.resolution = (720, 1280)
        self.crop_size_w = 0
        self.crop_size_h = 0
        self.resolution_cropped = (self.resolution[0] - self.crop_size_h, self.resolution[1] - 2 * self.crop_size_w)

        self.img_shape = (self.resolution_cropped[0], 2 * self.resolution_cropped[1], 3)
        self.img_height, self.img_width = self.resolution_cropped[:2]

        self.shm = shared_memory.SharedMemory(create=True, size=np.prod(self.img_shape) * np.uint8().itemsize)
        self.img_array = np.ndarray((self.img_shape[0], self.img_shape[1], 3), dtype=np.uint8, buffer=self.shm.buf)
        image_queue = Queue()
        toggle_streaming = Event()
        self.tv = OpenTeleVision(self.resolution_cropped, self.shm.name, image_queue, toggle_streaming, cert_file=None,
                                 key_file=None, ngrok=False)
        self.processor = VuerPreprocessor()

        RetargetingConfig.set_default_urdf_dir('../assets')
        with Path(config_file_path).open('r') as f:
            cfg = yaml.safe_load(f)
        left_retargeting_config = RetargetingConfig.from_dict(cfg['left'])
        right_retargeting_config = RetargetingConfig.from_dict(cfg['right'])
        self.left_retargeting = left_retargeting_config.build()
        self.right_retargeting = right_retargeting_config.build()

    def step(self):
        head_mat, left_wrist_mat, right_wrist_mat, left_hand_mat, right_hand_mat = self.processor.process(self.tv)
        # print(right_hand_mat)
        # import pdb;pdb.set_trace()

        head_rmat = head_mat[:3, :3]

        left_pose = np.concatenate([left_wrist_mat[:3, 3] + np.array([-0.6, 0, 1.6]),
                                    rotations.quaternion_from_matrix(left_wrist_mat[:3, :3])[[1, 2, 3, 0]]])
        right_pose = np.concatenate([right_wrist_mat[:3, 3] + np.array([-0.6, 0, 1.6]),
                                     rotations.quaternion_from_matrix(right_wrist_mat[:3, :3])[[1, 2, 3, 0]]])
        left_qpos = self.left_retargeting.retarget(left_hand_mat[tip_indices])[
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]  # [[4, 5, 6, 7, 10, 11, 8, 9, 0, 1, 2, 3]]
        right_qpos = self.right_retargeting.retarget(right_hand_mat[tip_indices])[
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]  # [[4, 5, 6, 7, 10, 11, 8, 9, 0, 1, 2, 3]]

        return head_rmat, left_pose, right_pose, left_qpos, right_qpos


class Sim:
    def __init__(self,
                 print_freq=False):
        self.print_freq = print_freq

        # initialize gym
        self.gym = gymapi.acquire_gym()

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
        self.left_cam_offset = np.array([0, 0.033, 0])
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

        self.previous_pose = None

    def set_seg_id(self, actor_handler, seg_id):
        # 使用gym.get_actor_rigid_body_count获取刚体数量
        rigid_body_count = self.gym.get_actor_rigid_body_count(self.env, actor_handler)
        print(f"Actor has {rigid_body_count} rigid bodies.")

        # 遍历每个刚体并设置分割ID
        for i in range(rigid_body_count):
            self.gym.set_rigid_body_segmentation_id(self.env, actor_handler, i, seg_id)

    def step(self, head_position, head_rmat, arm_pose, hand_pose):

        if self.print_freq:
            start = time.time()

        # 获取hand相对于franka底座的相对姿态
        relative_pose = get_reletive_hand_pose(arm_pose, right_arm_pose, hand_pose_type="ndarray")

        # 在pinocchio中计算右臂末端link位姿
        arm_states = np.zeros(self.dof, dtype=gymapi.DofState.dtype)
        # if self.previous_pose is None:
        #     current_pose, eef_relative_pose = pink_solve_ik(target_pose=relative_pose)
        # else:
        #     current_pose, eef_relative_pose = pink_solve_ik(target_pose=relative_pose,
        #                                                                 initial_joint_state=self.previous_pose)
        # self.previous_pose = current_pose
        #
        # hand_pose = np.full(12, 0., dtype=np.float64)

        # joint_position = np.concatenate([current_pose, hand_pose])
        joint_position = [0.,  0.,  0.,  -1.507,  0.,  1.507,
         0.,  0.1,          0. ,         0.   ,       0.    ,      0.,
         0.   ,       0.    ,      0.     ,     0.    ,      0.    ,      0.,
         0.]

        print(self.dof)

        arm_states['pos'] = joint_position
        print(arm_states['pos'])
        self.gym.set_actor_dof_states(self.env, self.arm_handle, arm_states, gymapi.STATE_POS)

        self.cam_pos = head_position








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
        left_depth_image = img_normalize(left_depth_image)
        right_depth_image = img_normalize(right_depth_image)

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


if __name__ == '__main__':
    teleoperator = VuerTeleop('inspire_hand_0_4_6.yml')
    simulator = Sim()

    try:
        cnt = 0
        while True:
            # head_rmat, left_pose, right_pose, left_qpos, right_qpos = teleoperator.step()

            cam_pos_abs = relative_pose_to_absolute(right_arm_pose, camera2base)

            head_rmat = euler_to_matrix(cam_pos_abs[3:])

            head_position = cam_pos_abs[:3]
            print(head_position)

            # arm_pose = left_pose
            arm_pose = np.array([-0.9,  0.1,  1.5,  0.5, -0.5,  0.5,  0.5])

            hand_pose = np.full(12, 0.1)
            left_color_image, right_color_image, left_depth_image, right_depth_image, left_mask, right_mask = \
                simulator.step(head_position, head_rmat, arm_pose, hand_pose)

            max_id = np.max(left_mask)
            color_map = set_seg_color_map(max_id)
            left_seg_image = color_map[left_mask]

            max_id = np.max(right_mask)
            color_map = set_seg_color_map(max_id)
            right_seg_image = color_map[right_mask]

            left_mask = np.stack([left_mask, left_mask, left_mask], axis=-1)
            left_franka = cv2.bitwise_and(left_color_image, left_mask*255)


            depth = np.hstack((left_depth_image, right_depth_image))
            color = np.hstack((left_color_image, right_color_image))
            seg = np.hstack((left_seg_image, right_seg_image))

            cnt = cnt + 1
            if cnt == 10:
                cv2.imwrite('./depth.png', depth)
                cv2.imwrite('./color.png', color)
                cv2.imwrite('./seg.png', seg)
                cv2.imwrite('./left_franka.png', left_franka)
                cv2.imwrite('./left_mask.png', left_mask*255)
                # 创建3D坐标轴
                fig = plt.figure(figsize=(15, 15))
                ax = fig.add_subplot(111, projection='3d')

                # 绘制位姿
                plot_pose(camera2base, ax)

                # 显示图形
                plt.show()
                exit(0)

    except KeyboardInterrupt:
        simulator.end()
        exit(0)
