# from isaacgym import gymapi
# from isaacgym import gymutil
# from isaacgym import gymtorch
#
# import math
# import numpy as np
# import torch
#
# from TeleVision import OpenTeleVision
# from Preprocessor import VuerPreprocessor
# from constants_vuer import tip_indices
# from dex_retargeting.retargeting_config import RetargetingConfig
# from pytransform3d import rotations
#
# from pathlib import Path
# import argparse
# import time
# import yaml
# from multiprocessing import Array, Process, shared_memory, Queue, Manager, Event, Semaphore

# isaac gym库存在问题，一定要先import pinocchio再import isaacgym
import pinocchio
import os
import pink
from pink.tasks import FrameTask
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
from teleop_utils import *


# from inspire_hand_utils.inspire_hand_agent import InspireHandAgent

class VuerTeleop:
    def __init__(self, config_file_path):

        self.resolution = (720, 1280)
        self.crop_size_w = 0
        self.crop_size_h = 0
        self.resolution_cropped = (self.resolution[0]-self.crop_size_h, self.resolution[1]-2*self.crop_size_w)

        self.img_shape = (self.resolution_cropped[0], 2 * self.resolution_cropped[1], 3)
        self.img_height, self.img_width = self.resolution_cropped[:2]

        self.shm = shared_memory.SharedMemory(create=True, size=np.prod(self.img_shape) * np.uint8().itemsize)
        self.img_array = np.ndarray((self.img_shape[0], self.img_shape[1], 3), dtype=np.uint8, buffer=self.shm.buf)
        image_queue = Queue()
        toggle_streaming = Event()
        # Quest
        # self.tv = OpenTeleVision(self.resolution_cropped, self.shm.name, image_queue, toggle_streaming, cert_file=None, key_file=None,  ngrok=False)
        # Vision Pro
        self.tv = OpenTeleVision(self.resolution_cropped, self.shm.name, image_queue, toggle_streaming, cert_file="./cert/cert.pem", key_file="./cert/key.pem", ngrok=False)
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

        head_rmat = head_mat[:3, :3]

        left_pose = np.concatenate([left_wrist_mat[:3, 3] + np.array([-0.6, 0, 1.6]),
                                    rotations.quaternion_from_matrix(left_wrist_mat[:3, :3])[[1, 2, 3, 0]]])
        right_pose = np.concatenate([right_wrist_mat[:3, 3] + np.array([-0.6, 0, 1.6]),
                                     rotations.quaternion_from_matrix(right_wrist_mat[:3, :3])[[1, 2, 3, 0]]])
        left_qpos = self.left_retargeting.retarget(left_hand_mat[tip_indices])[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]
        right_qpos = self.right_retargeting.retarget(right_hand_mat[tip_indices])[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]

        return head_rmat, left_pose, right_pose, left_qpos, right_qpos

class Sim:
    def __init__(self,
                 print_freq=False):
        self.print_freq = print_freq

        # initialize gym
        self.gym = gymapi.acquire_gym()


        # self.hand_agent = InspireHandAgent(port='/dev/ttyUSB0',
        #                                 hand_id=1,
        #                                 baudrate=115200
        #                                 )

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
        table_asset = self.gym.create_box(self.sim, 0.8, 0.8, 0.1, table_asset_options)

        # load cube asset
        cube_asset_options = gymapi.AssetOptions()
        cube_asset_options.density = 10
        cube_asset = self.gym.create_box(self.sim, 0.05, 0.05, 0.05, cube_asset_options)

        # 定义urdf文件路径
        asset_root = "../assets"
        left_asset_path = "inspire_hand/inspire_hand_left.urdf"
        right_asset_path = "inspire_hand/inspire_hand_right.urdf"
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True # 这个 base link 不会受到物理模拟中的力或重力影响
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS # 所有关节都使用位置控制模式

        # 左右手urdf导入
        left_asset = self.gym.load_asset(self.sim, asset_root, left_asset_path, asset_options)
        right_asset = self.gym.load_asset(self.sim, asset_root, right_asset_path, asset_options)
        # 获取左手自由度（和右手一样）
        self.dof = self.gym.get_asset_dof_count(left_asset)

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
        pose.p = gymapi.Vec3(0, 0, 1.2)
        pose.r = gymapi.Quat(0, 0, 0, 1)
        table_handle = self.gym.create_actor(self.env, table_asset, pose, 'table', 0)
        color = gymapi.Vec3(0.5, 0.5, 1.0)
        self.gym.set_rigid_body_color(self.env, table_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)

        # cube
        pose = gymapi.Transform() # 在 Isaac Gym 中，gymapi.Transform() 用于构造一个“位姿”对象（3D 平移 + 旋转）
        pose.p = gymapi.Vec3(0, 0, 1.25)
        pose.r = gymapi.Quat(0, 0, 0, 1)
        # actor是GymAsset的实例。函数create_actor将一个参与者添加到环境中，并返回一个参与者句柄，该句柄可用于以后与该参与者交互
        cube_handle = self.gym.create_actor(self.env, cube_asset, pose, 'cube', 0)
        color = gymapi.Vec3(1, 0.5, 0.5)
        self.gym.set_rigid_body_color(self.env, cube_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)

        # left_hand
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(-0.6, 0, 1.6)
        pose.r = gymapi.Quat(0, 0, 0, 1)
        self.left_handle = self.gym.create_actor(self.env, left_asset, pose, 'left', 1, 1)
        self.gym.set_actor_dof_states(self.env, self.left_handle, np.zeros(self.dof, gymapi.DofState.dtype),
                                      gymapi.STATE_ALL)
        left_idx = self.gym.get_actor_index(self.env, self.left_handle, gymapi.DOMAIN_SIM)

        # right_hand
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(-0.6, 0, 1.6)
        pose.r = gymapi.Quat(0, 0, 0, 1)

        # 这里创建了一个大小为 self.dof 的 NumPy 数组，类型是 gymapi.DofState.dtype（即含 pos, vel 等字段）。初始化为全零后，表示所有关节的 pos、vel 等字段为零。
        # gymapi.STATE_ALL 表示同时更新“关节位置 + 关节速度”。
        self.right_handle = self.gym.create_actor(self.env, right_asset, pose, 'right', 1, 1)

        # 返回该 Actor 在 整个位姿/根状态张量（root state tensor）中的下标/索引。
        # 在后续需要更新根状态或读取 Actor 的根状态时，往往会用到这个索引在大张量中找到对应行。
        self.gym.set_actor_dof_states(self.env, self.right_handle, np.zeros(self.dof, gymapi.DofState.dtype),
                                      gymapi.STATE_ALL)
        right_idx = self.gym.get_actor_index(self.env, self.right_handle, gymapi.DOMAIN_SIM)

        self.root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim) # 获取所有actor root state
        self.gym.refresh_actor_root_state_tensor(self.sim) # 对root state进行更新
        self.root_states = gymtorch.wrap_tensor(self.root_state_tensor) # 将root state包装成tensor 的数据格式

        # 通过left_idx和right_idx来访问root state中的每个手的状态
        self.left_root_states = self.root_states[left_idx]
        self.right_root_states = self.root_states[right_idx]

        # create default viewer
        self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
        if self.viewer is None:
            print("*** Failed to create viewer")
            quit()
        cam_pos = gymapi.Vec3(1, 1, 2)
        cam_target = gymapi.Vec3(0, 0, 1)
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        self.cam_lookat_offset = np.array([1, 0, 0])
        self.left_cam_offset = np.array([0, 0, 0])
        self.right_cam_offset = np.array([0, 0, 0])
        # self.left_cam_offset = np.array([0, 0.0, 0])
        # self.right_cam_offset = np.array([0, 0.0, 0])
        self.cam_pos = np.array([-0.6, 0, 1.6])

        # create left 1st preson viewer
        camera_props = gymapi.CameraProperties()
        camera_props.width = 1280
        camera_props.height = 720
        self.left_camera_handle = self.gym.create_camera_sensor(self.env, camera_props)
        self.gym.set_camera_location(self.left_camera_handle,
                                     self.env,
                                     gymapi.Vec3(*(self.cam_pos + self.left_cam_offset)),
                                     gymapi.Vec3(*(self.cam_pos + self.left_cam_offset + self.cam_lookat_offset)))

        # create right 1st preson viewer
        camera_props = gymapi.CameraProperties()
        camera_props.width = 1280
        camera_props.height = 720
        self.right_camera_handle = self.gym.create_camera_sensor(self.env, camera_props)
        self.gym.set_camera_location(self.right_camera_handle,
                                     self.env,
                                     gymapi.Vec3(*(self.cam_pos + self.right_cam_offset)),
                                     gymapi.Vec3(*(self.cam_pos + self.right_cam_offset + self.cam_lookat_offset)))

    def step(self, head_rmat, left_pose, right_pose, left_qpos, right_qpos):
        # left_pose, right_pose代表手根部的姿态, 为[x, y, z, qx, qy, qz, qw]形式
        # left_qpos, right_qpos代表手部所有关节的姿态
        # print(f"right_qpos is {right_qpos}")
        hand_pose = [right_qpos[4], right_pose[6], right_qpos[2], right_qpos[0], right_qpos[9], right_qpos[8]]
        # print(f"hand_pose is {hand_pose}")

        # self.hand_agent.set_joint_position(np.array(hand_pose))

        if self.print_freq:
            start = time.time()

        print(f"left_pose is {left_pose}")
        print(f"right_pose is {right_pose}")

        # 将Vuer中的手部root state传入方针中并转换为torch类型，方便后续使用set_actor_root_state_tensor进行操作
        self.left_root_states[0:7] = torch.tensor(left_pose, dtype=float)

        self.right_root_states[0:7] = torch.tensor(right_pose, dtype=float)

        # 修改双手的root state,让Gym知道“刚才对 root_states 这块内存的修改要生效了”，从而完成更新左右手在仿真场景中的实际位姿
        # 在 Isaac Gym 中，“root state” 指的是每个 actor（物理对象）的“根节点”或“基座”所处的状态，通常包括位置，旋转，速度等
        # 在这里相当于在修改两个手的空间位置和方向
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

        # 定义存储左手状态的变量
        # gymapi.DofState 是一个用来描述单个 DOF（自由度）状态的结构体（structure），包含“pos”和“vel”
        # 这段代码用于修改每个手的的关节位置
        left_states = np.zeros(self.dof, dtype=gymapi.DofState.dtype)
        left_states['pos'] = left_qpos
        self.gym.set_actor_dof_states(self.env, self.left_handle, left_states, gymapi.STATE_POS)

        right_states = np.zeros(self.dof, dtype=gymapi.DofState.dtype)
        right_states['pos'] = right_qpos
        self.gym.set_actor_dof_states(self.env, self.right_handle, right_states, gymapi.STATE_POS)

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
        left_image = self.gym.get_camera_image(self.sim, self.env, self.left_camera_handle, gymapi.IMAGE_COLOR)
        right_image = self.gym.get_camera_image(self.sim, self.env, self.right_camera_handle, gymapi.IMAGE_COLOR)
        left_image = left_image.reshape(left_image.shape[0], -1, 4)[..., :3]
        right_image = right_image.reshape(right_image.shape[0], -1, 4)[..., :3]

        self.gym.draw_viewer(self.viewer, self.sim, True)
        self.gym.sync_frame_time(self.sim)

        if self.print_freq:
            end = time.time()
            print('Frequency:', 1 / (end - start))

        return left_image, right_image

    def end(self):
        self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)


if __name__ == '__main__':
    teleoperator = VuerTeleop('inspire_hand_0_4_6.yml')
    simulator = Sim()

    try:
        while True:
            head_rmat, left_pose, right_pose, left_qpos, right_qpos = teleoperator.step()
            left_img, right_img = simulator.step(head_rmat, left_pose, right_pose, left_qpos, right_qpos)
            np.copyto(teleoperator.img_array, np.hstack((left_img, right_img)))
    except KeyboardInterrupt:
        simulator.end()
        exit(0)
