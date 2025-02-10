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


# IK 参数
dt = 1e-2
stop_thres = 1e-6
max_iterate_step = 400

# Franka 底座位置
base_height = 0.9
left_arm_pose = gymapi.Transform()
left_arm_pose.p = gymapi.Vec3(-1, 0.25, base_height)
left_arm_pose.r = gymapi.Quat(0, 0, 1, 0)

# 根据机械臂初始位姿获得机械臂坐标系相对于世界坐标系的变换矩阵
left_base2world = gympose2matrix(left_arm_pose)

right_arm_pose = gymapi.Transform()
right_arm_pose.p = gymapi.Vec3(-1, -0.25, base_height)
right_arm_pose.r = gymapi.Quat(0, 0, 1, 0)

right_base2world = gympose2matrix(right_arm_pose)

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
        package_dirs=["."],
        root_joint=None,
    )
print(f"URDF description successfully loaded in {franka_pin}")



#末端姿态连接
eef_link_trans=np.array([0, 0, 0.107])
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

# 使用yourdfpy可视化pinocchio中的urdf
# viz = yourdfpy.URDF.load(pinocchio_urdf)
# viz.show()


# 输入末端工具的目标姿态以及初始姿态，迭代计算IK
def pink_solve_ik(target_pose, initial_joint_state = None, robot = franka_pin ):

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

    print(f"Desired precision is error_norm < {stop_thres}")

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
        if nb_steps >=max_iterate_step:
            break
    print(f"Terminated after {nb_steps} steps with {error_norm = :.2}")
    return configuration.q, franka_eef_pose


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
        self.tv = OpenTeleVision(self.resolution_cropped, self.shm.name, image_queue, toggle_streaming, cert_file=None, key_file=None,  ngrok=False)
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
        left_qpos = self.left_retargeting.retarget(left_hand_mat[tip_indices])[[4, 5, 6, 7, 10, 11, 8, 9, 0, 1, 2, 3]]
        right_qpos = self.right_retargeting.retarget(right_hand_mat[tip_indices])[[4, 5, 6, 7, 10, 11, 8, 9, 0, 1, 2, 3]]

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
        pose.p = gymapi.Vec3(0, 0, base_height+0.5)
        pose.r = gymapi.Quat(0, 0, 0, 1)
        # actor是GymAsset的实例。函数create_actor将一个参与者添加到环境中，并返回一个参与者句柄，该句柄可用于以后与该参与者交互
        cube_handle = self.gym.create_actor(self.env, cube_asset, pose, 'cube', 0)
        color = gymapi.Vec3(1, 0.5, 0.5)
        self.gym.set_rigid_body_color(self.env, cube_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)

        # 定义urdf文件路径
        asset_root = "../assets"
        left_asset_path = "inspire_hand/inspire_hand_left.urdf"
        right_asset_path = "inspire_hand/inspire_hand_right.urdf"
        asset_options = gymapi.AssetOptions()
        asset_options.disable_gravity = True
        # asset_options.flip_visual_attachments = True
        asset_options.fix_base_link = True # 这个 base link 不会受到物理模拟中的力或重力影响
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS # 所有关节都使用位置控制模式

        # 左右手urdf导入
        left_asset = self.gym.load_asset(self.sim, asset_root, left_asset_path, asset_options)
        right_asset = self.gym.load_asset(self.sim, asset_root, right_asset_path, asset_options)
        # 获取左手自由度（和右手一样）
        self.dof = self.gym.get_asset_dof_count(left_asset)

        arm_asset_path = "franka_gym/robots/franka_panda.urdf"
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.disable_gravity = True
        asset_options.flip_visual_attachments = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS  # 所有关节都使用位置控制模式
        left_arm_asset = self.gym.load_asset(self.sim, asset_root, arm_asset_path, asset_options)
        right_arm_asset = self.gym.load_asset(self.sim, asset_root, arm_asset_path, asset_options)


        # left_arm
        self.left_arm_handle = self.gym.create_actor(self.env, left_arm_asset, left_arm_pose, 'left_arm', 1, 1)
        self.gym.set_actor_dof_states(self.env, self.left_arm_handle, np.zeros(self.dof, gymapi.DofState.dtype),
                                      gymapi.STATE_ALL)
        left_arm_idx = self.gym.get_actor_index(self.env, self.left_arm_handle, gymapi.DOMAIN_SIM)

        self.right_arm_handle = self.gym.create_actor(self.env, right_arm_asset, right_arm_pose, 'right_arm', 1, 1)
        self.gym.set_actor_dof_states(self.env, self.right_arm_handle, np.zeros(self.dof, gymapi.DofState.dtype),
                                      gymapi.STATE_ALL)
        right_arm_idx = self.gym.get_actor_index(self.env, self.right_arm_handle, gymapi.DOMAIN_SIM)

        hand_initial_rotation_euler = R.from_euler('xyz', [np.deg2rad(0), np.deg2rad(-90), np.deg2rad(90)])
        hand_initial_rotation_quat = hand_initial_rotation_euler.as_quat()

        # left_hand
        left_hand_pose = gymapi.Transform()
        left_hand_pose.p = gymapi.Vec3(-0.3, -0.5, 1.0)
        # left_hand_pose.r = gymapi.Quat(0, 0, 0, 1)
        left_hand_pose.r = gymapi.Quat(hand_initial_rotation_quat[0],
                                       hand_initial_rotation_quat[1],
                                       hand_initial_rotation_quat[2],
                                       hand_initial_rotation_quat[3] )
        self.left_handle = self.gym.create_actor(self.env, left_asset, left_hand_pose, 'left', 1, 1)
        self.gym.set_actor_dof_states(self.env, self.left_handle, np.zeros(self.dof, gymapi.DofState.dtype),
                                      gymapi.STATE_ALL)
        left_idx = self.gym.get_actor_index(self.env, self.left_handle, gymapi.DOMAIN_SIM)

        # right_hand
        right_hand_pose = gymapi.Transform()
        right_hand_pose.p = gymapi.Vec3(-0.3, 0.5, 1.0)
        right_hand_pose.r = gymapi.Quat(hand_initial_rotation_quat[0],
                                       hand_initial_rotation_quat[1],
                                       hand_initial_rotation_quat[2],
                                       hand_initial_rotation_quat[3] )
        self.right_handle = self.gym.create_actor(self.env, right_asset, right_hand_pose, 'right', 1, 1)
        self.gym.set_actor_dof_states(self.env, self.right_handle, np.zeros(self.dof, gymapi.DofState.dtype),
                                      gymapi.STATE_ALL)
        right_idx = self.gym.get_actor_index(self.env, self.right_handle, gymapi.DOMAIN_SIM)

        left_reletive_pose = get_reletive_hand_pose(left_hand_pose, left_arm_pose)
        right_reletive_pose = get_reletive_hand_pose(right_hand_pose, right_arm_pose)

        left_arm_states = np.zeros(7, dtype=gymapi.DofState.dtype)
        left_arm_states['pos'], _ = pink_solve_ik(target_pose=left_reletive_pose)
        self.gym.set_actor_dof_states(self.env, self.left_arm_handle, left_arm_states, gymapi.STATE_POS)

        right_arm_states = np.zeros(7, dtype=gymapi.DofState.dtype)
        right_arm_states['pos'], _ = pink_solve_ik(target_pose=right_reletive_pose)
        self.gym.set_actor_dof_states(self.env, self.right_arm_handle, right_arm_states, gymapi.STATE_POS)

        self.root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim) # 获取所有actor root state
        self.gym.refresh_actor_root_state_tensor(self.sim) # 对root state进行更新
        self.root_states = gymtorch.wrap_tensor(self.root_state_tensor) # 将root state包装成tensor 的数据格式

        # 通过left_idx和right_idx来访问root state中的每个手的状态，用于手部root state的更新
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
        self.left_cam_offset = np.array([0, 0.033, 0])
        self.right_cam_offset = np.array([0, -0.033, 0])
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

        self.previous_left_pose = None
        self.previous_right_pose = None

    def step(self, head_rmat, left_pose, right_pose, left_qpos, right_qpos):
        # left_pose, right_pose代表手根部的姿态, 为[x, y, z, qx, qy, qz, qw]形式
        # left_qpos, right_qpos代表手部所有关节的姿态

        if self.print_freq:
            start = time.time()

        # 获取hand相对于franka底座的相对姿态
        left_reletive_pose = get_reletive_hand_pose(left_pose, left_arm_pose, hand_pose_type="ndarray")
        right_reletive_pose = get_reletive_hand_pose(right_pose, right_arm_pose, hand_pose_type="ndarray")

        # ik结算franka姿态并在isaac gym中进行控制
        left_arm_states = np.zeros(7, dtype=gymapi.DofState.dtype)
        if self.previous_left_pose is None:
            current_left_pose, left_eef_relative_pose = pink_solve_ik(target_pose=left_reletive_pose)
        else:
            current_left_pose, left_eef_relative_pose = pink_solve_ik(target_pose=left_reletive_pose, initial_joint_state=self.previous_left_pose)
        self.previous_left_pose = current_left_pose
        left_arm_states['pos'] = current_left_pose
        self.gym.set_actor_dof_states(self.env, self.left_arm_handle, left_arm_states, gymapi.STATE_POS)

        # 在pinocchio中计算左臂末端link位姿进行retargeting
        pinocchio.forwardKinematics(franka_pin.model, franka_pin.data, current_left_pose)
        pinocchio.updateFramePlacements(franka_pin.model, franka_pin.data)
        ee_pose = franka_pin.data.oMf[ee_frame_id]
        left_eef_pose = get_absolute_hand_pose(ee_pose.homogeneous, left_arm_pose, hand_pose_type="ndarray")
        left_eef_pose = matrix_to_vector(left_eef_pose)

        # 直接在isaac gym中读取左臂末端link的位姿进行retargeting
        # rigid_body_states = self.gym.get_actor_rigid_body_states(
        #     self.env, self.left_arm_handle, gymapi.STATE_ALL
        # )
        # end_effector_state = rigid_body_states[-1]
        # position, orientation = end_effector_state[0]
        # left_eef_pose = np.array(list(position) + list(orientation))
        #
        # print("左eef姿态", left_eef_pose)
        # left_eef_pose = vector_to_matrix(left_eef_pose)
        # left_eef_pose = left_eef_pose @ eef_link_matrix
        # left_eef_pose = matrix_to_vector(left_eef_pose)

        # 在pinocchio中计算右臂末端link位姿进行retargeting
        right_arm_states = np.zeros(7, dtype=gymapi.DofState.dtype)
        if self.previous_right_pose is None:
            current_right_pose, right_eef_relative_pose = pink_solve_ik(target_pose=right_reletive_pose)
        else:
            current_right_pose, right_eef_relative_pose = pink_solve_ik(target_pose=right_reletive_pose, initial_joint_state=self.previous_right_pose)
        self.previous_right_pose = current_right_pose
        right_arm_states['pos'] = current_right_pose
        self.gym.set_actor_dof_states(self.env, self.right_arm_handle, right_arm_states, gymapi.STATE_POS)

        pinocchio.forwardKinematics(franka_pin.model, franka_pin.data, current_right_pose)
        pinocchio.updateFramePlacements(franka_pin.model, franka_pin.data)
        ee_pose = franka_pin.data.oMf[ee_frame_id]
        right_eef_pose = get_absolute_hand_pose(ee_pose.homogeneous, right_arm_pose, hand_pose_type="ndarray")
        right_eef_pose = matrix_to_vector(right_eef_pose)

        # 直接在isaac gym中读取右臂末端link的位姿进行retargeting
        # rigid_body_states = self.gym.get_actor_rigid_body_states(
        #     self.env, self.right_arm_handle, gymapi.STATE_ALL
        # )
        # end_effector_state = rigid_body_states[-1]
        # position, orientation = end_effector_state[0]
        # right_eef_pose = np.array(list(position) + list(orientation))
        # right_eef_pose = vector_to_matrix(right_eef_pose)
        # right_eef_pose = right_eef_pose @ eef_link_matrix
        # right_eef_pose = matrix_to_vector(right_eef_pose)


        # 直接显示quest中得到的手部姿态，不进行retargeting
        # self.left_root_states[0:7] = torch.tensor(left_pose, dtype=float)
        # self.right_root_states[0:7] = torch.tensor(right_pose, dtype=float)

        # 根据ik结果得到的机械臂末端link姿态进行retargeting
        self.left_root_states[0:7] = torch.tensor(left_eef_pose, dtype=float)
        self.right_root_states[0:7] = torch.tensor(right_eef_pose, dtype=float)
        # 更新手部root state状态
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))


        # 修改每个手的的关节角度
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
    teleoperator = VuerTeleop('inspire_hand.yml')
    simulator = Sim()

    try:
        while True:
            head_rmat, left_pose, right_pose, left_qpos, right_qpos = teleoperator.step()
            left_img, right_img = simulator.step(head_rmat, left_pose, right_pose, left_qpos, right_qpos)
            np.copyto(teleoperator.img_array, np.hstack((left_img, right_img)))
    except KeyboardInterrupt:
        simulator.end()
        exit(0)
