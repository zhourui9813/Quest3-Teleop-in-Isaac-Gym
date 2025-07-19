# isaac gym库存在问题，一定要先import pinocchio再import isaacgym
import pinocchio
import os
import pink
from pink.tasks import FrameTask
from teleop.teleop_utils import *
from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
import numpy as np
import math
import cv2
import time



# Franka 底座位置
base_height = 0.9

left_arm_base = gymapi.Transform()
left_arm_base.p = gymapi.Vec3(-1, 0.25, base_height)
left_arm_base.r = gymapi.Quat(0, 0, 0, 1)

# 根据机械臂初始位姿获得机械臂坐标系相对于世界坐标系的变换矩阵
left_base2world = gympose2matrix(left_arm_base)

right_arm_base = gymapi.Transform()
right_arm_base.p = gymapi.Vec3(-1, -0.25, base_height)
right_arm_base.r = gymapi.Quat(0, 0, 0, 1)

right_base2world = gympose2matrix(right_arm_base)

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
                 assets_path,
                 left_arm_pose,
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
        cube_asset = self.gym.create_box(self.sim, 0.04, 0.04, 0.04, cube_asset_options)

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
        table_handle = self.gym.create_actor(self.env, table_asset, pose, 'table', -1)
        color = gymapi.Vec3(0.5, 0.5, 1.0)
        self.gym.set_rigid_body_color(self.env, table_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)

        # cube
        pose = gymapi.Transform()  # 在 Isaac Gym 中，gymapi.Transform() 用于构造一个“位姿”对象（3D 平移 + 旋转）
        pose.p = gymapi.Vec3(-0.4, 0, base_height + 0.5)
        pose.r = gymapi.Quat(0, 0, 0, 1)
        # actor是GymAsset的实例。函数create_actor将一个参与者添加到环境中，并返回一个参与者句柄，该句柄可用于以后与该参与者交互
        cube_handle = self.gym.create_actor(self.env, cube_asset, pose, 'cube', -1)
        cube_shape = self.gym.get_actor_rigid_shape_properties(self.env, cube_handle)
        for s in cube_shape:
            s.friction = 5.0
        self.gym.set_actor_rigid_shape_properties(self.env, cube_handle, cube_shape)
        color = gymapi.Vec3(1, 0.5, 0.5)
        self.gym.set_rigid_body_color(self.env, cube_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)

        # 定义urdf文件路径
        asset_root = assets_path
        right_arm_asset_path = "franka_inspire_hand/franka_description/robots/franka_panda_right.urdf"
        left_arm_asset_path = "franka_inspire_hand/franka_description/robots/franka_panda_left.urdf"
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.disable_gravity = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS  # 所有关节都使用位置控制模式
        right_arm_asset = self.gym.load_asset(self.sim, asset_root, right_arm_asset_path, asset_options)
        left_arm_asset = self.gym.load_asset(self.sim, asset_root, left_arm_asset_path, asset_options)

        self.right_dof = self.gym.get_asset_dof_count(right_arm_asset)
        self.left_dof = self.gym.get_asset_dof_count(left_arm_asset)

        self.right_arm_handle = self.gym.create_actor(self.env, right_arm_asset, right_arm_pose, 'right_arm', 1, 1)
        self.left_arm_handle = self.gym.create_actor(self.env, left_arm_asset, left_arm_pose, 'left_arm', 2, 1)
        for arm in [self.left_arm_handle, self.right_arm_handle]:
            finger_shapes = self.gym.get_actor_rigid_shape_properties(self.env, arm)
            for s in finger_shapes:
                s.friction = 5.0
            self.gym.set_actor_rigid_shape_properties(self.env, arm, finger_shapes)

        self.gym.set_actor_dof_states(self.env, self.right_arm_handle, np.zeros(self.right_dof, gymapi.DofState.dtype),
                                      gymapi.STATE_ALL)
        self.gym.set_actor_dof_states(self.env, self.left_arm_handle, np.zeros(self.left_dof, gymapi.DofState.dtype),
                                      gymapi.STATE_ALL)

        right_arm_states = np.zeros(self.right_dof, dtype=gymapi.DofState.dtype)
        left_arm_states = np.zeros(self.left_dof, dtype=gymapi.DofState.dtype)

        init_joint_position = [0., 0., 0., -1.507, 0., 1.507,
                               0., 0.1, 0., 0., 0., 0.,
                               0., 0., 0., 0., 0., 0.,
                               0.]

        # right_arm_states['pos'] = init_joint_position
        # left_arm_states['pos'] = init_joint_position

        left_arm_states['pos'] = init_joint_position
        left_arm_states['vel'] = np.zeros_like(init_joint_position)  # new
        right_arm_states['pos'] = init_joint_position
        right_arm_states['vel'] = np.zeros_like(init_joint_position)  # new


        assert self.left_dof == len(init_joint_position), \
            f"DOF mismatch (asset={self.left_dof}, init={len(init_joint_position)})"

        self.previous_right_joint_position = np.array(init_joint_position[:7])
        self.previous_left_joint_position = np.array(init_joint_position[:7])

        self.gym.set_actor_dof_states(self.env, self.right_arm_handle, right_arm_states, gymapi.STATE_POS)
        self.gym.set_actor_dof_states(self.env, self.left_arm_handle, left_arm_states, gymapi.STATE_POS)

        self.set_seg_id(self.right_arm_handle, 1)
        self.set_seg_id(self.left_arm_handle, 2)

        # create default viewer
        self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
        if self.viewer is None:
            print("*** Failed to create viewer")
            quit()
        cam_pos = gymapi.Vec3(1, 1, 2)
        cam_target = gymapi.Vec3(0, 0, 1)
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        self.cam_lookat_offset = np.array([1, 0, 0])
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

    def depth_img_to_color(self, depth_img):
        # 先进行深度图归一化
        normalized_depth = self.depth_img_normalize(depth_img)

        # 使用cv2的颜色映射将灰度深度图转换为彩色图
        colored_depth = cv2.applyColorMap(normalized_depth, cv2.COLORMAP_JET)

        # 转换BGR到RGB
        colored_depth = cv2.cvtColor(colored_depth, cv2.COLOR_BGR2RGB)

        return colored_depth

    def step(self, head_rmat, head_pose, left_arm_pose, right_arm_pose, left_hand_pose, right_hand_pose, issac_viewer=True):

        if self.print_freq:
            start = time.time()

        # head_rmat = R.from_rotvec(head_pose[3:]).as_matrix()
        # self.cam_pos = head_pose

        # 获取hand相对于franka底座的相对姿态
        # 注意：这里的left_arm_pose和right_arm_pose是从teleoperator传来的手部姿态
        # 需要使用全局的arm pose作为底座姿态
        left_relative_pose = get_relative_hand_pose(left_arm_pose, left_arm_base)
        right_relative_pose = get_relative_hand_pose(right_arm_pose, right_arm_base)

        # 在pinocchio中计算左臂末端link位姿
        if self.previous_left_joint_position is None:
            current_left_joint_position, left_eef_relative_pose = pink_solve_ik(target_pose=left_relative_pose,
                                                                                robot=self.robot_pin,
                                                                                frame_name="hand_base_link",
                                                                                dt=self.ik_dt,
                                                                                stop_thres=self.ik_thresh,
                                                                                max_iterate_step=self.ik_max_iterate_step,
                                                                                initial_joint_state=None
                                                                                )
        else:
            current_left_joint_position, left_eef_relative_pose = pink_solve_ik(target_pose=left_relative_pose,
                                                                                robot=self.robot_pin,
                                                                                frame_name="hand_base_link",
                                                                                dt=self.ik_dt,
                                                                                stop_thres=self.ik_thresh,
                                                                                max_iterate_step=self.ik_max_iterate_step,
                                                                                initial_joint_state=self.previous_left_joint_position[
                                                                                                    :7]
                                                                                )

        # 在pinocchio中计算右臂末端link位姿
        if self.previous_right_joint_position is None:
            current_right_joint_position, right_eef_relative_pose = pink_solve_ik(target_pose=right_relative_pose,
                                                                                  robot=self.robot_pin,
                                                                                  frame_name="hand_base_link",
                                                                                  dt=self.ik_dt,
                                                                                  stop_thres=self.ik_thresh,
                                                                                  max_iterate_step=self.ik_max_iterate_step,
                                                                                  initial_joint_state=None
                                                                                  )
        else:
            current_right_joint_position, right_eef_relative_pose = pink_solve_ik(target_pose=right_relative_pose,
                                                                                  robot=self.robot_pin,
                                                                                  frame_name="hand_base_link",
                                                                                  dt=self.ik_dt,
                                                                                  stop_thres=self.ik_thresh,
                                                                                  max_iterate_step=self.ik_max_iterate_step,
                                                                                  initial_joint_state=self.previous_right_joint_position[:7]
                                                                                  )

        self.previous_left_joint_position = current_left_joint_position
        self.previous_right_joint_position = current_right_joint_position

        left_joint_position = np.concatenate([current_left_joint_position, left_hand_pose])
        right_joint_position = np.concatenate([current_right_joint_position, right_hand_pose])

        left_arm_states = np.zeros(self.left_dof, dtype=gymapi.DofState.dtype)
        right_arm_states = np.zeros(self.right_dof, dtype=gymapi.DofState.dtype)

        left_arm_states['pos'] = left_joint_position
        right_arm_states['pos'] = right_joint_position

        self.gym.set_actor_dof_states(self.env, self.left_arm_handle, left_arm_states, gymapi.STATE_POS)
        self.gym.set_actor_dof_states(self.env, self.right_arm_handle, right_arm_states, gymapi.STATE_POS)

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
        left_depth_image = self.depth_img_to_color(left_depth_image)
        right_depth_image = self.depth_img_to_color(right_depth_image)

        left_mask = self.gym.get_camera_image(self.sim, self.env, self.left_camera_handle,
                                              gymapi.IMAGE_SEGMENTATION)
        right_mask = self.gym.get_camera_image(self.sim, self.env, self.right_camera_handle,
                                               gymapi.IMAGE_SEGMENTATION)

        if issac_viewer:
            self.gym.draw_viewer(self.viewer, self.sim, True)
        self.gym.sync_frame_time(self.sim)

        if self.print_freq:
            end = time.time()
            print('Frequency:', 1 / (end - start))

        return left_color_image, right_color_image, left_depth_image, right_depth_image, left_mask.astype(
            np.uint8), right_mask.astype(np.uint8)

    def end(self):
        self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)