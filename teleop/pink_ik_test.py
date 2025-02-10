#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0

"""Move an end-effector to a prescribed target."""

import numpy as np
import pinocchio
from robot_descriptions.loaders.pinocchio import load_robot_description
import os
import pink
from pink.tasks import FrameTask
import yourdfpy
from scipy.spatial.transform import Rotation as R
from teleop_utils import *

# IK parameters
dt = 1e-2
stop_thres = 1e-8

if __name__ == "__main__":
    urdf_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "assets",
        "franka_pinocchio",
        "robots",
        "franka_panda.urdf",
    )

    robot = pinocchio.RobotWrapper.BuildFromURDF(
        filename=urdf_path,
        package_dirs=["."],
        root_joint=None,
    )
    print(f"URDF description successfully loaded in {robot}")

    # if yourdfpy is None:
    #     print("If you ``pip install yourdfpy``, this example will display it.")
    # else:  # yourdfpy is not None
    #     viz = yourdfpy.URDF.load(urdf_path)
    #     viz.show()

    # 末端姿态连接
    eef_link_trans = np.array([0, 0, 0.107])
    # 'xyz' 表示依次按 roll（X轴）、pitch（Y轴）、yaw（Z轴）的顺序旋转
    rpy = np.array([0, 0, 0])
    rot1 = R.from_euler('xyz', rpy)

    # 第二个旋转（R.from_euler('xyz', [np.deg2rad(-90), np.deg2rad(0), np.deg2rad(-90)])）
    rot2 = R.from_euler('xyz', [np.deg2rad(-90), np.deg2rad(0), np.deg2rad(-90)])

    # 组合两个旋转矩阵（按顺序相乘）
    eef_link_rotation = rot1




    # Frame details
    joint_name = robot.model.names[-1]
    parent_joint = robot.model.getJointId(joint_name)
    parent_frame = robot.model.getFrameId(joint_name)
    placement = ndarray_to_se3(trans_rot2matrix(eef_link_trans, eef_link_rotation))


    FRAME_NAME = "ee_frame"
    ee_frame = robot.model.addFrame(
        pinocchio.Frame(
            FRAME_NAME,
            parent_joint,
            parent_frame,
            placement,
            pinocchio.FrameType.OP_FRAME,
        )
    )
    robot.data = pinocchio.Data(robot.model)
    low = robot.model.lowerPositionLimit
    high = robot.model.upperPositionLimit
    # robot.q0 = pinocchio.neutral(robot.model)
    robot.q0 = (low + high)/2

    ee_frame_id = robot.model.getFrameId(FRAME_NAME)

    # Task details
    np.random.seed(0)
    q_final = np.array(
        [
            np.random.uniform(low=low[i], high=high[i], size=(1,))[0]
            for i in range(7)
        ]
    )
    pinocchio.forwardKinematics(robot.model, robot.data, q_final)
    target_pose = robot.data.oMi[parent_joint]
    ee_task = FrameTask(FRAME_NAME, [1.0, 1.0, 1.0], [1.0, 1.0, 1.0])
    ee_task.set_target(target_pose)

    configuration = pink.Configuration(robot.model, robot.data, robot.q0)
    error_norm = np.linalg.norm(ee_task.compute_error(configuration))
    franka_eef_pose = configuration.get_transform_frame_to_world(FRAME_NAME)

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
        if nb_steps >= 500:
            break

    viz = yourdfpy.URDF.load(urdf_path)
    viz.update_cfg(configuration.q)
    viz.show()
    print(f"Terminated after {nb_steps} steps with {error_norm = :.2}")