import math
import numpy as np
from scipy.spatial.transform import Rotation as R
np.set_printoptions(precision=2, suppress=True)
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from pytransform3d import rotations

import time
import cv2
from constants_vuer import *
from TeleVision import OpenTeleVision
# import pyzed.sl as sl
from dynamixel.active_cam import DynamixelAgent
from multiprocessing import Array, Process, shared_memory, Queue, Manager, Event, Semaphore

resolution = (720, 1280)
crop_size_w = 1
crop_size_h = 0
resolution_cropped = (resolution[0] - crop_size_h, resolution[1] - 2 * crop_size_w)

agent = DynamixelAgent(port="/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTA3H8CB-if00-port0")
agent._robot.set_torque_mode(True)

# Create a Camera object
# zed = sl.Camera()

# Create a InitParameters object and set configuration parameters
# init_params = sl.InitParameters()
# init_params.camera_resolution = sl.RESOLUTION.HD720  # Use HD720 opr HD1200 video mode, depending on camera type.
# init_params.camera_fps = 60  # Set fps at 60

# Open the camera
# err = zed.open(init_params)
# if err != sl.ERROR_CODE.SUCCESS:
#     print("Camera Open : " + repr(err) + ". Exit program.")
#     exit()

# Capture 50 frames and stop
# i = 0
# image_left = sl.Mat()
# image_right = sl.Mat()
# runtime_parameters = sl.RuntimeParameters()

# img_shape = (resolution_cropped[0], 2 * resolution_cropped[1], 3)
# img_height, img_width = resolution_cropped[:2]
# shm = shared_memory.SharedMemory(create=True, size=np.prod(img_shape) * np.uint8().itemsize)
# img_array = np.ndarray((img_shape[0], img_shape[1], 3), dtype=np.uint8, buffer=shm.buf)
# image_queue = Queue()
# toggle_streaming = Event()
# tv = OpenTeleVision(resolution_cropped, shm.name, image_queue, toggle_streaming)

# 假设我们有一个代表头部姿态的 RPY (Roll, Pitch, Yaw) 角度
roll = np.deg2rad(0)  # Roll 角度 (单位：弧度)
pitch = np.deg2rad(0)  # Pitch 角度 (单位：弧度)
yaw = np.deg2rad(0)  # Yaw 角度 (单位：弧度)

# 使用 SciPy 的 Rotation 类生成旋转矩阵
r = R.from_euler('xyz', [roll, pitch, yaw])  # 'xyz' 表示使用 Roll, Pitch, Yaw 的顺序
head_rotation_matrix = r.as_matrix()

# 输出旋转矩阵
print("Head rotation matrix (3x3):\n", head_rotation_matrix)

roll_range = np.linspace(np.deg2rad(-20), np.deg2rad(20), 50)  # -30到30之间变化，分10步
pitch_range = np.linspace(np.deg2rad(-20), np.deg2rad(20), 50)  # -30到30之间变化，分10步
yaw_range = np.linspace(np.deg2rad(-20), np.deg2rad(20), 50)  # -30到30之间变化，分10步

while True:
    for roll in roll_range:
        for pitch in pitch_range:
            for yaw in yaw_range:
                start = time.time()

                r = R.from_euler('xyz', [roll, pitch, yaw])  # 'xyz' 表示使用 Roll, Pitch, Yaw 的顺序
                head_rotation_matrix = r.as_matrix()

                head_mat = grd_yup2grd_zup[:3, :3] @ head_rotation_matrix @ grd_yup2grd_zup[:3, :3].T
                if np.sum(head_mat) == 0:
                    head_mat = np.eye(3)
                head_rot = rotations.quaternion_from_matrix(head_mat[0:3, 0:3])
                try:
                    ypr = rotations.euler_from_quaternion(head_rot, 2, 1, 0, False)
                    # print(ypr)
                    # agent._robot.command_joint_state([0., 0.4])
                    agent._robot.command_joint_state(ypr[:2])
                    # print("success")
                except:
                    # print("failed")
                    # exit()
                    pass
                end = time.time()
