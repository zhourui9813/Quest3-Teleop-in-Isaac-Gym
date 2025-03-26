import math
import numpy as np
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

np.set_printoptions(precision=2, suppress=True)
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from pytransform3d import rotations

import time
import cv2
from TeleVision import OpenTeleVision
# import pyzed.sl as sl
import pyrealsense2 as rs
from dynamixel.active_cam import DynamixelAgent
from multiprocessing import Array, Process, shared_memory, Queue, Manager, Event, Semaphore
from real.camera_realsense import CameraRealSense
from real.recorder_rgbd_video import RecorderRGBDVideo
from real.common.cv2_util import optimal_row_cols, ImageDepthTransform, ImageDepthVisTransform
from real.multi_camera_visualizer import MultiCameraVisualizer
import copy
import time
import multiprocessing as mp
from multiprocessing.managers import SharedMemoryManager


max_obs_buffer_size = 60
camera_obs_latency = 0.125
bit_rate = 6000*1000
resolution = (1280, 720) # (640, 480)
obs_image_resolution = (640, 480)
cam_vis_resolution = (1920, 1080)
capture_fps = 30
video_paths = f"test.mp4"
num_threads = 23
get_time_budget = 0.2
bgr_to_rgb = True

resolution = (1280, 720)
crop_size_w = 1
crop_size_h = 0
resolution_cropped = (resolution[0] - crop_size_h, resolution[1] - 2 * crop_size_w)

# agent = DynamixelAgent(port="/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT8IT033-if00-port0")
# agent._robot.set_torque_mode(True)

# Create a Camera objec

# Create a InitParameters object and set configuration parameters

# Open the camera


# Capture 50 frames and stop
i = 0


img_shape = (resolution_cropped[0], 2 * resolution_cropped[1], 3)
img_height, img_width = resolution_cropped[:2]
shm = shared_memory.SharedMemory(create=True, size=np.prod(img_shape) * np.uint8().itemsize)
img_array = np.ndarray((img_shape[0], img_shape[1], 3), dtype=np.uint8, buffer=shm.buf)
image_queue = Queue()
toggle_streaming = Event()
tv = OpenTeleVision(resolution_cropped, shm.name, image_queue, toggle_streaming, cert_file=None, key_file=None, ngrok=False)

with SharedMemoryManager() as shm_manager:
    # compute resolution for vis
    rw, rh, col, row = optimal_row_cols(
        n_cameras=2,
        in_wh_ratio=4 / 3,
        max_resolution=cam_vis_resolution
    )

    transform = ImageDepthTransform(input_res=resolution, output_res=obs_image_resolution, bgr_to_rgb=bgr_to_rgb)
    vis_transform = ImageDepthVisTransform(input_res=resolution, output_res=(rw, rh), bgr_to_rgb=bgr_to_rgb)

    print("Transform Initialization completed")

    # TODO: use crea_hevc_nvenc to speedup the process.
    # video_depth_recorder = VideoDepthRecorder.create_hevc_nvenc(
    #     fps=capture_fps,
    #     input_pix_fmt='bgr24',
    #     bit_rate=bit_rate
    # )
    recorder_rgbd_video = RecorderRGBDVideo.create_h264(
        fps=capture_fps,
        input_pix_fmt='rgb24',
        bit_rate=bit_rate
    )

    print("VideoDepthRecorder Initialization completed")

    serial_number_list = CameraRealSense.get_connected_devices_serial()
    device_id = serial_number_list[0]

    camera = CameraRealSense(
        shm_manager=shm_manager,
        device_id=device_id,
        get_time_budget=get_time_budget,
        resolution=resolution,
        capture_fps=capture_fps,
        put_fps=None,
        put_downsample=False,
        get_max_k=max_obs_buffer_size,
        receive_latency=camera_obs_latency,
        num_threads=num_threads,
        transform=transform,
        vis_transform=vis_transform,
        recording_transform=None,  # will be set to transform by defaut
        recorder=recorder_rgbd_video,
        verbose=True,
    )

    print("Resolution for vis: ", rw, rh)
    camera.start(wait=False)
    time.sleep(2.0)
    print("start process. ")


# while True:
#     time.sleep(1)

while True:
    start = time.time()
    last_camera_data = None
    k = 4
    last_camera_data = camera.get(k=None, out=last_camera_data)
    # if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
    # zed.retrieve_image(image_left, sl.VIEW.LEFT)
    # zed.retrieve_image(image_right, sl.VIEW.RIGHT)
    # timestamp = zed.get_timestamp(sl.TIME_REFERENCE.CURRENT)  # Get the timestamp at the time the image was captured
    # print("Image resolution: {0} x {1} || Image timestamp: {2}\n".format(image.get_width(), image.get_height(),
    #         timestamp.get_milliseconds()))

    bgr = np.hstack((last_camera_data['color'][crop_size_h:, crop_size_w:-crop_size_w],
                     last_camera_data['color'][crop_size_h:, crop_size_w:-crop_size_w]))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGRA2RGB)

    np.copyto(img_array, rgb)

    end = time.time()
    # print(1/(end-start))
zed.close()