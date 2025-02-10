import sys
import time
import math
import numpy as np
from scipy.spatial.transform import Rotation as R

# PyQt5相关
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QSlider, QVBoxLayout, QWidget, QLabel, QSizePolicy, QShortcut
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QKeySequence

# matplotlib 相关
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

# 3D 旋转相关
from pytransform3d import rotations

# 引入您已有的矩阵和 Dynamixel 控制
from constants_vuer import grd_yup2grd_zup
from dynamixel.active_cam import DynamixelAgent

# -------------------------------
# 在此初始化 DynamixelAgent
# -------------------------------
agent = DynamixelAgent(port="/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTA3H8CB-if00-port0")
agent._robot.set_torque_mode(True)

# 全局 roll/pitch/yaw（单位：度）
roll_deg = 0
pitch_deg = 0
yaw_deg = 0

def rotation_matrix(roll_deg, pitch_deg, yaw_deg):
    """
    使用 scipy Rotation 通过 'xyz' 欧拉角(roll, pitch, yaw) 生成 3x3 旋转矩阵。
    注意：roll/pitch/yaw 传入的单位是度，这里要先转弧度。
    """
    roll_rad = np.deg2rad(roll_deg)
    pitch_rad = np.deg2rad(pitch_deg)
    yaw_rad = np.deg2rad(yaw_deg)
    r = R.from_euler('xyz', [roll_rad, pitch_rad, yaw_rad])
    return r.as_matrix()

class ArrowGimbalControl(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('3D Arrow + Dynamixel Control')
        self.showMaximized()  # 或者 showFullScreen()

        # 按下 ESC 退出
        self.esc_shortcut = QShortcut(QKeySequence(Qt.Key_Escape), self)
        self.esc_shortcut.activated.connect(self.close)

        # 创建 Matplotlib Figure 和 Canvas
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        # 使得 FigureCanvas 在窗口拉伸时自适应
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.updateGeometry()

        # 3D 坐标轴
        self.ax = self.figure.add_subplot(111, projection='3d')
        self.figure.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

        # 三个滑块 + 标签
        self.roll_label = QLabel('Roll (°)')
        self.roll_slider = QSlider(Qt.Horizontal)
        self.roll_slider.setMinimum(-180)
        self.roll_slider.setMaximum(180)
        self.roll_slider.setValue(0)
        self.roll_slider.valueChanged.connect(self.update_roll)

        self.pitch_label = QLabel('Pitch (°)')
        self.pitch_slider = QSlider(Qt.Horizontal)
        self.pitch_slider.setMinimum(-180)
        self.pitch_slider.setMaximum(180)
        self.pitch_slider.setValue(0)
        self.pitch_slider.valueChanged.connect(self.update_pitch)

        self.yaw_label = QLabel('Yaw (°)')
        self.yaw_slider = QSlider(Qt.Horizontal)
        self.yaw_slider.setMinimum(-180)
        self.yaw_slider.setMaximum(180)
        self.yaw_slider.setValue(0)
        self.yaw_slider.valueChanged.connect(self.update_yaw)

        # 布局
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        layout.addWidget(self.roll_label)
        layout.addWidget(self.roll_slider)
        layout.addWidget(self.pitch_label)
        layout.addWidget(self.pitch_slider)
        layout.addWidget(self.yaw_label)
        layout.addWidget(self.yaw_slider)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # 首次绘制
        self.update_plot()

    def update_roll(self, value):
        global roll_deg
        roll_deg = value
        self.update_plot()

    def update_pitch(self, value):
        global pitch_deg
        pitch_deg = value
        self.update_plot()

    def update_yaw(self, value):
        global yaw_deg
        yaw_deg = value
        self.update_plot()

    def update_plot(self):
        """同时更新 3D 箭头可视化和云台角度。"""
        global roll_deg, pitch_deg, yaw_deg

        # --- 第一步：更新 3D 箭头可视化 ---
        self.ax.clear()
        self.ax.set_xlim([-2, 2])
        self.ax.set_ylim([-2, 2])
        self.ax.set_zlim([-2, 2])
        self.ax.set_xlabel('X axis')
        self.ax.set_ylabel('Y axis')
        self.ax.set_zlabel('Z axis')

        # 定义人体局部坐标系下的三个方向向量
        # 这里假定:
        #   前方(主箭头)  -> +X
        #   上方(辅助箭头)-> +Z
        #   左方(辅助箭头)-> +Y  (也可以改成右方 -Y，看习惯)
        arrow_forward = np.array([1, 0, 0])
        arrow_up = np.array([0, 0, 1])
        arrow_left = np.array([0, 1, 0])  # 也可以命名为 arrow_side

        # 计算旋转矩阵 (从 roll_deg, pitch_deg, yaw_deg)
        R_mat = rotation_matrix(roll_deg, pitch_deg, yaw_deg)

        # 将这三个方向向量旋转到全局坐标
        arrow_forward_glob = R_mat @ arrow_forward
        arrow_up_glob = R_mat @ arrow_up
        arrow_left_glob = R_mat @ arrow_left

        # 绘制主箭头 (加粗)
        self.ax.quiver(
            0, 0, 0,
            arrow_forward_glob[0], arrow_forward_glob[1], arrow_forward_glob[2],
            color='r',
            length=2,                 # 箭头长度
            arrow_length_ratio=0.2,   # 箭头尖端比例
            linewidth=4               # 线宽(加粗)
        )

        # 绘制上方箭头(细一些)
        self.ax.quiver(
            0, 0, 0,
            arrow_up_glob[0], arrow_up_glob[1], arrow_up_glob[2],
            color='g',
            length=2,
            arrow_length_ratio=0.2,
            linewidth=1
        )

        # 绘制侧方箭头(细一些)
        self.ax.quiver(
            0, 0, 0,
            arrow_left_glob[0], arrow_left_glob[1], arrow_left_glob[2],
            color='b',
            length=2,
            arrow_length_ratio=0.2,
            linewidth=1
        )

        # --- 第二步：计算并发送关节命令给 Dynamixel (云台) ---
        try:
            # 将欧拉角(度) 转为 弧度，再转成 3x3
            r_scipy = R.from_euler('xyz', [
                np.deg2rad(roll_deg),
                np.deg2rad(pitch_deg),
                np.deg2rad(yaw_deg)
            ])
            head_rotation_matrix = r_scipy.as_matrix()


            # head_mat = grd_yup2grd_zup[:3, :3] @ head_rotation_matrix @ grd_yup2grd_zup[:3, :3].T
            # 这个旋转矩阵可能要根据实际情况修改，demo里先不使用
            head_mat = head_rotation_matrix
            if np.sum(head_mat) == 0:
                head_mat = np.eye(3)

            # 转成四元数
            head_rot_quat = rotations.quaternion_from_matrix(head_mat)
            # 再将四元数转成 yaw-pitch-roll 或其他顺序
            ypr = rotations.euler_from_quaternion(head_rot_quat, 2, 1, 0, False)

            # 发送指令给 Dynamixel (云台)
            agent._robot.command_joint_state(ypr[:2])

        except Exception as e:
            print("Command joint state failed:", e)

        # --- 第三步：刷新绘图 ---
        self.canvas.draw()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ArrowGimbalControl()
    window.show()
    sys.exit(app.exec_())
