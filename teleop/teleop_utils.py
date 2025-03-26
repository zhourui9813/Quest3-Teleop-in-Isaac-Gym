# isaac gym库存在问题，一定要先import pinocchio再import isaacgym
import pinocchio
from isaacgym import gymapi
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

# 将 gymapi.Transform 转换为齐次变换矩阵
def gympose2matrix(pose):
    # 提取四元数和位移
    quat = [pose.r.x, pose.r.y, pose.r.z, pose.r.w]
    translation = np.array([pose.p.x, pose.p.y, pose.p.z])

    rot = R.from_quat(quat)
    rot_mat = rot.as_matrix()

    TransMatrix = np.eye(4)
    TransMatrix[:3, :3] = rot_mat
    TransMatrix[:3, 3] = translation
    return TransMatrix

# 将 4×4 的齐次变换矩阵转换为 gymapi.Transform 对象
def matrix2gympose(matrix):
    # 提取平移和旋转矩阵
    translation = matrix[:3, 3]
    rot_mat = matrix[:3, :3]
    quat = R.from_matrix(rot_mat).as_quat()

    # 构造 gymapi.Transform
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(translation[0], translation[1], translation[2])
    pose.r = gymapi.Quat(quat[0], quat[1], quat[2], quat[3])
    return pose

# 将平移与旋转转换为齐次变换矩阵
def trans_rot2matrix(trans, rot):
    rot_mat = rot.as_matrix()
    T = np.eye(4)
    T[:3, :3] = rot_mat
    T[:3, 3] = trans
    return T

# 计算手部位姿 hand_pose（在世界坐标系下）相对于臂部位姿 arm_pose（在世界坐标系下）的相对位姿，
# 返回一个 gymapi.Transform 对象，表示手部在臂部坐标系下的位姿。
def get_reletive_hand_pose(hand_pose, arm_pose, hand_pose_type = "gympose"):
    if hand_pose_type == "gympose":
        T_hand = gympose2matrix(hand_pose)
    elif hand_pose_type == "ndarray":
        if hand_pose.shape != (7,):
            raise ValueError("The shape of hand_pose should be (7,)")

        translation = hand_pose[:3]
        quat = hand_pose[3:7]

        rot = R.from_quat(quat)
        rot_mat = rot.as_matrix()

        T_hand = np.eye(4)
        T_hand[:3, :3] = rot_mat
        T_hand[:3, 3] = translation

    T_arm = gympose2matrix(arm_pose)

    # 计算相对变换矩阵
    T_relative = np.linalg.inv(T_arm) @ T_hand

    return T_relative

# 根据手部在臂部坐标系中的相对位姿，计算手部在世界坐标系中的绝对位姿。
def get_absolute_hand_pose(relative_hand_pose, arm_pose, hand_pose_type="gympose"):
    if hand_pose_type == "gympose":
        T_relative = gympose2matrix(relative_hand_pose)
    else:
        T_relative = relative_hand_pose

    # 将臂部位姿转换为齐次变换矩阵
    T_arm = gympose2matrix(arm_pose)

    # 计算手部在世界坐标系中的绝对位姿
    T_hand = T_arm @ T_relative

    return T_hand

# 将4×4齐次变换矩阵（ndarray 类型）转换为 Pinocchio 的 SE3 对象
def ndarray_to_se3(T: np.ndarray) -> pinocchio.SE3:
    if T.shape != (4, 4):
        raise ValueError("The shape of input matrix should be (4, 4)")

    R = T[:3, :3]
    t = T[:3, 3]
    se3_obj = pinocchio.SE3(R, t)
    return se3_obj

# 将表示位姿的齐次变换矩阵转换为[x,y,z,qx,qy,qz,qw]的向量形式
def matrix_to_vector(T_matrix):

    translation = T_matrix[:3, 3]
    Rotation = T_matrix[:3, :3]

    quat = pinocchio.Quaternion(Rotation)
    quat_arr = quat.coeffs()

    vec = np.concatenate((translation, quat_arr), axis=0)
    return vec

# 将[x,y,z,qx,qy,qz,qw]的向量转换为表示位姿的齐次变换矩阵形式
def vector_to_matrix(vec):

    translation = vec[:3]
    quat = vec[3:]

    rot = R.from_quat(quat)
    R_mat = rot.as_matrix()

    T = np.eye(4)
    T[:3, :3] = R_mat
    T[:3, 3] = translation
    return T

def euler_to_matrix(euler):

    rotation = R.from_euler('xyz', euler)
    return rotation.as_matrix()


def relative_pose_to_absolute(base_pose, relative_pose):
    # 将机械臂base的位姿转换为变换矩阵
    T_base = gympose2matrix(base_pose)

    # 提取相对位姿的平移和欧拉角
    rel_translation = np.array(relative_pose[:3])
    rel_rpy = np.array(relative_pose[3:])

    # 将欧拉角转换为四元数
    rel_quat = R.from_euler('xyz', rel_rpy).as_quat()

    # 构造相对位姿的变换矩阵
    T_relative = np.eye(4)
    T_relative[:3, :3] = R.from_quat(rel_quat).as_matrix()
    T_relative[:3, 3] = rel_translation

    # 计算绝对位姿的变换矩阵
    T_absolute = np.dot(T_base, T_relative)

    # 提取绝对位置的坐标和欧拉角
    abs_translation = T_absolute[:3, 3]
    abs_rot_mat = T_absolute[:3, :3]
    abs_rpy = R.from_matrix(abs_rot_mat).as_euler('xyz')

    return abs_translation.tolist() + abs_rpy.tolist()


def plot_pose(transformation, ax):
    """
    绘制位姿
    :param transformation: 位姿数组，前三个为位置，后三个为欧拉角（弧度）
    :param ax: Matplotlib 3D坐标轴
    """
    # 提取位置和欧拉角
    position = transformation[:3]
    roll, pitch, yaw = transformation[3:]

    # 使用 scipy.spatial.transform.Rotation 创建旋转矩阵
    rotation = R.from_euler('xyz', [roll, pitch, yaw])
    R_matrix = rotation.as_matrix()

    # 基准坐标系原点
    origin = np.array([0, 0, 0])

    # 绘制基准坐标系
    ax.quiver(origin[0], origin[1], origin[2], 1, 0, 0, color='r', length=1.0, arrow_length_ratio=0.1,
              label='X-axis (ref)')
    ax.quiver(origin[0], origin[1], origin[2], 0, 1, 0, color='g', length=1.0, arrow_length_ratio=0.1,
              label='Y-axis (ref)')
    ax.quiver(origin[0], origin[1], origin[2], 0, 0, 1, color='b', length=1.0, arrow_length_ratio=0.1,
              label='Z-axis (ref)')

    # 绘制位姿坐标系
    ax.quiver(position[0], position[1], position[2], R_matrix[0, 0], R_matrix[1, 0], R_matrix[2, 0], color='r',
              length=1.0, arrow_length_ratio=0.1, label='X-axis (pose)')
    ax.quiver(position[0], position[1], position[2], R_matrix[0, 1], R_matrix[1, 1], R_matrix[2, 1], color='g',
              length=1.0, arrow_length_ratio=0.1, label='Y-axis (pose)')
    ax.quiver(position[0], position[1], position[2], R_matrix[0, 2], R_matrix[1, 2], R_matrix[2, 2], color='b',
              length=1.0, arrow_length_ratio=0.1, label='Z-axis (pose)')

    # 绘制位姿原点与基准坐标系原点的连线
    ax.plot([origin[0], position[0]], [origin[1], position[1]], [origin[2], position[2]], 'k--',
            label='Connection line')

    # 设置坐标轴标签和标题
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Pose Visualization')

    # 设置坐标轴范围以放大显示
    ax.set_xlim([-3, 3])
    ax.set_ylim([-3, 3])
    ax.set_zlim([-3, 3])

    ax.legend()



