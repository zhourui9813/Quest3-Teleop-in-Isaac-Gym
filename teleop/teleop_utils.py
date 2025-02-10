# isaac gym库存在问题，一定要先import pinocchio再import isaacgym
import pinocchio
from isaacgym import gymapi
import numpy as np
from scipy.spatial.transform import Rotation as R

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
