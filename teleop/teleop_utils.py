# isaac gym库存在问题，一定要先import pinocchio再import isaacgym
import pinocchio
from isaacgym import gymapi
import numpy as np
from scipy.spatial.transform import Rotation as R
import scipy.spatial.transform as st
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

def gympose2pose6d(pose):
    # 提取四元数和位移
    quat = [pose.r.x, pose.r.y, pose.r.z, pose.r.w]
    translation = np.array([pose.p.x, pose.p.y, pose.p.z])

    # 转换四元数为欧拉角
    rot_euler = R.from_quat(quat).as_euler('xyz', degrees=False)

    # 合并位移与欧拉角并返回
    return np.concatenate([translation, rot_euler])

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
# def get_reletive_hand_pose(hand_pose, arm_pose, hand_pose_type = "gympose"):
#     if hand_pose_type == "gympose":
#         T_hand = gympose2matrix(hand_pose)
#     elif hand_pose_type == "ndarray":
#         if hand_pose.shape != (7,):
#             raise ValueError("The shape of hand_pose should be (7,)")
#
#         translation = hand_pose[:3]
#         quat = hand_pose[3:7]
#
#         rot = R.from_quat(quat)
#         rot_mat = rot.as_matrix()
#
#         T_hand = np.eye(4)
#         T_hand[:3, :3] = rot_mat
#         T_hand[:3, 3] = translation
#
#     T_arm = gympose2matrix(arm_pose)
#
#     # 计算相对变换矩阵
#     T_relative = np.linalg.inv(T_arm) @ T_hand
#
#     return T_relative

def get_relative_hand_pose(hand_pose, arm_pose):
    """
    hand_pose:
        - isaacgym.gymapi.Transform: 通过属性 .p/.r 自动解析 (xyzw)
        - array-like 长度7: [px, py, pz, qx, qy, qz, qw]  (xyzw)
    arm_pose: 仍按原逻辑是 gympose (传给 gympose2matrix)

    返回: 4x4 齐次矩阵, hand 相对 arm.
    """

    # ---- 自动识别 hand_pose ----
    if hasattr(hand_pose, "p") and hasattr(hand_pose, "r"):
        # Isaac Gym Transform: hand_pose.p 为位置, hand_pose.r 为四元数 (x,y,z,w).
        t = np.array([hand_pose.p.x, hand_pose.p.y, hand_pose.p.z], dtype=float)
        q = np.array([hand_pose.r.x, hand_pose.r.y, hand_pose.r.z, hand_pose.r.w], dtype=float)  # xyzw
        rot_mat = R.from_quat(q).as_matrix()  # SciPy默认 scalar-last (x,y,z,w)
        T_hand = np.eye(4, dtype=float)
        T_hand[:3, :3] = rot_mat
        T_hand[:3, 3] = t
    else:
        arr = np.asarray(hand_pose, dtype=float).reshape(-1)
        if arr.shape != (7,):
            raise ValueError("hand_pose 应为 isaacgym.gymapi.Transform 或长为7的序列 [px,py,pz,qx,qy,qz,qw]。")
        t = arr[:3]
        q = arr[3:7]  # 假定 xyzw
        rot_mat = R.from_quat(q).as_matrix()
        T_hand = np.eye(4, dtype=float)
        T_hand[:3, :3] = rot_mat
        T_hand[:3, 3] = t

    # arm_pose 按原写法 (gympose)
    T_arm = gympose2matrix(arm_pose)

    # 相对变换
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

    # matrix = pose_to_mat(transformation)
    # R_matrix = matrix[:3, :3]

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


def transform_relative_pose(absolute_pose, realitive_pose):
    """
    计算 T2 在全局坐标系中的绝对位姿 T2'
    :param T1: (6,) 数组，表示 T1 的绝对位姿 (x, y, z, roll, pitch, yaw)
    :param T2: (6,) 数组，表示 T2 在 T1 坐标系下的相对位姿 (x, y, z, roll, pitch, yaw)
    :return: (6,) 数组，表示 T2 在全局坐标系下的绝对位姿
    """
    # 提取平移和旋转
    trans_absolute, rot_absolute = absolute_pose[:3], R.from_euler('xyz', absolute_pose[3:], degrees=False)
    trans_relative, rot_relative = realitive_pose[:3], R.from_euler('xyz', realitive_pose[3:], degrees=False)

    # 计算全局平移
    t2_global = trans_absolute + rot_absolute.apply(trans_relative)  # t2 在 T1 坐标系下，转换到全局坐标系

    # 计算全局旋转
    r2_global = rot_absolute * rot_relative  # 旋转矩阵相乘（相对旋转叠加）

    # 转换回欧拉角
    euler2_global = r2_global.as_euler('xyz', degrees=False)

    return np.hstack((t2_global, euler2_global))


def pose_euler_to_quaternion(pose):
    """
    将六位姿数组中的后三位欧拉角(Roll, Pitch, Yaw)转换为四元数
    :param pose: 长度为6的数组 [x, y, z, roll, pitch, yaw]
    :return: 长度为7的数组 [x, y, z, qx, qy, qz, qw]
    """
    # 提取平移部分和旋转部分
    translation = pose[:3]
    euler_angles = pose[3:]

    # 使用scipy将欧拉角转换为四元数
    rotation = R.from_euler('xyz', euler_angles, degrees=False)
    quaternion = rotation.as_quat()  # 返回格式为 [qx, qy, qz, qw]

    # 拼接平移和四元数部分
    result = np.hstack((translation, quaternion))
    return result

def pos_rot_to_mat(pos, rot):
    shape = pos.shape[:-1]
    mat = np.zeros(shape + (4,4), dtype=pos.dtype)
    mat[...,:3,3] = pos
    mat[...,:3,:3] = rot.as_matrix()
    mat[...,3,3] = 1
    return mat

def mat_to_pos_rot(mat):
    pos = (mat[...,:3,3].T / mat[...,3,3].T).T
    rot = st.Rotation.from_matrix(mat[...,:3,:3])
    return pos, rot

def pos_rot_to_pose(pos, rot):
    shape = pos.shape[:-1]
    pose = np.zeros(shape+(6,), dtype=pos.dtype)
    pose[...,:3] = pos
    pose[...,3:] = rot.as_rotvec()
    return pose

def pose_to_pos_rot(pose):
    pos = pose[...,:3]
    rot = st.Rotation.from_rotvec(pose[...,3:])
    return pos, rot

def pose_to_mat(pose):
    return pos_rot_to_mat(*pose_to_pos_rot(pose))

def mat_to_pose(mat):
    return pos_rot_to_pose(*mat_to_pos_rot(mat))

def transform_pose(tx, pose):
    """
    tx: tx_new_old
    pose: tx_old_obj
    result: tx_new_obj
    """
    pose_mat = pose_to_mat(pose)
    tf_pose_mat = tx @ pose_mat
    tf_pose = mat_to_pose(tf_pose_mat)
    return tf_pose

def transform_point(tx, point):
    return point @ tx[:3,:3].T + tx[:3,3]

def project_point(k, point):
    x = point @ k.T
    uv = x[...,:2] / x[...,[2]]
    return uv

def apply_delta_pose(pose, delta_pose):
    new_pose = np.zeros_like(pose)

    # simple add for position
    new_pose[:3] = pose[:3] + delta_pose[:3]

    # matrix multiplication for rotation
    rot = st.Rotation.from_rotvec(pose[3:])
    drot = st.Rotation.from_rotvec(delta_pose[3:])
    new_pose[3:] = (drot * rot).as_rotvec()

    return new_pose

def normalize(vec, tol=1e-7):
    return vec / np.maximum(np.linalg.norm(vec), tol)

def rot_from_directions(from_vec, to_vec):
    from_vec = normalize(from_vec)
    to_vec = normalize(to_vec)
    axis = np.cross(from_vec, to_vec)
    axis = normalize(axis)
    angle = np.arccos(np.dot(from_vec, to_vec))
    rotvec = axis * angle
    rot = st.Rotation.from_rotvec(rotvec)
    return rot

def normalize(vec, eps=1e-12):
    norm = np.linalg.norm(vec, axis=-1)
    norm = np.maximum(norm, eps)
    out = (vec.T / norm).T
    return out

def rot6d_to_mat(d6):
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = normalize(a1)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = normalize(b2)
    b3 = np.cross(b1, b2, axis=-1)
    out = np.stack((b1, b2, b3), axis=-2)
    return out

def mat_to_rot6d(mat):
    batch_dim = mat.shape[:-2]
    out = mat[..., :2, :].copy().reshape(batch_dim + (6,))
    return out

def mat_to_pose10d(mat):
    pos = mat[...,:3,3]
    rotmat = mat[...,:3,:3]
    d6 = mat_to_rot6d(rotmat)
    d10 = np.concatenate([pos, d6], axis=-1)
    return d10

def pose10d_to_mat(d10):
    pos = d10[...,:3]
    d6 = d10[...,3:]
    rotmat = rot6d_to_mat(d6)
    out = np.zeros(d10.shape[:-1]+(4,4), dtype=d10.dtype)
    out[...,:3,:3] = rotmat
    out[...,:3,3] = pos
    out[...,3,3] = 1
    return out



def pose7d_to_mat(pose):
    # 提取位置和四元数
    position = pose[:3]  # 前三维是位置 [x, y, z]
    quaternion = pose[3:]  # 后四维是四元数 [w, x, y, z]

    # 将四元数转换为旋转矩阵
    rotation = R.from_quat(quaternion).as_matrix()

    # 构建齐次变换矩阵
    homogeneous_matrix = np.eye(4)
    homogeneous_matrix[:3, :3] = rotation
    homogeneous_matrix[:3, 3] = position

    return homogeneous_matrix

def extract_frames(video_path, output_dir, frame_interval=1):
    """
    使用 OpenCV 逐帧抽取视频并保存
    :param video_path: 输入视频路径
    :param output_dir: 输出帧保存目录
    :param frame_interval: 帧间隔，表示每隔多少帧抽取一帧，默认为1（逐帧抽取）
    """
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Unable to open the video")
        return

    frame_count = 0
    while True:
        # 读取一帧
        ret, frame = cap.read()
        if not ret:
            break  # 视频结束

        # 按照帧间隔保存帧
        if frame_count % frame_interval == 0:
            frame_name = f"{frame_count:05d}.jpg"
            output_path = os.path.join(output_dir, frame_name)
            cv2.imwrite(output_path, frame)
            print(f"save frame: {output_path}")

        frame_count += 1

    # 释放资源
    cap.release()
    print("Frames extraction is complete")