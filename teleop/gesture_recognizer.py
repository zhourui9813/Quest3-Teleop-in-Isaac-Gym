# gesture_recognizer.py
import numpy as np
from enum import Enum, auto
from typing import Dict


class Pinch(Enum):
    """常用捏合组合；可按需再扩展。"""
    THUMB_INDEX = auto()
    THUMB_MIDDLE = auto()
    THUMB_RING = auto()
    THUMB_PINKY = auto()


class GestureRecognizer:
    """
    21‑点手部模型的手势识别器。

    参数
    ----
    finger_tip_indices : Dict[str, int]
        手指名称到指尖在 21‑点模型中的索引映射。
        默认按 mediapipe / inspire 模型：
        {thumb:4, index:8/9, middle:12/14, …}
    """

    def __init__(self,
                 finger_tip_indices: Dict[str, int] = None,
                 pos_coeff: float = 1.5,
                 neg_coeff: float = 0.8):
        # 默认索引（与您代码里保持一致）
        default_indices = {
            "thumb": 4, "index": 9, "middle": 14,
            "ring": 19, "pinky": 24
        }
        self.tip_idx = finger_tip_indices or default_indices
        self.pos_coeff = pos_coeff  # 正判捏合距离系数
        self.neg_coeff = neg_coeff  # 排他距离系数

    # ------------------- 公共 API -------------------
    def detect(self, hand_mat: np.ndarray) -> Dict[Pinch, bool]:
        """
        检测单只手的捏合手势。

        参数
        ----
        hand_mat : (25, 3) ndarray
            21/25‑点手部关键点坐标，顺序与 tip_idx 一致。

        返回
        ----
        gestures : Dict[Pinch, bool]
            各捏合手势的触发情况。
        """
        # 以食指—中指间距做参考，估一个自适应阈
        ref_len = np.linalg.norm(hand_mat[1] - hand_mat[2])
        pos_th = 0.1 if ref_len == 0 else ref_len / self.pos_coeff
        neg_th = 0.2 if ref_len == 0 else ref_len / self.neg_coeff

        # 工具函数
        def _close(a, b, th):  # a、b 为指名字符串
            idx_a, idx_b = self.tip_idx[a], self.tip_idx[b]
            return np.linalg.norm(hand_mat[idx_a] - hand_mat[idx_b]) < th

        # 遍历需要检测的捏合组合
        results = {}
        for finger in ("index", "middle", "ring", "pinky"):
            pinch = getattr(Pinch, f"THUMB_{finger.upper()}")
            # ① thumb 与 finger 要靠得近
            if not _close("thumb", finger, pos_th):
                results[pinch] = False
                continue
            # ② 其它手指指尖必须“张开”（避免误触）
            # ---- 条件 2：拇指必须与“非目标”手指保持足够远 ----
            # 找到除了拇指和目标指（finger_name）以外的 3 根手指
            other_fingers = [
                name for name in self.tip_idx
                if name not in ("thumb", finger)
            ]

            # 逐指检查：若拇指与任意其他指过近（<_neg_th），就视为失败
            thumb_far_from_all_others = True
            for other in other_fingers:
                if _close("thumb", other, neg_th):  # 距离 < neg_th ⇒ 过近
                    thumb_far_from_all_others = False
                    break

            # 只有两个条件都满足，才算识别到当前捏合手势
            results[pinch] = thumb_far_from_all_others
        return results

    # ------------- 也可直接判断某一手势 -------------
    def is_pinch(self, hand_mat: np.ndarray, pinch: Pinch) -> bool:
        return self.detect(hand_mat)[pinch]
