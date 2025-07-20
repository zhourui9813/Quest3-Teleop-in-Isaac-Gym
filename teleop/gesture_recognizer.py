# gesture_recognizer.py
import numpy as np
from enum import Enum, auto
from typing import Dict


class Pinch(Enum):
    """Common pinch combinations; can be extended as needed."""
    THUMB_INDEX = auto()
    THUMB_MIDDLE = auto()
    THUMB_RING = auto()
    THUMB_PINKY = auto()


class GestureRecognizer:
    """
    Gesture recognizer for 21-point hand model.

    Parameters
    ----------
    finger_tip_indices : Dict[str, int]
        Mapping from finger names to fingertip indices in the 21-point model.
        Default follows mediapipe / inspire model:
        {thumb:4, index:8/9, middle:12/14, ...}
    """

    def __init__(self,
                 finger_tip_indices: Dict[str, int] = None,
                 pos_coeff: float = 1.5,
                 neg_coeff: float = 0.8):
        # Default indices (consistent with existing code)
        default_indices = {
            "thumb": 4, "index": 9, "middle": 14,
            "ring": 19, "pinky": 24
        }
        self.tip_idx = finger_tip_indices or default_indices
        self.pos_coeff = pos_coeff  # Positive pinch distance coefficient
        self.neg_coeff = neg_coeff  # Negative/exclusion distance coefficient

    # ------------------- Public API -------------------
    def detect(self, hand_mat: np.ndarray) -> Dict[Pinch, bool]:
        """
        Detect pinch gestures for a single hand.

        Parameters
        ----------
        hand_mat : (25, 3) ndarray
            21/25-point hand keypoint coordinates, order consistent with tip_idx.

        Returns
        -------
        gestures : Dict[Pinch, bool]
            Trigger status of each pinch gesture.
        """
        # Use index-middle finger distance as reference for adaptive threshold
        ref_len = np.linalg.norm(hand_mat[1] - hand_mat[2])
        pos_th = 0.1 if ref_len == 0 else ref_len / self.pos_coeff
        neg_th = 0.2 if ref_len == 0 else ref_len / self.neg_coeff

        # Utility function
        def _close(a, b, th):  # a, b are finger name strings
            idx_a, idx_b = self.tip_idx[a], self.tip_idx[b]
            return np.linalg.norm(hand_mat[idx_a] - hand_mat[idx_b]) < th

        # Iterate through pinch combinations to detect
        results = {}
        for finger in ("index", "middle", "ring", "pinky"):
            pinch = getattr(Pinch, f"THUMB_{finger.upper()}")
            # ① thumb and finger must be close together
            if not _close("thumb", finger, pos_th):
                results[pinch] = False
                continue
            # ② Other fingertips must be "open" (avoid false triggers)
            # ---- Condition 2: Thumb must be far enough from "non-target" fingers ----
            # Find the 3 fingers other than thumb and target finger (finger_name)
            other_fingers = [
                name for name in self.tip_idx
                if name not in ("thumb", finger)
            ]

            # Check each finger: if thumb is too close to any other finger (<neg_th), consider as failure
            thumb_far_from_all_others = True
            for other in other_fingers:
                if _close("thumb", other, neg_th):  # distance < neg_th => too close
                    thumb_far_from_all_others = False
                    break

            # Only when both conditions are satisfied, the current pinch gesture is recognized
            results[pinch] = thumb_far_from_all_others
        return results

    # ------------- Can also directly judge a specific gesture -------------
    def is_pinch(self, hand_mat: np.ndarray, pinch: Pinch) -> bool:
        return self.detect(hand_mat)[pinch]
