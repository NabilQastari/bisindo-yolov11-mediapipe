"""
box_utils.py: Adaptive Bounding Box Calculations for Single and Dual Hands in BISINDO.
"""

from typing import List, Tuple, Optional, Dict
import numpy as np


def clip_bbox(
    xmin: float, ymin: float, xmax: float, ymax: float,
    max_w: float, max_h: float
) -> Tuple[float, float, float, float]:
    """Clips bounding box coordinates to image boundaries."""
    return (
        max(0.0, min(xmin, max_w)),
        max(0.0, min(ymin, max_h)),
        max(0.0, min(xmax, max_w)),
        max(0.0, min(ymax, max_h))
    )


def calculate_wrist_distance(
    wrist_left: Tuple[float, float],
    wrist_right: Tuple[float, float]
) -> float:
    """
    Computes normalized Euclidean distance between left and right wrists.
    Both inputs are (x, y) normalized coordinates in [0, 1].
    """
    dx = wrist_left[0] - wrist_right[0]
    dy = wrist_left[1] - wrist_right[1]
    return float(np.sqrt(dx * dx + dy * dy))


def compute_single_hand_bbox(
    landmarks: List[Tuple[float, float, float]],
    img_w: int,
    img_h: int,
    padding_ratio: float = 0.15
) -> Tuple[int, int, int, int]:
    """
    Computes bounding box (xmin, ymin, xmax, ymax) in integer pixel coordinates
    for a single hand (21 landmarks) with padding.
    """
    xs = [lm[0] * img_w for lm in landmarks]
    ys = [lm[1] * img_h for lm in landmarks]

    raw_xmin, raw_xmax = min(xs), max(xs)
    raw_ymin, raw_ymax = min(ys), max(ys)

    w_box = raw_xmax - raw_xmin
    h_box = raw_ymax - raw_ymin

    pad_w = w_box * padding_ratio
    pad_h = h_box * padding_ratio

    xmin, ymin, xmax, ymax = clip_bbox(
        raw_xmin - pad_w,
        raw_ymin - pad_h,
        raw_xmax + pad_w,
        raw_ymax + pad_h,
        float(img_w),
        float(img_h)
    )

    return int(round(xmin)), int(round(ymin)), int(round(xmax)), int(round(ymax))


def compute_dual_hand_unified_bbox(
    landmarks_left: Optional[List[Tuple[float, float, float]]],
    landmarks_right: Optional[List[Tuple[float, float, float]]],
    img_w: int,
    img_h: int,
    padding_ratio: float = 0.15,
    interaction_threshold: float = 0.35
) -> Dict[str, any]:
    """
    Adaptive Dual-Hand Unified Bounding Box:
    - If only one hand is visible, returns single hand bbox.
    - If both hands are visible:
        * If Euclidean wrist distance <= interaction_threshold (or bounding boxes overlap),
          creates ONE unified bounding box encapsulating all 42 landmarks.
        * If wrists are far apart (> threshold), can return individual bboxes or unified
          depending on the gesture context.

    Returns dict with:
        'mode': 'single_left', 'single_right', 'unified_dual', or 'separated_dual'
        'boxes': list of (xmin, ymin, xmax, ymax) in pixel coordinates
        'wrist_distance': float or None
    """
    if landmarks_left is None and landmarks_right is None:
        return {'mode': 'none', 'boxes': [], 'wrist_distance': None}

    if landmarks_left is not None and landmarks_right is None:
        box = compute_single_hand_bbox(landmarks_left, img_w, img_h, padding_ratio)
        return {'mode': 'single_left', 'boxes': [box], 'wrist_distance': None}

    if landmarks_left is None and landmarks_right is not None:
        box = compute_single_hand_bbox(landmarks_right, img_w, img_h, padding_ratio)
        return {'mode': 'single_right', 'boxes': [box], 'wrist_distance': None}

    # Both hands visible (42 landmarks total)
    wrist_l = (landmarks_left[0][0], landmarks_left[0][1])
    wrist_r = (landmarks_right[0][0], landmarks_right[0][1])
    distance = calculate_wrist_distance(wrist_l, wrist_r)

    # In BISINDO, 2-hand signs almost always interact or define a shared spatial context.
    # When distance is within threshold, create unified bounding box.
    if distance <= interaction_threshold:
        all_lms = landmarks_left + landmarks_right
        xs = [lm[0] * img_w for lm in all_lms]
        ys = [lm[1] * img_h for lm in all_lms]

        raw_xmin, raw_xmax = min(xs), max(xs)
        raw_ymin, raw_ymax = min(ys), max(ys)

        w_box = raw_xmax - raw_xmin
        h_box = raw_ymax - raw_ymin

        pad_w = w_box * padding_ratio
        pad_h = h_box * padding_ratio

        xmin, ymin, xmax, ymax = clip_bbox(
            raw_xmin - pad_w,
            raw_ymin - pad_h,
            raw_xmax + pad_w,
            raw_ymax + pad_h,
            float(img_w),
            float(img_h)
        )
        unified_box = (int(round(xmin)), int(round(ymin)), int(round(xmax)), int(round(ymax)))
        return {'mode': 'unified_dual', 'boxes': [unified_box], 'wrist_distance': distance}
    else:
        # Dual hands present but far apart
        box_l = compute_single_hand_bbox(landmarks_left, img_w, img_h, padding_ratio)
        box_r = compute_single_hand_bbox(landmarks_right, img_w, img_h, padding_ratio)
        return {'mode': 'separated_dual', 'boxes': [box_l, box_r], 'wrist_distance': distance}


def to_yolo_format(
    box: Tuple[int, int, int, int],
    img_w: int,
    img_h: int
) -> Tuple[float, float, float, float]:
    """
    Converts pixel (xmin, ymin, xmax, ymax) to normalized YOLO format:
    (x_center, y_center, width, height) with values in [0, 1].
    """
    xmin, ymin, xmax, ymax = box
    w = max(0, xmax - xmin)
    h = max(0, ymax - ymin)
    x_center = xmin + w / 2.0
    y_center = ymin + h / 2.0

    return (
        round(x_center / img_w, 6),
        round(y_center / img_h, 6),
        round(w / img_w, 6),
        round(h / img_h, 6)
    )
