"""
Utility functions for bounding box calculations and geometric joint analysis.
"""
from .box_utils import (
    calculate_wrist_distance,
    compute_single_hand_bbox,
    compute_dual_hand_unified_bbox,
    to_yolo_format,
    clip_bbox
)
from .geometry import (
    calculate_angle_3d,
    get_finger_flexion_angles,
    is_finger_extended
)

__all__ = [
    "calculate_wrist_distance",
    "compute_single_hand_bbox",
    "compute_dual_hand_unified_bbox",
    "to_yolo_format",
    "clip_bbox",
    "calculate_angle_3d",
    "get_finger_flexion_angles",
    "is_finger_extended"
]
