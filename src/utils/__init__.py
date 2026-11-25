"""
Utility functions for bounding box calculations and geometric joint analysis.
"""
from .box_utils import (
    calculate_wrist_distance,
    compute_single_hand_bbox,
    compute_dual_hand_unified_bbox,
    compute_square_unified_roi,
    crop_square_roi,
    to_yolo_format,
    clip_bbox
)
from .geometry import (
    LANDMARK_FEATURE_DIM,
    calculate_angle_3d,
    get_finger_flexion_angles,
    is_finger_extended,
    count_extended_fingers,
    extract_dual_hand_features,
    disambiguate_bisindo_pair
)
from .camera_gui import probe_cameras, launch_input_selector_gui

__all__ = [
    "calculate_wrist_distance",
    "compute_single_hand_bbox",
    "compute_dual_hand_unified_bbox",
    "compute_square_unified_roi",
    "crop_square_roi",
    "to_yolo_format",
    "clip_bbox",
    "LANDMARK_FEATURE_DIM",
    "calculate_angle_3d",
    "get_finger_flexion_angles",
    "is_finger_extended",
    "count_extended_fingers",
    "extract_dual_hand_features",
    "disambiguate_bisindo_pair",
    "probe_cameras",
    "launch_input_selector_gui"
]
