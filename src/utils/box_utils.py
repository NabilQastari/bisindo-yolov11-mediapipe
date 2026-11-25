"""
box_utils.py: Adaptive Bounding Box Calculations for Single and Dual Hands in BISINDO.
"""

from typing import List, Tuple, Optional, Dict
import cv2
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


def _landmarks_pixel_bounds(
    landmarks: List[Tuple[float, float, float]],
    img_w: int,
    img_h: int
) -> Tuple[float, float, float, float]:
    """Returns unclipped pixel bounds (xmin, ymin, xmax, ymax) for 21 landmarks."""
    xs = [lm[0] * img_w for lm in landmarks]
    ys = [lm[1] * img_h for lm in landmarks]
    return min(xs), min(ys), max(xs), max(ys)


def _make_square_box(
    raw_xmin: float,
    raw_ymin: float,
    raw_xmax: float,
    raw_ymax: float,
    padding_ratio: float,
    min_side: float = 80.0
) -> Tuple[int, int, int, int]:
    """Converts a rectangular pixel box into a padded 1:1 square box."""
    cx = (raw_xmin + raw_xmax) * 0.5
    cy = (raw_ymin + raw_ymax) * 0.5
    side = max(raw_xmax - raw_xmin, raw_ymax - raw_ymin) * (1.0 + 2.0 * padding_ratio)
    side = max(side, min_side)
    half = side * 0.5
    return (
        int(round(cx - half)),
        int(round(cy - half)),
        int(round(cx + half)),
        int(round(cy + half))
    )


def compute_square_unified_roi(
    landmarks_left: Optional[List[Tuple[float, float, float]]],
    landmarks_right: Optional[List[Tuple[float, float, float]]],
    img_w: int,
    img_h: int,
    padding_ratio: float = 0.35,
    interaction_threshold: float = 0.75
) -> Dict[str, any]:
    """
    Computes a 1:1 square unified ROI matching the square framing of Citra BISINDO.
    - Single hand: squares the 21 landmarks with padding.
    - Dual hands: merges all 42 landmarks when wrists are within interaction_threshold
      or boxes are close; otherwise keeps a unified square over all 42 landmarks so
      2-hand BISINDO signs are never cropped off.
    """
    if landmarks_left is None and landmarks_right is None:
        return {'mode': 'none', 'square_box': None, 'boxes': [], 'wrist_distance': None}

    if landmarks_left is not None and landmarks_right is None:
        xmin, ymin, xmax, ymax = _landmarks_pixel_bounds(landmarks_left, img_w, img_h)
        sq = _make_square_box(xmin, ymin, xmax, ymax, padding_ratio)
        cx1, cy1, cx2, cy2 = clip_bbox(sq[0], sq[1], sq[2], sq[3], float(img_w), float(img_h))
        vis = (int(round(cx1)), int(round(cy1)), int(round(cx2)), int(round(cy2)))
        return {'mode': 'single_left', 'square_box': sq, 'boxes': [vis], 'wrist_distance': None}

    if landmarks_left is None and landmarks_right is not None:
        xmin, ymin, xmax, ymax = _landmarks_pixel_bounds(landmarks_right, img_w, img_h)
        sq = _make_square_box(xmin, ymin, xmax, ymax, padding_ratio)
        cx1, cy1, cx2, cy2 = clip_bbox(sq[0], sq[1], sq[2], sq[3], float(img_w), float(img_h))
        vis = (int(round(cx1)), int(round(cy1)), int(round(cx2)), int(round(cy2)))
        return {'mode': 'single_right', 'square_box': sq, 'boxes': [vis], 'wrist_distance': None}

    # Both hands visible
    wrist_l = (landmarks_left[0][0], landmarks_left[0][1])
    wrist_r = (landmarks_right[0][0], landmarks_right[0][1])
    distance = calculate_wrist_distance(wrist_l, wrist_r)

    l_xmin, l_ymin, l_xmax, l_ymax = _landmarks_pixel_bounds(landmarks_left, img_w, img_h)
    r_xmin, r_ymin, r_xmax, r_ymax = _landmarks_pixel_bounds(landmarks_right, img_w, img_h)

    gap_x = max(0.0, max(l_xmin - r_xmax, r_xmin - l_xmax)) / max(1, img_w)
    gap_y = max(0.0, max(l_ymin - r_ymax, r_ymin - l_ymax)) / max(1, img_h)
    box_gap = float(np.sqrt(gap_x * gap_x + gap_y * gap_y))

    all_xmin = min(l_xmin, r_xmin)
    all_ymin = min(l_ymin, r_ymin)
    all_xmax = max(l_xmax, r_xmax)
    all_ymax = max(l_ymax, r_ymax)
    sq = _make_square_box(all_xmin, all_ymin, all_xmax, all_ymax, padding_ratio)

    if distance <= interaction_threshold or box_gap <= 0.20:
        cx1, cy1, cx2, cy2 = clip_bbox(sq[0], sq[1], sq[2], sq[3], float(img_w), float(img_h))
        vis = (int(round(cx1)), int(round(cy1)), int(round(cx2)), int(round(cy2)))
        return {'mode': 'unified_dual', 'square_box': sq, 'boxes': [vis], 'wrist_distance': distance}
    else:
        box_l = compute_single_hand_bbox(landmarks_left, img_w, img_h, 0.18)
        box_r = compute_single_hand_bbox(landmarks_right, img_w, img_h, 0.18)
        return {'mode': 'separated_dual', 'square_box': sq, 'boxes': [box_l, box_r], 'wrist_distance': distance}


def crop_square_roi(
    frame_bgr: np.ndarray,
    square_box: Tuple[int, int, int, int],
    out_size: int = 224
) -> np.ndarray:
    """
    Extracts an exact square region (with edge replication padding if square_box
    crosses frame borders) and resizes it to (out_size, out_size).
    """
    x1, y1, x2, y2 = square_box
    h, w = frame_bgr.shape[:2]

    pad_left = max(0, -x1)
    pad_top = max(0, -y1)
    pad_right = max(0, x2 - w)
    pad_bottom = max(0, y2 - h)

    if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
        padded = cv2.copyMakeBorder(
            frame_bgr, pad_top, pad_bottom, pad_left, pad_right,
            borderType=cv2.BORDER_REPLICATE
        )
        x1 += pad_left
        x2 += pad_left
        y1 += pad_top
        y2 += pad_top
    else:
        padded = frame_bgr

    crop = padded[y1:y2, x1:x2]
    if crop.size == 0:
        return cv2.resize(frame_bgr, (out_size, out_size), interpolation=cv2.INTER_AREA)

    return cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_AREA)
