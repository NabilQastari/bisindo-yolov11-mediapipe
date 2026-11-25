"""
geometry.py: 3D Joint Angle Calculations and Geometric Disambiguation for BISINDO.
"""

from typing import List, Tuple, Dict, Optional
import numpy as np


def calculate_angle_3d(
    pointA: Tuple[float, float, float],
    pointB: Tuple[float, float, float],
    pointC: Tuple[float, float, float]
) -> float:
    """
    Calculates the 3D angle (in degrees) at pointB formed by segment (pointA - pointB)
    and segment (pointC - pointB).
    """
    v1 = np.array([pointA[0] - pointB[0], pointA[1] - pointB[1], pointA[2] - pointB[2]], dtype=np.float32)
    v2 = np.array([pointC[0] - pointB[0], pointC[1] - pointB[1], pointC[2] - pointB[2]], dtype=np.float32)

    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)

    if norm1 < 1e-6 or norm2 < 1e-6:
        return 0.0

    cosine = np.dot(v1, v2) / (norm1 * norm2)
    cosine = np.clip(cosine, -1.0, 1.0)
    angle_rad = np.arccos(cosine)
    return float(np.degrees(angle_rad))


def is_finger_extended(
    wrist: Tuple[float, float, float],
    mcp: Tuple[float, float, float],
    pip: Tuple[float, float, float],
    tip: Tuple[float, float, float],
    angle_pip: float
) -> bool:
    """
    Determines if a finger is extended based on joint angle (> 150 degrees)
    and distance from wrist.
    """
    dist_tip = np.linalg.norm(np.array(tip) - np.array(wrist))
    dist_pip = np.linalg.norm(np.array(pip) - np.array(wrist))
    dist_mcp = np.linalg.norm(np.array(mcp) - np.array(wrist))

    return (angle_pip > 140.0) and (dist_tip > dist_pip > dist_mcp)


def get_finger_flexion_angles(landmarks: List[Tuple[float, float, float]]) -> Dict[str, Dict[str, float]]:
    """
    Computes 3D flexion angles for each joint of the 5 fingers.
    Landmarks indices:
      Wrist: 0
      Thumb: 1(CMC), 2(MCP), 3(IP), 4(TIP)
      Index: 5(MCP), 6(PIP), 7(DIP), 8(TIP)
      Middle: 9(MCP), 10(PIP), 11(DIP), 12(TIP)
      Ring: 13(MCP), 14(PIP), 15(DIP), 16(TIP)
      Pinky: 17(MCP), 18(PIP), 19(DIP), 20(TIP)
    """
    angles = {}

    # Thumb
    angles['thumb'] = {
        'mcp': calculate_angle_3d(landmarks[1], landmarks[2], landmarks[3]),
        'ip': calculate_angle_3d(landmarks[2], landmarks[3], landmarks[4])
    }

    # Index
    angles['index'] = {
        'mcp': calculate_angle_3d(landmarks[0], landmarks[5], landmarks[6]),
        'pip': calculate_angle_3d(landmarks[5], landmarks[6], landmarks[7]),
        'dip': calculate_angle_3d(landmarks[6], landmarks[7], landmarks[8])
    }

    # Middle
    angles['middle'] = {
        'mcp': calculate_angle_3d(landmarks[0], landmarks[9], landmarks[10]),
        'pip': calculate_angle_3d(landmarks[9], landmarks[10], landmarks[11]),
        'dip': calculate_angle_3d(landmarks[10], landmarks[11], landmarks[12])
    }

    # Ring
    angles['ring'] = {
        'mcp': calculate_angle_3d(landmarks[0], landmarks[13], landmarks[14]),
        'pip': calculate_angle_3d(landmarks[13], landmarks[14], landmarks[15]),
        'dip': calculate_angle_3d(landmarks[14], landmarks[15], landmarks[16])
    }

    # Pinky
    angles['pinky'] = {
        'mcp': calculate_angle_3d(landmarks[0], landmarks[17], landmarks[18]),
        'pip': calculate_angle_3d(landmarks[17], landmarks[18], landmarks[19]),
        'dip': calculate_angle_3d(landmarks[18], landmarks[19], landmarks[20])
    }

    return angles


def count_extended_fingers(landmarks: List[Tuple[float, float, float]]) -> Dict[str, any]:
    """
    Counts how many fingers are extended vs curled.
    Returns status per finger and total count.
    """
    angles = get_finger_flexion_angles(landmarks)
    wrist = landmarks[0]

    # Index
    index_ext = is_finger_extended(wrist, landmarks[5], landmarks[6], landmarks[8], angles['index']['pip'])
    # Middle
    middle_ext = is_finger_extended(wrist, landmarks[9], landmarks[10], landmarks[12], angles['middle']['pip'])
    # Ring
    ring_ext = is_finger_extended(wrist, landmarks[13], landmarks[14], landmarks[16], angles['ring']['pip'])
    # Pinky
    pinky_ext = is_finger_extended(wrist, landmarks[17], landmarks[18], landmarks[20], angles['pinky']['pip'])

    # Thumb extension (distance tip to pinky MCP)
    thumb_dist = np.linalg.norm(np.array(landmarks[4]) - np.array(landmarks[17]))
    thumb_mcp_dist = np.linalg.norm(np.array(landmarks[2]) - np.array(landmarks[17]))
    thumb_ext = (thumb_dist > thumb_mcp_dist * 1.2) and (angles['thumb']['ip'] > 135.0)

    extended = {
        'thumb': thumb_ext,
        'index': index_ext,
        'middle': middle_ext,
        'ring': ring_ext,
        'pinky': pinky_ext
    }

    total_count = sum(1 for val in extended.values() if val)
    return {'extended': extended, 'total': total_count, 'angles': angles}


LANDMARK_FEATURE_DIM = 174
_FINGERTIP_INDICES = (4, 8, 12, 16, 20)


def _single_hand_features(landmarks: Optional[List[Tuple[float, float, float]]]) -> np.ndarray:
    """
    Extracts an 83-dim wrist-relative, palm-scale-normalized feature vector
    for a single hand (21 landmarks). Returns zeros if hand is absent.
    """
    if not landmarks or len(landmarks) < 21:
        return np.zeros(83, dtype=np.float32)

    pts = np.asarray(landmarks, dtype=np.float32)
    rel = pts - pts[0:1]
    palm_scale = max(float(np.linalg.norm(rel[9])), 1e-4)
    norm_pts = np.clip(rel / palm_scale, -4.0, 4.0).reshape(-1)  # 63 floats

    angles = get_finger_flexion_angles(landmarks)
    angle_vec = np.array([
        angles['thumb']['mcp'], angles['thumb']['ip'],
        angles['index']['mcp'], angles['index']['pip'], angles['index']['dip'],
        angles['middle']['mcp'], angles['middle']['pip'], angles['middle']['dip'],
        angles['ring']['mcp'], angles['ring']['pip'], angles['ring']['dip'],
        angles['pinky']['mcp'], angles['pinky']['pip'], angles['pinky']['dip'],
    ], dtype=np.float32) / 180.0  # 14 floats

    tip_dists = np.array([
        min(float(np.linalg.norm(rel[idx])) / (palm_scale * 4.0), 1.5)
        for idx in _FINGERTIP_INDICES
    ], dtype=np.float32)  # 5 floats

    presence = np.array([1.0], dtype=np.float32)  # 1 float
    return np.concatenate([norm_pts, angle_vec, tip_dists, presence])


def extract_dual_hand_features(
    landmarks_left: Optional[List[Tuple[float, float, float]]],
    landmarks_right: Optional[List[Tuple[float, float, float]]]
) -> np.ndarray:
    """
    Extracts a 174-dim dual-hand geometric feature vector:
    - 83 dims for left hand
    - 83 dims for right hand
    - 8 dims for inter-hand spatial relationships
    """
    feat_l = _single_hand_features(landmarks_left)
    feat_r = _single_hand_features(landmarks_right)

    if landmarks_left and len(landmarks_left) >= 21 and landmarks_right and len(landmarks_right) >= 21:
        pts_l = np.asarray(landmarks_left, dtype=np.float32)
        pts_r = np.asarray(landmarks_right, dtype=np.float32)
        scale_l = max(float(np.linalg.norm(pts_l[9] - pts_l[0])), 1e-4)
        scale_r = max(float(np.linalg.norm(pts_r[9] - pts_r[0])), 1e-4)
        avg_scale = 0.5 * (scale_l + scale_r)

        rel_wrist = np.clip((pts_r[0] - pts_l[0]) / avg_scale, -6.0, 6.0) / 6.0
        wrist_dist = min(float(np.linalg.norm(pts_r[0] - pts_l[0])) / avg_scale, 6.0) / 6.0

        tips_l = pts_l[list(_FINGERTIP_INDICES)]
        tips_r = pts_r[list(_FINGERTIP_INDICES)]
        pairwise = np.linalg.norm(tips_l[:, None, :] - tips_r[None, :, :], axis=-1)
        min_tip_dist = min(float(np.min(pairwise)) / avg_scale, 6.0) / 6.0

        index_tip_dist = min(float(np.linalg.norm(pts_l[8] - pts_r[8])) / avg_scale, 6.0) / 6.0
        thumb_tip_dist = min(float(np.linalg.norm(pts_l[4] - pts_r[4])) / avg_scale, 6.0) / 6.0

        inter = np.array([
            rel_wrist[0], rel_wrist[1], rel_wrist[2],
            wrist_dist, min_tip_dist, index_tip_dist, thumb_tip_dist, 1.0
        ], dtype=np.float32)
    else:
        inter = np.zeros(8, dtype=np.float32)

    return np.concatenate([feat_l, feat_r, inter])


def disambiguate_bisindo_pair(
    probs: Dict[str, float],
    landmarks_left: Optional[List[Tuple[float, float, float]]],
    landmarks_right: Optional[List[Tuple[float, float, float]]],
    margin_threshold: float = 0.25
) -> Tuple[str, float, Optional[str]]:
    """
    Authentic 2-hand BISINDO geometric disambiguation for visually ambiguous pairs
    (M vs N and B vs F) when top-2 probabilities are close.
    """
    ranked = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
    if not ranked:
        return "-", 0.0, None

    c1, p1 = ranked[0]
    if len(ranked) < 2:
        return c1, float(p1), None

    c2, p2 = ranked[1]
    if not (landmarks_left and landmarks_right) or (p1 - p2) >= margin_threshold:
        return c1, float(p1), None

    pair = {c1, c2}
    info_l = count_extended_fingers(landmarks_left)
    info_r = count_extended_fingers(landmarks_right)

    if pair == {'M', 'N'}:
        # Base hand is open palm (>= 4 extended fingers); active hand places 3 (M) or 2 (N) fingers
        if info_l['total'] >= info_r['total']:
            base_info, active_info = info_l, info_r
        else:
            base_info, active_info = info_r, info_l

        if base_info['total'] >= 4:
            ext = active_info['extended']
            if ext['ring']:
                return 'M', float(max(p1, p2)), 'GeoDisambig: 3 jari pada telapak -> M'
            if ext['index'] and ext['middle'] and not ext['ring']:
                return 'N', float(max(p1, p2)), 'GeoDisambig: 2 jari pada telapak -> N'

    elif pair == {'B', 'F'}:
        # One hand holds 1 vertical index finger; other hand places 3 (B) or 2 (F) horizontal fingers
        if info_l['total'] <= info_r['total']:
            vert_info, horiz_info = info_l, info_r
        else:
            vert_info, horiz_info = info_r, info_l

        if vert_info['total'] <= 2:
            ext = horiz_info['extended']
            if ext['ring']:
                return 'B', float(max(p1, p2)), 'GeoDisambig: 3 jari horizontal -> B'
            if ext['index'] and ext['middle'] and not ext['ring']:
                return 'F', float(max(p1, p2)), 'GeoDisambig: 2 jari horizontal -> F'

    return c1, float(p1), None

