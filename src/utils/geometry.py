"""
geometry.py: 3D Joint Angle Calculations and Geometric Disambiguation for BISINDO.
"""

from typing import List, Tuple, Dict
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


def classify_gesture_landmarks(landmarks: List[Tuple[float, float, float]]) -> Dict[str, any]:
    """
    Classifies fundamental sign language alphabet poses based on 3D landmark geometry.
    Complements YOLOv11 by providing invariant geometric pose verification.
    """
    if not landmarks or len(landmarks) < 21:
        return {'letter': '-', 'confidence': 0.0, 'note': 'Landmark tidak lengkap'}

    analysis = count_extended_fingers(landmarks)
    ext = analysis['extended']
    total = analysis['total']

    # 1. 5 Fingers Open - Not an alphabet
    if total == 5:
        return {'letter': '-', 'confidence': 0.95, 'note': 'Tangan Terbuka (5 Jari) - Bukan Huruf'}

    # 2. Letter L: Thumb + Index extended
    if ext['thumb'] and ext['index'] and not ext['middle'] and not ext['ring'] and not ext['pinky']:
        return {'letter': 'L', 'confidence': 0.95, 'note': 'Pose Huruf L (Ibu Jari + Telunjuk)'}

    # 3. Letter V: Index + Middle extended
    if ext['index'] and ext['middle'] and not ext['ring'] and not ext['pinky'] and not ext['thumb']:
        return {'letter': 'V', 'confidence': 0.95, 'note': 'Pose Huruf V (Telunjuk + Tengah)'}

    # 4. Letter W: Index + Middle + Ring extended
    if ext['index'] and ext['middle'] and ext['ring'] and not ext['pinky'] and not ext['thumb']:
        return {'letter': 'W', 'confidence': 0.92, 'note': 'Pose Huruf W (3 Jari Tegak)'}

    # 5. Letter I: Only pinky extended
    if ext['pinky'] and not ext['index'] and not ext['middle'] and not ext['ring'] and not ext['thumb']:
        return {'letter': 'I', 'confidence': 0.95, 'note': 'Pose Huruf I (Kelingking Tegak)'}

    # 6. Letter Y: Thumb + Pinky extended (shaka sign)
    if ext['thumb'] and ext['pinky'] and not ext['index'] and not ext['middle'] and not ext['ring']:
        return {'letter': 'Y', 'confidence': 0.95, 'note': 'Pose Huruf Y (Ibu Jari + Kelingking)'}

    # 7. Letter B: 4 fingers extended, thumb folded
    if ext['index'] and ext['middle'] and ext['ring'] and ext['pinky'] and not ext['thumb']:
        return {'letter': 'B', 'confidence': 0.92, 'note': 'Pose Huruf B (4 Jari Rapat Tegak)'}

    # 8. Letter D: Index extended, others touching/folded
    if ext['index'] and not ext['middle'] and not ext['ring'] and not ext['pinky']:
        return {'letter': 'D', 'confidence': 0.90, 'note': 'Pose Huruf D (Telunjuk Tegak)'}

    # 9. Letter A / S / E / M / N (All fingers folded into fist)
    if total == 0:
        return {'letter': 'A', 'confidence': 0.85, 'note': 'Pose Kepalan Tangan (A/S)'}

    # 10. Letter C: Check curvature between thumb tip and index tip
    thumb_tip = np.array(landmarks[4])
    index_tip = np.array(landmarks[8])
    tip_dist = np.linalg.norm(thumb_tip - index_tip)
    if 0.08 < tip_dist < 0.22 and not ext['pinky']:
        return {'letter': 'C', 'confidence': 0.80, 'note': 'Pose Huruf C (Lengkung Telapak)'}

    return {'letter': '-', 'confidence': 0.0, 'note': 'Mencari kecocokan alfabet...'}

