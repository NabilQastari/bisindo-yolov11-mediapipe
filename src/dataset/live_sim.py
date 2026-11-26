"""
live_sim.py: Simulates real webcam conditions on static dataset images so models
learn the SAME input distribution they face at live-inference time.

Variations: horizontal flip (mirror / handedness swap), hand placed far from camera
inside a 16:9 frame, brightness / color-gain shifts, Gaussian / motion blur,
low-resolution upscaling and sensor noise. Hands are then re-detected by MediaPipe
and cropped with the production square-ROI function (exactly like pipeline.py).
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.modules.dual_hand_detector import DualHandDetector, HandDetectionResult
from src.utils.box_utils import compute_square_unified_roi, crop_square_roi


def _motion_kernel(length: int, angle_deg: float) -> np.ndarray:
    k = np.zeros((length, length), np.float32)
    c = length // 2
    dx = np.cos(np.radians(angle_deg)) * c
    dy = np.sin(np.radians(angle_deg)) * c
    cv2.line(k, (int(round(c - dx)), int(round(c - dy))), (int(round(c + dx)), int(round(c + dy))), 1.0, 1)
    s = k.sum()
    return k / s if s > 0 else k


def make_live_variant(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Returns a webcam-like variant (usually 1280x720) of a close-up dataset image."""
    out = img
    if rng.random() < 0.5:
        out = cv2.flip(out, 1)

    if rng.random() < 0.20:
        frame = out.copy()  # near / close-up framing, same as dataset
    else:
        side = int(rng.uniform(260, 640))
        border = np.concatenate([out[0], out[-1], out[:, 0], out[:, -1]]).reshape(-1, 3).mean(0)
        canvas = np.empty((720, 1280, 3), np.float32)
        canvas[:] = border
        low = rng.normal(0, 1, (9, 16, 3)).astype(np.float32)
        canvas += cv2.resize(low, (1280, 720), interpolation=cv2.INTER_CUBIC) * 10.0

        sq = cv2.resize(out, (side, side), interpolation=cv2.INTER_AREA).astype(np.float32)
        x0 = int(rng.integers(0, 1280 - side + 1))
        y0 = int(rng.integers(0, 720 - side + 1))
        placed = canvas.copy()
        placed[y0:y0 + side, x0:x0 + side] = sq
        mask = np.zeros((720, 1280), np.float32)
        mask[y0:y0 + side, x0:x0 + side] = 1.0
        mask = cv2.GaussianBlur(mask, (0, 0), max(2.0, side * 0.04))[..., None]
        frame = np.clip(canvas * (1 - mask) + placed * mask, 0, 255).astype(np.uint8)

    # Photometric changes
    alpha = rng.uniform(0.6, 1.3)
    beta = rng.uniform(-30, 35)
    gains = rng.uniform(0.9, 1.1, size=3).astype(np.float32)
    frame = np.clip(frame.astype(np.float32) * alpha * gains + beta, 0, 255).astype(np.uint8)

    # Blur (Gaussian or motion)
    r = rng.random()
    if r < 0.30:
        k = int(rng.choice([3, 5, 7]))
        frame = cv2.GaussianBlur(frame, (k, k), 0)
    elif r < 0.55:
        frame = cv2.filter2D(frame, -1, _motion_kernel(int(rng.integers(5, 16)), float(rng.uniform(0, 180))))

    # Low-resolution webcam
    if rng.random() < 0.40:
        h, w = frame.shape[:2]
        f = rng.uniform(1.5, 3.5)
        small = cv2.resize(frame, (max(8, int(w / f)), max(8, int(h / f))), interpolation=cv2.INTER_AREA)
        frame = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

    sigma = rng.uniform(0, 5)
    if sigma > 0.5:
        frame = np.clip(frame.astype(np.float32) + rng.normal(0, sigma, frame.shape), 0, 255).astype(np.uint8)
    return frame


def detect_and_crop(
    detector: DualHandDetector,
    frame: np.ndarray,
    padding: float = 0.35,
    out_size: int = 224
) -> Tuple[Optional[np.ndarray], HandDetectionResult]:
    """Production-identical ROI pipeline: MediaPipe -> square unified ROI -> 224 crop."""
    res = detector.process_frame(frame)
    if not (res.has_left or res.has_right):
        return None, res
    h, w = frame.shape[:2]
    roi = compute_square_unified_roi(res.landmarks_left, res.landmarks_right, w, h, padding, 0.75)
    return crop_square_roi(frame, roi["square_box"], out_size), res


def generate_live_cls_variants(
    manifest_train: List[Dict[str, str]],
    out_path: Path,
    variants_per_image: int = 4,
    seed: int = 7
) -> int:
    """Writes live-simulated square ROI crops into out_path/train/<class>/live_*.jpg."""
    rng = np.random.default_rng(seed)
    detector = DualHandDetector(
        static_image_mode=True, max_num_hands=2,
        min_detection_confidence=0.2, min_presence_confidence=0.2
    )
    saved = 0
    for i, item in enumerate(manifest_train):
        img = cv2.imread(item["src"])
        if img is None:
            continue
        for v in range(variants_per_image):
            frame = make_live_variant(img, rng)
            crop, _ = detect_and_crop(detector, frame, padding=float(rng.uniform(0.28, 0.45)))
            if crop is None:
                continue
            dst = out_path / "train" / item["class"] / f"live_{i:04d}_{v}.jpg"
            cv2.imwrite(str(dst), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
            saved += 1
    detector.close()
    return saved
