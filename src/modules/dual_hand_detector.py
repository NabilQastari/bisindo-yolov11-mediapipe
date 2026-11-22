"""
dual_hand_detector.py: MediaPipe Tasks Dual-Hand Keypoint Extractor for BISINDO.

Uses the official MediaPipe Tasks HandLandmarker API (compatible with Python 3.12+).
Tracks up to 2 hands simultaneously, providing 42 total 3D landmarks (x, y, z)
and mapping them reliably to left and right hands.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


@dataclass
class HandDetectionResult:
    has_left: bool = False
    has_right: bool = False
    landmarks_left: Optional[List[Tuple[float, float, float]]] = None
    landmarks_right: Optional[List[Tuple[float, float, float]]] = None
    wrist_left: Optional[Tuple[float, float]] = None
    wrist_right: Optional[Tuple[float, float]] = None
    scores: Dict[str, float] = field(default_factory=dict)


class DualHandDetector:
    DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent.parent / "weights" / "hand_landmarker.task"

    def __init__(
        self,
        model_path: Optional[str] = None,
        static_image_mode: bool = False,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.25,
        min_presence_confidence: float = 0.25,
        min_tracking_confidence: float = 0.3
    ):
        target_model = Path(model_path) if model_path else self.DEFAULT_MODEL_PATH
        if not target_model.exists():
            raise FileNotFoundError(f"MediaPipe task model not found at {target_model}")

        self.static_mode = static_image_mode
        running_mode = vision.RunningMode.IMAGE if static_image_mode else vision.RunningMode.VIDEO

        base_options = python.BaseOptions(model_asset_path=str(target_model))
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=max_num_hands,
            running_mode=running_mode,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        self._frame_timestamp_ms = 0

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        timestamp_ms: Optional[int] = None
    ) -> HandDetectionResult:
        """
        Processes a single BGR frame and extracts dual-hand 3D landmarks.
        """
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        if self.static_mode:
            results = self.landmarker.detect(mp_image)
        else:
            if timestamp_ms is None:
                self._frame_timestamp_ms += 33  # ~30 FPS step
                timestamp_ms = self._frame_timestamp_ms
            else:
                self._frame_timestamp_ms = timestamp_ms
            results = self.landmarker.detect_for_video(mp_image, timestamp_ms)

        out = HandDetectionResult()

        if not results.hand_landmarks:
            return out

        for hand_lms, hand_info in zip(results.hand_landmarks, results.handedness):
            label = hand_info[0].category_name  # 'Left' or 'Right'
            score = hand_info[0].score

            # Convert 21 normalized landmarks to list of tuples (x, y, z)
            lms_tuples = [(lm.x, lm.y, lm.z) for lm in hand_lms]
            wrist_coords = (hand_lms[0].x, hand_lms[0].y)

            if label == "Left":
                out.has_left = True
                out.landmarks_left = lms_tuples
                out.wrist_left = wrist_coords
                out.scores["Left"] = score
            elif label == "Right":
                out.has_right = True
                out.landmarks_right = lms_tuples
                out.wrist_right = wrist_coords
                out.scores["Right"] = score
            else:
                # Fallback if label is ambiguous
                if not out.has_right:
                    out.has_right = True
                    out.landmarks_right = lms_tuples
                    out.wrist_right = wrist_coords
                    out.scores["Right"] = score
                else:
                    out.has_left = True
                    out.landmarks_left = lms_tuples
                    out.wrist_left = wrist_coords
                    out.scores["Left"] = score

        return out

    def draw_landmarks(
        self,
        frame_bgr: np.ndarray,
        result: HandDetectionResult
    ) -> np.ndarray:
        """
        Draws landmarks on the frame with distinct colors for left and right hands.
        """
        annotated = frame_bgr.copy()
        h, w, _ = frame_bgr.shape

        for is_present, lms, color in [
            (result.has_left, result.landmarks_left, (0, 255, 0)),     # Green for Left
            (result.has_right, result.landmarks_right, (0, 165, 255)) # Orange for Right
        ]:
            if is_present and lms:
                for pt in lms:
                    cx, cy = int(pt[0] * w), int(pt[1] * h)
                    cv2.circle(annotated, (cx, cy), 3, color, -1)

                wrist_pt = lms[0]
                wx, wy = int(wrist_pt[0] * w), int(wrist_pt[1] * h)
                cv2.circle(annotated, (wx, wy), 6, (0, 0, 255), -1)

        return annotated

    def close(self):
        """Releases MediaPipe resources."""
        self.landmarker.close()
