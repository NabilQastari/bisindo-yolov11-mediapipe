"""
pipeline.py: End-to-End Real-Time BISINDO Recognition Pipeline.

Integrates:
1. MediaPipe Dual-Hand Detector (42 Keypoints)
2. Velocity-Aware Transition Filter (Anti-Ghost Recognition)
3. Adaptive Dual-Hand Unified Bounding Box
4. YOLOv11 Inference Head
5. 3D Joint Angle Geometric Disambiguation
"""

import time
import argparse
from pathlib import Path
from typing import Optional, List
import cv2
import numpy as np

from src.modules.dual_hand_detector import DualHandDetector, HandDetectionResult
from src.modules.velocity_filter import VelocityAwareFilter, TransitionState
from src.utils.box_utils import compute_dual_hand_unified_bbox
from src.utils.geometry import count_extended_fingers


class BisindoPipeline:
    def __init__(
        self,
        yolo_model_path: Optional[str] = None,
        velocity_threshold: float = 0.40,
        stability_min_frames: int = 3,
        camera_id: int = 0
    ):
        print("Initializing BISINDO Pipeline...")
        self.camera_id = camera_id

        # 1. MediaPipe Tracker
        self.detector = DualHandDetector(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.35,
            min_tracking_confidence=0.35
        )

        # 2. Velocity-Aware Filter
        self.velocity_filter = VelocityAwareFilter(
            velocity_threshold=velocity_threshold,
            stability_min_frames=stability_min_frames,
            smoothing_alpha=0.65
        )

        # 3. YOLOv11 Model (Loaded if available, otherwise runs in keypoint-only mode)
        self.yolo_model = None
        if yolo_model_path and Path(yolo_model_path).exists():
            from ultralytics import YOLO
            print(f"Loading YOLOv11 weights from {yolo_model_path}...")
            self.yolo_model = YOLO(yolo_model_path)
            print("YOLOv11 model loaded successfully.")
        else:
            print("Running in MediaPipe & Filter preview mode (no YOLO weights provided).")

        # Text & Sentence Buffer
        self.recognized_letters: List[str] = []
        self.last_stable_letter: str = ""
        self.stable_letter_frames: int = 0

    def disambiguate_letter(self, letter: str, landmarks: List) -> str:
        """
        Disambiguation using 3D joint angles for easily confused letter pairs.
        """
        if not landmarks:
            return letter

        analysis = count_extended_fingers(landmarks)
        ext = analysis['extended']

        # Disambiguate M vs N
        if letter in ['M', 'N']:
            # In BISINDO, check number of visible fingers over thumb
            if ext['ring']:
                return 'M'
            else:
                return 'N'

        return letter

    def run(self):
        """Runs the live webcam loop."""
        cap = cv2.VideoCapture(self.camera_id)
        if not cap.isOpened():
            print(f"Error: Could not open camera {self.camera_id}")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        print("\n=== BISINDO Live Pipeline Started ===")
        print("Controls: [q] Quit  |  [c] Clear Text  |  [s] Save Frame\n")

        prev_time = time.perf_counter()
        fps = 0.0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                curr_time = time.perf_counter()
                fps = 0.9 * fps + 0.1 * (1.0 / max(1e-4, curr_time - prev_time))
                prev_time = curr_time

                h, w, _ = frame.shape

                # 1. Extract 42 Keypoints
                res = self.detector.process_frame(frame)

                # 2. Update Velocity Filter
                vel_res = self.velocity_filter.update(res.wrist_left, res.wrist_right)
                state = vel_res['state']
                is_stable = vel_res['is_stable']
                speed = vel_res['max_speed']

                # 3. Compute Adaptive Bounding Box
                bbox_dict = compute_dual_hand_unified_bbox(
                    landmarks_left=res.landmarks_left,
                    landmarks_right=res.landmarks_right,
                    img_w=w,
                    img_h=h,
                    padding_ratio=0.15,
                    interaction_threshold=0.45
                )

                # 4. Inference / Classification
                active_pred = "-"
                conf = 0.0

                if is_stable and bbox_dict['boxes']:
                    for box in bbox_dict['boxes']:
                        xmin, ymin, xmax, ymax = box
                        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

                    if self.yolo_model is not None:
                        # YOLO inference on primary box
                        primary_box = bbox_dict['boxes'][0]
                        xmin, ymin, xmax, ymax = primary_box
                        roi = frame[ymin:ymax, xmin:xmax]
                        if roi.size > 0:
                            yolo_res = self.yolo_model.predict(roi, verbose=False, conf=0.4)
                            if len(yolo_res[0].boxes) > 0:
                                best = yolo_res[0].boxes[0]
                                class_id = int(best.cls[0])
                                conf = float(best.conf[0])
                                raw_letter = self.yolo_model.names[class_id]

                                # Disambiguation
                                primary_lms = res.landmarks_right or res.landmarks_left
                                active_pred = self.disambiguate_letter(raw_letter, primary_lms)
                    else:
                        active_pred = "SIAP"

                    # Letter commit logic (must remain stable for 8 frames to append)
                    if active_pred not in ["-", "SIAP"]:
                        if active_pred == self.last_stable_letter:
                            self.stable_letter_frames += 1
                            if self.stable_letter_frames == 8:
                                self.recognized_letters.append(active_pred)
                        else:
                            self.last_stable_letter = active_pred
                            self.stable_letter_frames = 1
                elif state == TransitionState.TRANSITION:
                    # Draw yellow/orange bounding box indicating transition suppression
                    if bbox_dict['boxes']:
                        for box in bbox_dict['boxes']:
                            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 165, 255), 1)
                    active_pred = "TRANSISI..."
                else:
                    active_pred = "MENUNGGU TANGAN"

                # 5. Draw UI Overlay
                annotated = self.detector.draw_landmarks(frame, res)

                # Top Status Bar
                cv2.rectangle(annotated, (0, 0), (w, 65), (30, 30, 30), -1)
                cv2.putText(
                    annotated, f"BISINDO Real-Time (MediaPipe + YOLOv11)",
                    (20, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
                )
                cv2.putText(
                    annotated, f"FPS: {fps:.1f}",
                    (w - 140, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2
                )

                # State Indicator
                state_color = (0, 255, 0) if is_stable else ((0, 165, 255) if state == TransitionState.TRANSITION else (128, 128, 128))
                cv2.putText(
                    annotated, f"Status: {state.value} (Kecepatan: {speed:.2f})",
                    (20, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.55, state_color, 1
                )

                # Output Prediction Box
                cv2.rectangle(annotated, (20, h - 90), (w - 20, h - 20), (20, 20, 20), -1)
                cv2.rectangle(annotated, (20, h - 90), (w - 20, h - 20), (80, 80, 80), 1)
                cv2.putText(
                    annotated, f"Prediksi: {active_pred}",
                    (35, h - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2
                )
                sentence = "".join(self.recognized_letters[-25:])
                cv2.putText(
                    annotated, f"Kalimat: {sentence}",
                    (35, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1
                )

                cv2.imshow("BISINDO Real-Time Translator", annotated)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('c'):
                    self.recognized_letters.clear()
                    self.last_stable_letter = ""
                    print("Sentence buffer cleared.")
                elif key == ord('s'):
                    snap_name = f"snapshot_{int(time.time())}.jpg"
                    cv2.imwrite(snap_name, annotated)
                    print(f"Saved snapshot to {snap_name}")

        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.detector.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None, help="Path to trained YOLOv11 .pt model")
    parser.add_argument("--cam", type=int, default=0, help="Webcam ID (default 0)")
    args = parser.parse_args()

    pipeline = BisindoPipeline(yolo_model_path=args.model, camera_id=args.cam)
    pipeline.run()
