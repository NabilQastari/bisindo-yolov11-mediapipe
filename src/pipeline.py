"""
pipeline.py: Multi-Source Real-Time BISINDO Recognition Pipeline.

Supports multiple input sources:
- Live Webcam (auto-detection of available camera indices)
- Video File (.mp4, .avi, .mkv, .mov)
- Static Image or Image Directory (.jpg, .png, etc.)

Integrates:
1. MediaPipe Tasks HandLandmarker (42 Keypoints)
2. Velocity-Aware Transition Filter (Anti-Ghost Recognition)
3. Adaptive Dual-Hand Unified Bounding Box
4. YOLOv11 Inference Head (optional)
5. 3D Joint Angle Geometric Disambiguation (M vs N, etc.)
"""

import sys
import os
import time
import argparse
from pathlib import Path

# Ensure project root directory is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Optional, List, Union
import cv2
import numpy as np

from src.modules.dual_hand_detector import DualHandDetector, HandDetectionResult
from src.modules.velocity_filter import VelocityAwareFilter, TransitionState
from src.utils.box_utils import compute_dual_hand_unified_bbox
from src.utils.geometry import count_extended_fingers


def scan_available_cameras(max_to_test: int = 4) -> List[int]:
    """Scans system for available camera indices using DirectShow on Windows."""
    available = []
    print("Memindai kamera yang terhubung ke sistem...")
    for idx in range(max_to_test):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                available.append(idx)
            cap.release()
    return available


def interactive_select_source() -> Union[int, str]:
    """Displays an interactive CLI menu to pick input source."""
    print("\n" + "=" * 58)
    print("      SISTEM PENERJEMAH BISINDO - PILIHAN INPUT")
    print("=" * 58)
    print(" [1] Kamera Webcam Default (Kamera 0)")
    print(" [2] Pindai & Pilih Indeks Kamera (Eksternal / USB / Virtual)")
    print(" [3] File Video (.mp4, .avi, .mov)")
    print(" [4] Citra Tunggal atau Folder Dataset Citra")
    print("=" * 58)

    while True:
        choice = input("Pilih mode input (1-4) [Default: 1]: ").strip()
        if choice in ["", "1"]:
            return 0
        elif choice == "2":
            cameras = scan_available_cameras()
            if not cameras:
                print("Tidak ada kamera terdeteksi! Menggunakan fallback Kamera 0.")
                return 0
            print(f"Kamera terdeteksi pada indeks: {cameras}")
            cam_idx = input(f"Ketik indeks kamera yang ingin digunakan ({cameras}): ").strip()
            try:
                val = int(cam_idx)
                return val
            except ValueError:
                return cameras[0]
        elif choice == "3":
            path = input("Masukkan path lengkap file video (contoh: E:\\...\\video.mp4): ").strip().strip('"')
            if os.path.exists(path):
                return path
            else:
                print(f"File tidak ditemukan: {path}. Silakan coba lagi.")
        elif choice == "4":
            path = input("Masukkan path file citra atau folder citra: ").strip().strip('"')
            if os.path.exists(path):
                return path
            else:
                print(f"Path tidak ditemukan: {path}. Silakan coba lagi.")
        else:
            print("Pilihan tidak valid. Silakan ketik angka 1, 2, 3, atau 4.")


class BisindoPipeline:
    def __init__(
        self,
        yolo_model_path: Optional[str] = None,
        velocity_threshold: float = 0.40,
        stability_min_frames: int = 3
    ):
        print("\n[Inisialisasi Pipeline BISINDO]")
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

        # 3. YOLOv11 Model
        self.yolo_model = None
        if yolo_model_path and Path(yolo_model_path).exists():
            from ultralytics import YOLO
            print(f"Memuat bobot YOLOv11 dari {yolo_model_path}...")
            self.yolo_model = YOLO(yolo_model_path)
            print("Model YOLOv11 siap digunakan.")
        else:
            print("Berjalan dalam mode Preview Landmark & Filter Kecepatan (bobot YOLO belum diinput).")

        self.recognized_letters: List[str] = []
        self.last_stable_letter: str = ""
        self.stable_letter_frames: int = 0

    def disambiguate_letter(self, letter: str, landmarks: List) -> str:
        """Disambiguation using 3D joint angles for confusing pairs."""
        if not landmarks:
            return letter

        analysis = count_extended_fingers(landmarks)
        ext = analysis['extended']

        if letter in ['M', 'N']:
            if ext['ring']:
                return 'M'
            else:
                return 'N'

        return letter

    def process_and_annotate_frame(
        self,
        frame: np.ndarray,
        fps: float,
        source_label: str = ""
    ) -> np.ndarray:
        h, w, _ = frame.shape

        # 1. MediaPipe 42 keypoints
        res = self.detector.process_frame(frame)

        # 2. Velocity Filter
        vel_res = self.velocity_filter.update(res.wrist_left, res.wrist_right)
        state = vel_res['state']
        is_stable = vel_res['is_stable']
        speed = vel_res['max_speed']

        # 3. Adaptive Bounding Box
        bbox_dict = compute_dual_hand_unified_bbox(
            landmarks_left=res.landmarks_left,
            landmarks_right=res.landmarks_right,
            img_w=w,
            img_h=h,
            padding_ratio=0.15,
            interaction_threshold=0.45
        )

        active_pred = "-"
        conf = 0.0

        if is_stable and bbox_dict['boxes']:
            for box in bbox_dict['boxes']:
                xmin, ymin, xmax, ymax = box
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

            if self.yolo_model is not None:
                primary_box = bbox_dict['boxes'][0]
                xmin, ymin, xmax, ymax = primary_box
                roi = frame[ymin:ymax, xmin:xmax]
                if roi.size > 0:
                    yolo_res = self.yolo_model.predict(roi, verbose=False, conf=0.35)
                    if len(yolo_res[0].boxes) > 0:
                        best = yolo_res[0].boxes[0]
                        class_id = int(best.cls[0])
                        conf = float(best.conf[0])
                        raw_letter = self.yolo_model.names[class_id]

                        primary_lms = res.landmarks_right or res.landmarks_left
                        active_pred = self.disambiguate_letter(raw_letter, primary_lms)
            else:
                active_pred = "SIAP"

            if active_pred not in ["-", "SIAP"]:
                if active_pred == self.last_stable_letter:
                    self.stable_letter_frames += 1
                    if self.stable_letter_frames == 8:
                        self.recognized_letters.append(active_pred)
                else:
                    self.last_stable_letter = active_pred
                    self.stable_letter_frames = 1
        elif state == TransitionState.TRANSITION:
            if bbox_dict['boxes']:
                for box in bbox_dict['boxes']:
                    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 165, 255), 1)
            active_pred = "TRANSISI..."
        else:
            active_pred = "MENUNGGU TANGAN"

        # 4. Visual Rendering
        annotated = self.detector.draw_landmarks(frame, res)

        # Header Info Bar
        cv2.rectangle(annotated, (0, 0), (w, 65), (25, 25, 25), -1)
        cv2.putText(
            annotated, f"BISINDO (MediaPipe + YOLOv11) | {source_label}",
            (15, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2
        )
        cv2.putText(
            annotated, f"FPS: {fps:.1f}",
            (w - 130, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2
        )

        state_color = (0, 255, 0) if is_stable else ((0, 165, 255) if state == TransitionState.TRANSITION else (128, 128, 128))
        cv2.putText(
            annotated, f"Status: {state.value} (Speed: {speed:.2f}) | Dual-Box Mode: {bbox_dict['mode']}",
            (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.50, state_color, 1
        )

        # Bottom Sentence Box
        cv2.rectangle(annotated, (15, h - 90), (w - 15, h - 15), (20, 20, 20), -1)
        cv2.rectangle(annotated, (15, h - 90), (w - 15, h - 15), (70, 70, 70), 1)
        cv2.putText(
            annotated, f"Prediksi Huruf: {active_pred}",
            (30, h - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2
        )
        sentence = "".join(self.recognized_letters[-25:])
        cv2.putText(
            annotated, f"Kalimat Terangkai: {sentence}",
            (30, h - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 1
        )

        return annotated

    def run(self, source: Union[int, str]):
        """Runs the pipeline on the specified source."""
        is_image = False
        image_files = []
        image_idx = 0

        # Check if source is image or folder
        if isinstance(source, str):
            p = Path(source)
            if p.is_dir():
                image_files = [f for f in p.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
                if not image_files:
                    print(f"Tidak ada file citra di {source}")
                    return
                is_image = True
            elif p.is_file() and p.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                image_files = [p]
                is_image = True

        if is_image:
            print(f"\nMenjalankan mode Gambar ({len(image_files)} gambar ditemukan).")
            print("Kontrol: [n] Gambar Berikutnya | [p] Gambar Sebelumnya | [s] Simpan | [q] Keluar\n")
            while True:
                cur_img_path = image_files[image_idx]
                frame = cv2.imread(str(cur_img_path))
                if frame is None:
                    image_idx = (image_idx + 1) % len(image_files)
                    continue

                source_lbl = f"Citra {image_idx + 1}/{len(image_files)}: {cur_img_path.name}"
                annotated = self.process_and_annotate_frame(frame, fps=0.0, source_label=source_lbl)
                cv2.imshow("BISINDO Translator - Citra", annotated)

                key = cv2.waitKey(0) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('n'):
                    image_idx = (image_idx + 1) % len(image_files)
                elif key == ord('p'):
                    image_idx = (image_idx - 1 + len(image_files)) % len(image_files)
                elif key == ord('s'):
                    snap_name = f"snapshot_img_{int(time.time())}.jpg"
                    cv2.imwrite(snap_name, annotated)
                    print(f"Tersimpan: {snap_name}")

            cv2.destroyAllWindows()
            self.detector.close()
            return

        # Video / Webcam Mode
        if isinstance(source, int):
            cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
            source_lbl = f"Webcam {source}"
        else:
            cap = cv2.VideoCapture(source)
            source_lbl = f"Video: {Path(source).name}"

        if not cap.isOpened():
            print(f"Error: Gagal membuka sumber input '{source}'")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        print(f"\n=== BISINDO Pipeline Berjalan ({source_lbl}) ===")
        print("Kontrol: [q] Keluar | [c] Hapus Teks | [s] Simpan Tangkapan | [Spasi] Jeda/Lanjut\n")

        prev_time = time.perf_counter()
        fps = 0.0
        paused = False

        try:
            while True:
                if not paused:
                    ret, frame = cap.read()
                    if not ret:
                        if not isinstance(source, int):
                            print("Video selesai. Mengulang kembali dari awal...")
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            continue
                        else:
                            break

                    curr_time = time.perf_counter()
                    fps = 0.9 * fps + 0.1 * (1.0 / max(1e-4, curr_time - prev_time))
                    prev_time = curr_time

                    annotated = self.process_and_annotate_frame(frame, fps=fps, source_label=source_lbl)
                    cv2.imshow("BISINDO Real-Time Translator", annotated)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('c'):
                    self.recognized_letters.clear()
                    self.last_stable_letter = ""
                    print("Buffer teks dikosongkan.")
                elif key == ord(' '):
                    paused = not paused
                    print("Jeda" if paused else "Lanjut")
                elif key == ord('s'):
                    snap_name = f"snapshot_{int(time.time())}.jpg"
                    cv2.imwrite(snap_name, annotated)
                    print(f"Tersimpan: {snap_name}")

        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.detector.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BISINDO Real-Time Multi-Source Pipeline")
    parser.add_argument("--source", type=str, default=None, help="Input source: webcam index (0, 1), video file path, or image directory")
    parser.add_argument("--model", type=str, default=None, help="Path to trained YOLOv11 .pt model")
    args = parser.parse_args()

    # Determine source
    chosen_source = args.source
    if chosen_source is None:
        chosen_source = interactive_select_source()
    else:
        # Check if source is a number (webcam index)
        if chosen_source.isdigit():
            chosen_source = int(chosen_source)

    pipeline = BisindoPipeline(yolo_model_path=args.model)
    pipeline.run(source=chosen_source)
