"""
pipeline.py: Multi-Source Real-Time BISINDO Hybrid Ensemble Pipeline.

Supports multiple input sources:
- Live Webcam (auto-detection of available camera indices)
- Video File (.mp4, .avi, .mkv, .mov)
- Static Image or Image Directory (.jpg, .png, etc.)

Integrates:
1. MediaPipe Tasks HandLandmarker (42 Keypoints)
2. Velocity-Aware Transition Filter (Anti-Ghost Recognition)
3. Adaptive 1:1 Square Unified Dual-Hand ROI (Zero-distortion & clean crop)
4. Hybrid Ensemble: YOLOv11n-cls (0.60) + 42-Keypoint Landmark MLP (0.40)
5. Temporal Probability Smoothing + Authentic BISINDO Geometric Disambiguation
"""

import sys
import os
import time
import argparse
from collections import deque
from pathlib import Path
from typing import Optional, List, Union, Dict
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.modules.dual_hand_detector import DualHandDetector
from src.modules.velocity_filter import VelocityAwareFilter, TransitionState
from src.modules.landmark_classifier import LandmarkClassifierWrapper
from src.utils.box_utils import compute_square_unified_roi, crop_square_roi
from src.utils.geometry import disambiguate_bisindo_pair
from src.utils.camera_gui import launch_input_selector_gui

BISINDO_CLASSES = [chr(i) for i in range(ord('A'), ord('Z') + 1)]


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
                return int(cam_idx)
            except ValueError:
                return cameras[0]
        elif choice == "3":
            path = input("Masukkan path lengkap file video (contoh: E:\\...\\video.mp4): ").strip().strip('"')
            if os.path.exists(path):
                return path
            print(f"File tidak ditemukan: {path}. Silakan coba lagi.")
        elif choice == "4":
            path = input("Masukkan path file citra atau folder citra: ").strip().strip('"')
            if os.path.exists(path):
                return path
            print(f"Path tidak ditemukan: {path}. Silakan coba lagi.")
        else:
            print("Pilihan tidak valid. Silakan ketik angka 1, 2, 3, atau 4.")


class BisindoPipeline:
    def __init__(
        self,
        yolo_model_path: Optional[str] = None,
        landmark_weights_path: Optional[str] = None,
        velocity_threshold: float = 0.40,
        stability_min_frames: int = 3,
        static_image_mode: bool = False
    ):
        print("\n[Inisialisasi Pipeline BISINDO Hybrid Ensemble]")
        self.detector = DualHandDetector(
            static_image_mode=static_image_mode,
            max_num_hands=2,
            min_detection_confidence=0.25,
            min_presence_confidence=0.25,
            min_tracking_confidence=0.25
        )

        self.velocity_filter = VelocityAwareFilter(
            velocity_threshold=velocity_threshold,
            stability_min_frames=stability_min_frames,
            smoothing_alpha=0.65
        )

        # Load YOLOv11n-cls (preferred) or fallback to legacy detection weights
        self.yolo_model = None
        self.is_cls_model = False
        chosen_yolo = yolo_model_path
        if not chosen_yolo:
            cls_default = PROJECT_ROOT / "weights" / "bisindo_yolov11n_cls_best.pt"
            det_default = PROJECT_ROOT / "weights" / "bisindo_yolov11n_best.pt"
            if cls_default.exists():
                chosen_yolo = str(cls_default)
            elif det_default.exists():
                chosen_yolo = str(det_default)

        if chosen_yolo and Path(chosen_yolo).exists():
            from ultralytics import YOLO
            print(f"Memuat bobot YOLOv11 dari {chosen_yolo}...")
            self.yolo_model = YOLO(chosen_yolo)
            self.is_cls_model = (getattr(self.yolo_model, "task", "") == "classify")
            mode_lbl = "Classification (YOLOv11n-cls)" if self.is_cls_model else "Detection (YOLOv11n)"
            print(f"Model YOLOv11 siap digunakan [{mode_lbl}].")
        else:
            print("PERINGATAN: Bobot YOLOv11 belum ditemukan.")

        # Load 42-keypoint Landmark MLP
        self.landmark_clf: Optional[LandmarkClassifierWrapper] = None
        lm_path = Path(landmark_weights_path) if landmark_weights_path else LandmarkClassifierWrapper.DEFAULT_WEIGHTS
        if lm_path.exists():
            print(f"Memuat bobot Landmark MLP dari {lm_path}...")
            self.landmark_clf = LandmarkClassifierWrapper(str(lm_path))
            print("Model 42-Keypoint Landmark MLP siap digunakan.")
        else:
            print("PERINGATAN: Bobot Landmark MLP belum ditemukan.")

        self.prob_buffer: deque = deque(maxlen=5)
        self.mirror_mode: bool = False
        self.recognized_letters: List[str] = []
        self.last_stable_letter: str = ""
        self.stable_letter_frames: int = 0

    def _predict_yolo_probs(self, clean_frame: np.ndarray, roi_sq: np.ndarray) -> Optional[np.ndarray]:
        """Returns 26-class probability vector ordered A..Z from YOLO model."""
        if self.yolo_model is None:
            return None

        probs_vec = np.zeros(len(BISINDO_CLASSES), dtype=np.float32)
        cls_to_idx = {c: i for i, c in enumerate(BISINDO_CLASSES)}

        if self.is_cls_model:
            res = self.yolo_model.predict(roi_sq, imgsz=224, verbose=False)
            raw_probs = res[0].probs.data.cpu().numpy()
            names = self.yolo_model.names
            for k, name in names.items():
                if name in cls_to_idx:
                    probs_vec[cls_to_idx[name]] = float(raw_probs[int(k)])
            return probs_vec

        # Fallback if legacy detection weights are passed
        res = self.yolo_model.predict(clean_frame, verbose=False, conf=0.20)
        if len(res[0].boxes) > 0:
            for b in res[0].boxes:
                name = self.yolo_model.names[int(b.cls[0])]
                c = float(b.conf[0])
                if name in cls_to_idx:
                    probs_vec[cls_to_idx[name]] = max(probs_vec[cls_to_idx[name]], c)
            s = float(probs_vec.sum())
            if s > 0:
                probs_vec /= s
                return probs_vec
        return None

    def process_and_annotate_frame(
        self,
        frame: np.ndarray,
        fps: float,
        source_label: str = "",
        bypass_velocity_filter: bool = False
    ) -> np.ndarray:
        """
        Processes a clean frame through MediaPipe, Velocity Filter, Square Unified ROI,
        and Hybrid Ensemble (YOLOv11n-cls + Landmark MLP) and renders HUD on a separate copy.
        """
        clean_frame = frame.copy()
        h, w = clean_frame.shape[:2]

        # 1. MediaPipe 42 Keypoints on clean frame
        res = self.detector.process_frame(clean_frame)

        # 2. Velocity Filter
        if bypass_velocity_filter:
            has_hand = (res.has_left or res.has_right)
            state = TransitionState.STABLE if has_hand else TransitionState.ABSENT
            is_stable = has_hand
            speed = 0.0
        else:
            vel_res = self.velocity_filter.update(res.wrist_left, res.wrist_right)
            state = vel_res['state']
            is_stable = vel_res['is_stable']
            speed = vel_res['max_speed']

        # 3. 1:1 Square Unified ROI
        roi_info = compute_square_unified_roi(
            res.landmarks_left, res.landmarks_right, w, h,
            padding_ratio=0.35, interaction_threshold=0.75
        )

        # Draw landmarks ONLY on annotated copy
        annotated = self.detector.draw_landmarks(clean_frame, res)

        active_pred = "-"
        top3_str = "-"
        disambig_note: Optional[str] = None
        roi_sq_preview: Optional[np.ndarray] = None

        if is_stable and roi_info['square_box'] is not None:
            for box in roi_info['boxes']:
                cv2.rectangle(annotated, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

            # Crop exact clean square ROI (zero drawings inside!)
            roi_sq = crop_square_roi(clean_frame, roi_info['square_box'], out_size=224)
            roi_sq_preview = roi_sq

            p_yolo = self._predict_yolo_probs(clean_frame, roi_sq)
            p_lm = self.landmark_clf.predict_proba(res.landmarks_left, res.landmarks_right) if self.landmark_clf else None

            if p_yolo is not None and p_lm is not None:
                p_fused = 0.60 * p_yolo + 0.40 * p_lm
            elif p_yolo is not None:
                p_fused = p_yolo
            elif p_lm is not None:
                p_fused = p_lm
            else:
                p_fused = None

            if p_fused is not None:
                self.prob_buffer.append(p_fused)
                p_smooth = np.mean(self.prob_buffer, axis=0)
                prob_dict: Dict[str, float] = {BISINDO_CLASSES[i]: float(p_smooth[i]) for i in range(len(BISINDO_CLASSES))}

                final_letter, final_conf, disambig_note = disambiguate_bisindo_pair(
                    prob_dict, res.landmarks_left, res.landmarks_right
                )

                top3 = sorted(prob_dict.items(), key=lambda kv: kv[1], reverse=True)[:3]
                top3_str = " | ".join(f"{c}:{p*100:.0f}%" for c, p in top3)

                if final_conf >= 0.35:
                    active_pred = f"{final_letter} ({final_conf*100:.0f}%)"
                    if final_letter == self.last_stable_letter:
                        self.stable_letter_frames += 1
                        if self.stable_letter_frames == 8:
                            self.recognized_letters.append(final_letter)
                    else:
                        self.last_stable_letter = final_letter
                        self.stable_letter_frames = 1
                else:
                    active_pred = f"Kurang Jelas ({final_letter} {final_conf*100:.0f}%)"
            else:
                active_pred = "SIAP (Model Belum Dimuat)"

        elif state == TransitionState.TRANSITION:
            self.prob_buffer.clear()
            for box in roi_info['boxes']:
                cv2.rectangle(annotated, (box[0], box[1]), (box[2], box[3]), (0, 165, 255), 1)
            active_pred = "TRANSISI..."
        else:
            self.prob_buffer.clear()
            active_pred = "MENUNGGU TANGAN"

        # 4. Header Bar
        cv2.rectangle(annotated, (0, 0), (w, 65), (25, 25, 25), -1)
        mirror_lbl = " [MIRROR]" if self.mirror_mode else ""
        cv2.putText(
            annotated, f"BISINDO Hybrid (YOLOv11-cls + Landmark MLP) | {source_label}{mirror_lbl}",
            (15, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2
        )
        cv2.putText(
            annotated, f"FPS: {fps:.1f}",
            (w - 130, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2
        )

        state_color = (0, 255, 0) if is_stable else ((0, 165, 255) if state == TransitionState.TRANSITION else (160, 160, 160))
        cv2.putText(
            annotated, f"Status: {state.value} (Speed: {speed:.2f}) | ROI Mode: {roi_info['mode']}",
            (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.48, state_color, 1
        )

        # 5. Picture-in-Picture Model ROI Preview (Top-Right below header)
        if roi_sq_preview is not None and w >= 320 and h >= 260:
            pip_size = 140
            pip = cv2.resize(roi_sq_preview, (pip_size, pip_size), interpolation=cv2.INTER_AREA)
            px2, py1 = w - 15, 75
            px1, py2 = px2 - pip_size, py1 + pip_size
            annotated[py1:py2, px1:px2] = pip
            cv2.rectangle(annotated, (px1, py1), (px2, py2), (0, 255, 0), 2)
            cv2.rectangle(annotated, (px1, py2), (px2, py2 + 20), (25, 25, 25), -1)
            cv2.putText(annotated, "Model ROI", (px1 + 26, py2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 0), 1)

        # 6. Bottom HUD Sentence & Top-3 Box
        hud_top = max(70, h - 108)
        cv2.rectangle(annotated, (15, hud_top), (w - 15, h - 12), (20, 20, 20), -1)
        cv2.rectangle(annotated, (15, hud_top), (w - 15, h - 12), (80, 80, 80), 1)

        pred_line = f"Prediksi: {active_pred}"
        if disambig_note:
            pred_line += f"  [{disambig_note}]"
        cv2.putText(
            annotated, pred_line,
            (28, hud_top + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (0, 255, 255), 2
        )
        cv2.putText(
            annotated, f"Top-3: {top3_str}",
            (28, hud_top + 52), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1
        )
        sentence = "".join(self.recognized_letters[-28:])
        cv2.putText(
            annotated, f"Kalimat: {sentence}",
            (28, hud_top + 76), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1
        )
        cv2.putText(
            annotated, "[q] Keluar | [m] Mirror | [Spasi] Spasi/Jeda | [b] Hapus 1 | [c] Clear | [s] Foto",
            (28, hud_top + 93), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (150, 150, 150), 1
        )

        return annotated

    def run(self, source: Union[int, str]):
        """Runs the pipeline on the specified source."""
        is_image = False
        image_files: List[Path] = []
        image_idx = 0

        if isinstance(source, str):
            p = Path(source)
            if p.is_dir():
                image_files = sorted([f for f in p.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']])
                if not image_files:
                    print(f"Tidak ada file citra di {source}")
                    return
                is_image = True
            elif p.is_file() and p.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                image_files = [p]
                is_image = True

        if is_image:
            print(f"\nMenjalankan mode Gambar ({len(image_files)} gambar ditemukan).")
            print("Kontrol: [n] Berikutnya | [p] Sebelumnya | [s] Simpan | [q] Keluar\n")
            while True:
                cur_img_path = image_files[image_idx]
                frame = cv2.imread(str(cur_img_path))
                if frame is None:
                    image_idx = (image_idx + 1) % len(image_files)
                    continue

                self.prob_buffer.clear()
                source_lbl = f"Citra {image_idx + 1}/{len(image_files)}: {cur_img_path.name}"
                annotated = self.process_and_annotate_frame(
                    frame, fps=0.0, source_label=source_lbl, bypass_velocity_filter=True
                )
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

        is_webcam = isinstance(source, int)
        if is_webcam:
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
        print("Kontrol: [q] Keluar | [m] Mirror | [Spasi] Spasi/Jeda | [b] Hapus 1 | [c] Clear | [s] Foto\n")

        prev_time = time.perf_counter()
        fps = 0.0
        paused = False
        annotated = None

        try:
            while True:
                if not paused:
                    ret, frame = cap.read()
                    if not ret:
                        if not is_webcam:
                            print("Video selesai. Mengulang kembali dari awal...")
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            continue
                        break

                    if is_webcam and self.mirror_mode:
                        frame = cv2.flip(frame, 1)

                    curr_time = time.perf_counter()
                    fps = 0.9 * fps + 0.1 * (1.0 / max(1e-4, curr_time - prev_time))
                    prev_time = curr_time

                    annotated = self.process_and_annotate_frame(frame, fps=fps, source_label=source_lbl)
                    cv2.imshow("BISINDO Real-Time Translator", annotated)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('m'):
                    self.mirror_mode = not self.mirror_mode
                    self.prob_buffer.clear()
                    self.velocity_filter.reset()
                    print(f"Mirror mode: {'ON' if self.mirror_mode else 'OFF'}")
                elif key == ord('b'):
                    if self.recognized_letters:
                        self.recognized_letters.pop()
                elif key == ord('c'):
                    self.recognized_letters.clear()
                    self.last_stable_letter = ""
                    print("Buffer teks dikosongkan.")
                elif key == ord(' '):
                    if is_webcam:
                        self.recognized_letters.append(" ")
                    else:
                        paused = not paused
                        print("Jeda" if paused else "Lanjut")
                elif key == ord('s') and annotated is not None:
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
    parser.add_argument("--landmark-model", type=str, default=None, help="Path to trained Landmark MLP .pt model")
    parser.add_argument("--mirror", action="store_true", help="Enable horizontal mirror flip for webcam")
    parser.add_argument("--cli", action="store_true", help="Use text CLI menu instead of GUI selector")
    args = parser.parse_args()

    chosen_source = args.source
    initial_mirror = args.mirror

    if chosen_source is None:
        if args.cli:
            chosen_source = interactive_select_source()
        else:
            gui_res = launch_input_selector_gui(
                allow_files=True,
                title="BISINDO Translator - Pilih Input Kamera & Sumber",
                action_text="Mulai Penerjemah BISINDO"
            )
            if gui_res is None:
                print("Pemilihan input dibatalkan oleh pengguna.")
                sys.exit(0)
            chosen_source = gui_res["source"]
            initial_mirror = initial_mirror or gui_res["mirror"]
    elif chosen_source.isdigit():
        chosen_source = int(chosen_source)

    is_static = isinstance(chosen_source, str) and Path(chosen_source).exists() and (
        Path(chosen_source).is_dir() or Path(chosen_source).suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']
    )
    pipeline = BisindoPipeline(
        yolo_model_path=args.model,
        landmark_weights_path=args.landmark_model,
        static_image_mode=is_static
    )
    pipeline.mirror_mode = initial_mirror
    pipeline.run(source=chosen_source)
