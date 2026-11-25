"""
collector.py: Personal Webcam Calibration Data Collector for BISINDO.

Saves clean 224x224 square unified hand ROIs into data/custom_bisindo/<LETTER>/
when the user presses keys 'a'..'z' so they can personalize accuracy for
their own hands, camera, and room lighting.
"""

import sys
import time
import argparse
from pathlib import Path
import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.modules.dual_hand_detector import DualHandDetector
from src.utils.box_utils import compute_square_unified_roi, crop_square_roi
from src.utils.camera_gui import launch_input_selector_gui


def run_collector(
    camera_id: int = 0,
    output_dir: str = str(PROJECT_ROOT / "data" / "custom_bisindo"),
    burst_count: int = 15,
    initial_mirror: bool = False
):
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    detector = DualHandDetector(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.35,
        min_tracking_confidence=0.35
    )

    cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"Error: Gagal membuka kamera {camera_id}")
        detector.close()
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    mirror = initial_mirror
    capturing_letter = None
    remaining = 0
    last_cap_time = 0.0
    total_saved = 0

    print("\n=== BISINDO Personal Calibration Collector ===")
    print("Kontrol:")
    print("  [a-z] Tekan huruf apa saja untuk merekam 15 sampel pose tsb")
    print("  [m]   Toggle Mirror")
    print("  [ESC] Keluar (atau tutup jendela)")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if mirror:
                frame = cv2.flip(frame, 1)

            clean_frame = frame.copy()
            h, w = clean_frame.shape[:2]

            res = detector.process_frame(clean_frame)
            roi_info = compute_square_unified_roi(
                res.landmarks_left, res.landmarks_right, w, h,
                padding_ratio=0.35, interaction_threshold=0.75
            )

            annotated = detector.draw_landmarks(clean_frame, res)
            if roi_info["square_box"] is not None:
                for bx in roi_info["boxes"]:
                    cv2.rectangle(annotated, (bx[0], bx[1]), (bx[2], bx[3]), (0, 255, 0), 2)

                if capturing_letter is not None and remaining > 0:
                    now = time.perf_counter()
                    if now - last_cap_time >= 0.08:
                        roi_sq = crop_square_roi(clean_frame, roi_info["square_box"], out_size=224)
                        cls_dir = out_root / capturing_letter
                        cls_dir.mkdir(parents=True, exist_ok=True)
                        fname = cls_dir / f"cam_{int(time.time()*1000)}.jpg"
                        cv2.imwrite(str(fname), roi_sq, [cv2.IMWRITE_JPEG_QUALITY, 95])
                        remaining -= 1
                        total_saved += 1
                        last_cap_time = now
                        if remaining == 0:
                            print(f"Selesai merekam huruf '{capturing_letter}'! Total tersimpan: {total_saved}")
                            capturing_letter = None

            cv2.rectangle(annotated, (0, 0), (w, 60), (25, 25, 25), -1)
            status = (
                f"MEREKAM HURUF '{capturing_letter}' (sisa {remaining})"
                if capturing_letter else "Tekan huruf [a-z] untuk rekam 15 foto | [m] Mirror | [ESC] Keluar"
            )
            cv2.putText(annotated, status, (15, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
            cv2.imshow("BISINDO Collector (Custom Calibration)", annotated)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key == ord('m'):
                mirror = not mirror
            elif ord('a') <= key <= ord('z'):
                capturing_letter = chr(key).upper()
                remaining = burst_count
                last_cap_time = 0.0
                print(f"Mulai merekam {burst_count} sampel untuk huruf '{capturing_letter}'...")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BISINDO Webcam Calibration Collector")
    parser.add_argument("--cam", type=int, default=None, help="Webcam index (opens GUI if omitted)")
    parser.add_argument("--count", type=int, default=15, help="Frames per key press")
    parser.add_argument("--mirror", action="store_true", help="Enable horizontal mirror flip")
    args = parser.parse_args()

    chosen_cam = args.cam
    chosen_mirror = args.mirror
    if chosen_cam is None:
        gui_res = launch_input_selector_gui(
            allow_files=False,
            title="BISINDO Collector - Pilih Kamera Kalibrasi",
            action_text="Mulai Rekam Kalibrasi"
        )
        if gui_res is None:
            print("Pemilihan kamera dibatalkan.")
            sys.exit(0)
        chosen_cam = int(gui_res["source"])
        chosen_mirror = chosen_mirror or gui_res["mirror"]

    run_collector(camera_id=chosen_cam, burst_count=args.count, initial_mirror=chosen_mirror)
