"""
train_cls.py: Fine-tunes YOLOv11n-cls on cropped square BISINDO hand images
with strong lighting/color/rotation domain-generalization augmentations.
"""

import sys
import shutil
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO


def train_yolov11_cls(
    data_dir: str = str(PROJECT_ROOT / "data" / "bisindo_cls"),
    epochs: int = 35,
    batch_size: int = 16,
    img_size: int = 224,
    device: str = "cpu",
    output_dir: str = str(PROJECT_ROOT / "runs" / "classify")
):
    print("=== Training YOLOv11n-cls for BISINDO ===")
    print(f"Data Dir: {data_dir}")
    print(f"Epochs: {epochs} | Batch: {batch_size} | Image Size: {img_size} | Device: {device}")

    weights_dir = PROJECT_ROOT / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    base_weights = weights_dir / "yolo11n-cls.pt"
    model = YOLO(str(base_weights) if base_weights.exists() else "yolo11n-cls.pt")

    # If Ultralytics downloaded yolo11n-cls.pt into CWD, move it into weights/
    cwd_weights = Path.cwd() / "yolo11n-cls.pt"
    if cwd_weights.exists() and not base_weights.exists():
        shutil.move(str(cwd_weights), str(base_weights))

    results = model.train(
        data=data_dir,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        device=device,
        workers=0,
        optimizer="AdamW",
        lr0=0.002,
        weight_decay=0.0005,
        fliplr=0.0,
        flipud=0.0,
        hsv_h=0.03,
        hsv_s=0.6,
        hsv_v=0.5,
        degrees=18.0,
        translate=0.15,
        scale=0.35,
        erasing=0.15,
        project=output_dir,
        name="bisindo_yolov11n_cls",
        exist_ok=True,
        plots=True,
        verbose=True
    )

    best_weights = Path(output_dir) / "bisindo_yolov11n_cls" / "weights" / "best.pt"
    dest = weights_dir / "bisindo_yolov11n_cls_best.pt"
    if best_weights.exists():
        shutil.copy2(str(best_weights), str(dest))
        print(f"\nBest YOLOv11n-cls model saved to: {dest}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv11n-cls on BISINDO dataset")
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=224)
    args = parser.parse_args()

    train_yolov11_cls(epochs=args.epochs, batch_size=args.batch, img_size=args.imgsz)
