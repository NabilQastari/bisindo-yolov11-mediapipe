"""
train.py: Fine-Tuning YOLOv11n for BISINDO Sign Detection.

Runs transfer learning using the prepared BISINDO dataset.
Optimized for CPU / AMD iGPU with batch size and workers tuned for responsiveness.
"""

import sys
import argparse
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO


def train_yolov11(
    data_yaml: str = str(PROJECT_ROOT / "data" / "bisindo_yolo" / "data.yaml"),
    epochs: int = 30,
    batch_size: int = 8,
    img_size: int = 640,
    device: str = "cpu",
    output_dir: str = str(PROJECT_ROOT / "runs" / "train")
):
    print("=== Training YOLOv11n for BISINDO ===")
    print(f"Data YAML: {data_yaml}")
    print(f"Epochs: {epochs} | Batch: {batch_size} | Image Size: {img_size} | Device: {device}")

    # Base weights path on Drive E:
    base_weights = PROJECT_ROOT / "weights" / "yolo11n.pt"
    model = YOLO(str(base_weights) if base_weights.exists() else "yolo11n.pt")

    # Run training
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device=device,
        project=output_dir,
        name="bisindo_yolov11n",
        exist_ok=True,
        workers=2,
        plots=True,
        verbose=True
    )

    print("\nTraining completed!")
    best_weights = Path(output_dir) / "bisindo_yolov11n" / "weights" / "best.pt"
    if best_weights.exists():
        dest = Path(r"E:\Semester 5\C&V\Project Citra\weights\bisindo_yolov11n_best.pt")
        dest.parent.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy(str(best_weights), str(dest))
        print(f"Best model saved to: {dest}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv11 on BISINDO dataset")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs (default: 30)")
    parser.add_argument("--batch", type=int, default=8, help="Batch size (default: 8)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size (default: 640)")
    args = parser.parse_args()

    train_yolov11(epochs=args.epochs, batch_size=args.batch, img_size=args.imgsz)
