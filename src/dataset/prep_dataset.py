"""
prep_dataset.py: Preprocessing & Auto-Annotation Generator for BISINDO using MediaPipe.

Generates YOLOv11 training dataset from raw BISINDO images:
1. Applies MediaPipe Dual-Hand detection to find 42 keypoints.
2. Derives Adaptive Unified Bounding Boxes.
3. Formats labels to YOLO .txt (class_idx x_center y_center width height).
4. Splits dataset into train (80%), val (10%), and test (10%).
5. Generates data.yaml for Ultralytics YOLOv11 training.
"""

import os
import shutil
import random
from pathlib import Path
from typing import Dict, List, Tuple
import cv2
from tqdm import tqdm
import yaml

from src.modules.dual_hand_detector import DualHandDetector
from src.utils.box_utils import compute_dual_hand_unified_bbox, to_yolo_format

# Standard 26 classes of BISINDO alphabets
BISINDO_CLASSES = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
CLASS_TO_ID = {cls_name: idx for idx, cls_name in enumerate(BISINDO_CLASSES)}


def prepare_bisindo_yolo_dataset(
    source_dir: str = r"E:\Semester 5\C&V\Dataset\data 1\Citra BISINDO",
    output_dir: str = r"E:\Semester 5\C&V\Project Citra\data\bisindo_yolo",
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
    seed: int = 42
) -> Dict[str, any]:
    """
    Processes raw alphabet folders (A-Z) and prepares YOLO-formatted dataset.
    """
    random.seed(seed)
    src_path = Path(source_dir)
    out_path = Path(output_dir)

    if not src_path.exists():
        raise FileNotFoundError(f"Source dataset directory not found: {source_dir}")

    # Create destination directories
    for split in ['train', 'val', 'test']:
        (out_path / 'images' / split).mkdir(parents=True, exist_ok=True)
        (out_path / 'labels' / split).mkdir(parents=True, exist_ok=True)

    print(f"Initializing MediaPipe Hands Detector...")
    detector = DualHandDetector(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.4
    )

    stats = {
        'total_images': 0,
        'successful_annotations': 0,
        'fallback_annotations': 0,
        'splits': {'train': 0, 'val': 0, 'test': 0}
    }

    print(f"\nProcessing 26 BISINDO classes from {src_path}...")

    for class_name in BISINDO_CLASSES:
        class_dir = src_path / class_name
        if not class_dir.exists():
            print(f"Warning: Directory for class '{class_name}' does not exist at {class_dir}")
            continue

        class_id = CLASS_TO_ID[class_name]
        image_files = [f for f in class_dir.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
        random.shuffle(image_files)

        n_total = len(image_files)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        for idx, img_path in enumerate(image_files):
            stats['total_images'] += 1

            # Determine split
            if idx < n_train:
                split = 'train'
            elif idx < n_train + n_val:
                split = 'val'
            else:
                split = 'test'

            stats['splits'][split] += 1

            # Read image
            frame_bgr = cv2.imread(str(img_path))
            if frame_bgr is None:
                continue

            h, w, _ = frame_bgr.shape

            # Detect hands
            res = detector.process_frame(frame_bgr)
            bbox_dict = compute_dual_hand_unified_bbox(
                landmarks_left=res.landmarks_left,
                landmarks_right=res.landmarks_right,
                img_w=w,
                img_h=h,
                padding_ratio=0.18,
                interaction_threshold=0.45
            )

            # Generate YOLO bounding box lines
            label_lines = []
            if bbox_dict['boxes']:
                stats['successful_annotations'] += 1
                for box in bbox_dict['boxes']:
                    x_c, y_c, bw, bh = to_yolo_format(box, w, h)
                    label_lines.append(f"{class_id} {x_c:.6f} {y_c:.6f} {bw:.6f} {bh:.6f}\n")
            else:
                # Fallback: Center crop estimate if detection missed on static image
                stats['fallback_annotations'] += 1
                pad = 0.05
                x_c, y_c, bw, bh = 0.5, 0.5, 0.85, 0.85
                label_lines.append(f"{class_id} {x_c:.6f} {y_c:.6f} {bw:.6f} {bh:.6f}\n")

            # Unique base name
            base_name = f"{class_name}_{img_path.stem}"
            dest_img = out_path / 'images' / split / f"{base_name}.jpg"
            dest_txt = out_path / 'labels' / split / f"{base_name}.txt"

            # Save converted/copied image and label
            cv2.imwrite(str(dest_img), frame_bgr)
            with open(dest_txt, 'w', encoding='utf-8') as f:
                f.writelines(label_lines)

    detector.close()

    # Generate data.yaml for Ultralytics YOLO
    yaml_content = {
        'path': str(out_path.resolve()),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': len(BISINDO_CLASSES),
        'names': BISINDO_CLASSES
    }

    yaml_path = out_path / 'data.yaml'
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(yaml_content, f, default_flow_style=False, sort_keys=False)

    print("\nDataset preparation completed successfully!")
    print(f"Stats: {stats}")
    print(f"data.yaml written to: {yaml_path}")

    return stats


if __name__ == "__main__":
    prepare_bisindo_yolo_dataset()
