"""
prep_cls_dataset.py: Prepares clean 26-class image dataset for YOLOv11n-cls
and writes a deterministic split manifest shared with the Landmark MLP trainer.
"""

import sys
import json
import shutil
import random
from pathlib import Path
from typing import Dict, List
import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

BISINDO_CLASSES = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp'}


def prepare_bisindo_cls_dataset(
    base_source_dir: str = r"E:\Semester 5\C&V\Dataset\data 1\Citra BISINDO",
    custom_source_dir: str = str(PROJECT_ROOT / "data" / "custom_bisindo"),
    output_dir: str = str(PROJECT_ROOT / "data" / "bisindo_cls"),
    val_count_per_base_class: int = 2,
    target_size: int = 320,
    seed: int = 42,
    live_variants: int = 4
) -> Dict[str, any]:
    """
    Creates data/bisindo_cls/{train,val}/{A..Z} with zero train/val leakage
    and writes split_manifest.json for Landmark MLP training.
    """
    rng = random.Random(seed)
    base_path = Path(base_source_dir)
    custom_path = Path(custom_source_dir)
    out_path = Path(output_dir)

    if not base_path.exists() and not custom_path.exists():
        raise FileNotFoundError(
            f"Dataset tidak ditemukan di {base_path} maupun {custom_path}. "
            "Jalankan run_collect.bat untuk merekam sampel atau sediakan dataset dasar."
        )
    if not base_path.exists():
        print(f"[INFO] Base dataset tidak ada di {base_path}; hanya memakai custom_bisindo.")

    if out_path.exists():
        shutil.rmtree(out_path)

    for split in ("train", "val"):
        for cls_name in BISINDO_CLASSES:
            (out_path / split / cls_name).mkdir(parents=True, exist_ok=True)

    manifest = {"classes": BISINDO_CLASSES, "train": [], "val": []}
    stats = {"train_base": 0, "val_base": 0, "train_custom": 0, "val_custom": 0}

    for cls_name in BISINDO_CLASSES:
        # 1. Base dataset (12 images per class -> 10 train, 2 val)
        cls_dir = base_path / cls_name
        base_imgs = sorted([p for p in cls_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS]) if cls_dir.exists() else []
        rng.shuffle(base_imgs)

        n_val = min(val_count_per_base_class, max(1, len(base_imgs) // 5)) if len(base_imgs) >= 5 else 0
        val_imgs = base_imgs[:n_val]
        train_imgs = base_imgs[n_val:]

        for split, items, key in [("train", train_imgs, "train_base"), ("val", val_imgs, "val_base")]:
            for idx, src in enumerate(items):
                dst = out_path / split / cls_name / f"base_{idx:02d}_{src.stem}.jpg"
                img = cv2.imread(str(src))
                if img is None:
                    continue
                if target_size and (img.shape[0] != target_size or img.shape[1] != target_size):
                    img_out = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_AREA)
                else:
                    img_out = img
                cv2.imwrite(str(dst), img_out, [cv2.IMWRITE_JPEG_QUALITY, 95])
                manifest[split].append({"class": cls_name, "src": str(src), "dst": str(dst)})
                stats[key] += 1

        # 2. Optional user custom webcam calibration dataset
        cust_dir = custom_path / cls_name
        if cust_dir.exists():
            cust_imgs = sorted([p for p in cust_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS])
            rng.shuffle(cust_imgs)
            n_c_val = max(1, int(round(len(cust_imgs) * 0.15))) if len(cust_imgs) >= 5 else 0
            c_val = cust_imgs[:n_c_val]
            c_train = cust_imgs[n_c_val:]

            for split, items, key in [("train", c_train, "train_custom"), ("val", c_val, "val_custom")]:
                for idx, src in enumerate(items):
                    dst = out_path / split / cls_name / f"custom_{idx:03d}_{src.stem}.jpg"
                    img = cv2.imread(str(src))
                    if img is None:
                        continue
                    if target_size and (img.shape[0] != target_size or img.shape[1] != target_size):
                        img_out = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_AREA)
                    else:
                        img_out = img
                    cv2.imwrite(str(dst), img_out, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    manifest[split].append({"class": cls_name, "src": str(src), "dst": str(dst)})
                    stats[key] += 1

    manifest_path = out_path / "split_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    stats["train_live"] = 0
    if live_variants > 0:
        from src.dataset.live_sim import generate_live_cls_variants
        print(f"Generating live-simulated ROI variants ({live_variants} per train image)...")
        stats["train_live"] = generate_live_cls_variants(manifest["train"], out_path, live_variants, seed=seed)
        print(f"Live-simulated crops saved: {stats['train_live']}")

    total_train = stats["train_base"] + stats["train_custom"] + stats["train_live"]
    total_val = stats["val_base"] + stats["val_custom"]
    print(f"Prepared BISINDO classification dataset at: {out_path}")
    print(f"Train images: {total_train} (base={stats['train_base']}, custom={stats['train_custom']})")
    print(f"Val images:   {total_val} (base={stats['val_base']}, custom={stats['val_custom']})")
    print(f"Split manifest saved to: {manifest_path}")
    return {"stats": stats, "manifest_path": str(manifest_path)}


if __name__ == "__main__":
    prepare_bisindo_cls_dataset()
