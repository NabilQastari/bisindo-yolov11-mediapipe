"""
train_landmarks.py: Trains the 42-Keypoint Dual-Hand Landmark MLP on the
exact same train/val split used by YOLOv11n-cls.
"""

import sys
import json
import random
from pathlib import Path
from typing import List, Tuple, Optional
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.modules.dual_hand_detector import DualHandDetector
from src.modules.landmark_classifier import LandmarkMLP
from src.utils.geometry import extract_dual_hand_features


def _augment_raw_landmarks(
    lms_left: Optional[List[Tuple[float, float, float]]],
    lms_right: Optional[List[Tuple[float, float, float]]],
    rng: np.random.Generator
) -> Tuple[Optional[List[Tuple[float, float, float]]], Optional[List[Tuple[float, float, float]]]]:
    """
    Applies realistic 3D geometric augmentations (in-plane Z-rotation +-15 deg,
    scale jitter 0.90-1.10, Gaussian noise, and optional 2-hand shift) on raw landmarks.
    """
    valid_pts = []
    if lms_left:
        valid_pts.append(np.asarray(lms_left, dtype=np.float32))
    if lms_right:
        valid_pts.append(np.asarray(lms_right, dtype=np.float32))
    all_pts = np.concatenate(valid_pts, axis=0)
    centroid = np.mean(all_pts, axis=0)

    theta = np.radians(rng.uniform(-15.0, 15.0))
    cos_t, sin_t = float(np.cos(theta)), float(np.sin(theta))
    scale = float(rng.uniform(0.90, 1.10))

    def _transform(lms, extra_shift):
        if lms is None:
            return None
        arr = np.asarray(lms, dtype=np.float32) - centroid
        arr = arr * scale
        x_rot = arr[:, 0] * cos_t - arr[:, 1] * sin_t
        y_rot = arr[:, 0] * sin_t + arr[:, 1] * cos_t
        arr[:, 0] = x_rot
        arr[:, 1] = y_rot
        arr = arr + centroid + extra_shift
        arr = arr + rng.normal(0.0, 0.012, size=arr.shape).astype(np.float32)
        return [tuple(map(float, row)) for row in arr]

    two_hands = (lms_left is not None and lms_right is not None)
    shift_l = np.zeros(3, dtype=np.float32)
    shift_r = np.zeros(3, dtype=np.float32)
    if two_hands and rng.random() < 0.15:
        shift_l[:2] = rng.uniform(-0.03, 0.03, size=2).astype(np.float32)
        shift_r[:2] = rng.uniform(-0.03, 0.03, size=2).astype(np.float32)

    return _transform(lms_left, shift_l), _transform(lms_right, shift_r)


def train_landmark_classifier(
    manifest_path: str = str(PROJECT_ROOT / "data" / "bisindo_cls" / "split_manifest.json"),
    weights_out: str = str(PROJECT_ROOT / "weights" / "bisindo_landmark_mlp.pt"),
    augment_factor: int = 25,
    epochs: int = 70,
    seed: int = 42
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    classes: List[str] = manifest["classes"]
    cls_to_idx = {c: i for i, c in enumerate(classes)}

    detector = DualHandDetector(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.20,
        min_presence_confidence=0.20
    )

    X_train_list, y_train_list = [], []
    X_val_list, y_val_list = [], []
    skipped_train, skipped_val = 0, 0

    print("=== Extracting Dual-Hand Landmarks from Manifest ===")
    for item in manifest["train"]:
        img = cv2.imread(item["src"])
        if img is None:
            continue
        res = detector.process_frame(img)
        if not res.has_left and not res.has_right:
            skipped_train += 1
            continue
        lbl = cls_to_idx[item["class"]]
        X_train_list.append(extract_dual_hand_features(res.landmarks_left, res.landmarks_right))
        y_train_list.append(lbl)
        for _ in range(augment_factor):
            aug_l, aug_r = _augment_raw_landmarks(res.landmarks_left, res.landmarks_right, rng)
            X_train_list.append(extract_dual_hand_features(aug_l, aug_r))
            y_train_list.append(lbl)

    for item in manifest["val"]:
        img = cv2.imread(item["src"])
        if img is None:
            continue
        res = detector.process_frame(img)
        if not res.has_left and not res.has_right:
            skipped_val += 1
            continue
        X_val_list.append(extract_dual_hand_features(res.landmarks_left, res.landmarks_right))
        y_val_list.append(cls_to_idx[item["class"]])

    detector.close()

    X_train = np.stack(X_train_list).astype(np.float32)
    y_train = np.array(y_train_list, dtype=np.int64)
    X_val = np.stack(X_val_list).astype(np.float32)
    y_val = np.array(y_val_list, dtype=np.int64)

    mean = X_train.mean(axis=0)
    std = np.maximum(X_train.std(axis=0), 1e-6)

    X_train_norm = (X_train - mean) / std
    X_val_norm = (X_val - mean) / std

    print(f"Train samples (with {augment_factor}x aug): {len(X_train_norm)} (skipped no-hand: {skipped_train})")
    print(f"Val samples with landmarks: {len(X_val_norm)} (skipped no-hand: {skipped_val})")

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train_norm), torch.from_numpy(y_train)),
        batch_size=64,
        shuffle=True
    )

    model = LandmarkMLP(in_dim=X_train_norm.shape[1], num_classes=len(classes))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    best_val_acc = -1.0
    best_state = None
    X_val_t = torch.from_numpy(X_val_norm)
    y_val_t = torch.from_numpy(y_val)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for bx, by in train_loader:
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(bx)
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
            preds = val_logits.argmax(dim=-1)
            val_acc = float((preds == y_val_t).float().mean().item())

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == 1 or epoch == epochs:
            print(f"Epoch {epoch:02d}/{epochs} | Train Loss: {total_loss/len(X_train_norm):.4f} | Val Acc: {val_acc*100:.2f}%")

    out_file = Path(weights_out)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": best_state,
        "classes": classes,
        "mean": mean,
        "std": std,
        "best_val_acc": best_val_acc
    }, str(out_file))

    print(f"\nBest Landmark MLP Val Accuracy (on detected val samples): {best_val_acc*100:.2f}%")
    print(f"Saved weights to: {out_file}")
    return best_val_acc


if __name__ == "__main__":
    train_landmark_classifier()
