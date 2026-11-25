"""
landmark_classifier.py: Lightweight PyTorch Dual-Hand 42-Keypoint Classifier for BISINDO.

Provides lighting-, skin-tone-, and background-invariant classification from the
174-dim normalized dual-hand geometric feature vector.
"""

from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn

from src.utils.geometry import LANDMARK_FEATURE_DIM, extract_dual_hand_features


class LandmarkMLP(nn.Module):
    def __init__(self, in_dim: int = LANDMARK_FEATURE_DIM, num_classes: int = 26):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.30),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LandmarkClassifierWrapper:
    DEFAULT_WEIGHTS = Path(__file__).resolve().parent.parent.parent / "weights" / "bisindo_landmark_mlp.pt"

    def __init__(self, weights_path: Optional[str] = None):
        target = Path(weights_path) if weights_path else self.DEFAULT_WEIGHTS
        if not target.exists():
            raise FileNotFoundError(f"Landmark MLP weights not found at {target}")

        ckpt = torch.load(str(target), map_location="cpu", weights_only=False)
        self.classes: List[str] = ckpt["classes"]
        self.mean = np.asarray(ckpt["mean"], dtype=np.float32)
        self.std = np.asarray(ckpt["std"], dtype=np.float32)

        self.model = LandmarkMLP(in_dim=len(self.mean), num_classes=len(self.classes))
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()

    @torch.no_grad()
    def predict_proba(
        self,
        landmarks_left: Optional[List[Tuple[float, float, float]]],
        landmarks_right: Optional[List[Tuple[float, float, float]]]
    ) -> Optional[np.ndarray]:
        """Returns 26-class softmax probability array, or None if no hands detected."""
        if landmarks_left is None and landmarks_right is None:
            return None

        feat = extract_dual_hand_features(landmarks_left, landmarks_right)
        norm_feat = (feat - self.mean) / self.std
        tensor = torch.from_numpy(norm_feat).unsqueeze(0)
        logits = self.model(tensor)
        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        return probs
