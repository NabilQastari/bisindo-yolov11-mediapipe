"""
velocity_filter.py: Velocity-Aware Transition Filter for Anti-Ghost Detection.

Filters out unstable transition frames when hands move between signs,
preventing spurious classification and reducing computational load on CPU.
"""

from enum import Enum
import time
from typing import Optional, Tuple, Dict, List
import numpy as np


class TransitionState(Enum):
    STABLE = "STABLE"
    TRANSITION = "TRANSITION"
    ABSENT = "ABSENT"


class VelocityAwareFilter:
    def __init__(
        self,
        velocity_threshold: float = 0.45,       # Normalized distance per second
        stability_min_frames: int = 3,          # Minimum frames required to declare STABLE
        smoothing_alpha: float = 0.6            # EMA smoothing factor (0 = full history, 1 = instant)
    ):
        """
        Args:
            velocity_threshold: Speed threshold in normalized coordinate units per second.
                                Hand moving faster than this is considered in TRANSITION.
            stability_min_frames: Number of consecutive low-speed frames before state turns STABLE.
            smoothing_alpha: Exponential smoothing factor for velocity.
        """
        self.velocity_threshold = velocity_threshold
        self.stability_min_frames = stability_min_frames
        self.smoothing_alpha = smoothing_alpha

        # History tracking: left and right wrists
        self.last_pos_left: Optional[Tuple[float, float]] = None
        self.last_pos_right: Optional[Tuple[float, float]] = None
        self.last_timestamp: Optional[float] = None

        self.smoothed_speed_left: float = 0.0
        self.smoothed_speed_right: float = 0.0

        self.consecutive_stable_frames: int = 0
        self.current_state: TransitionState = TransitionState.ABSENT

    def reset(self):
        """Resets filter history."""
        self.last_pos_left = None
        self.last_pos_right = None
        self.last_timestamp = None
        self.smoothed_speed_left = 0.0
        self.smoothed_speed_right = 0.0
        self.consecutive_stable_frames = 0
        self.current_state = TransitionState.ABSENT

    def update(
        self,
        wrist_left: Optional[Tuple[float, float]],
        wrist_right: Optional[Tuple[float, float]],
        timestamp: Optional[float] = None
    ) -> Dict[str, any]:
        """
        Updates velocity tracking with new wrist positions.

        Args:
            wrist_left: (x, y) normalized coordinates of left wrist or None.
            wrist_right: (x, y) normalized coordinates of right wrist or None.
            timestamp: Optional monotonic timestamp. If None, time.perf_counter() is used.

        Returns:
            Dictionary with:
                'state': TransitionState (STABLE, TRANSITION, or ABSENT)
                'is_stable': bool (True if YOLO inference should execute)
                'max_speed': float (highest speed between visible hands)
                'speed_left': float
                'speed_right': float
                'consecutive_stable': int
        """
        if timestamp is None:
            timestamp = time.perf_counter()

        # If neither hand is present
        if wrist_left is None and wrist_right is None:
            self.reset()
            return {
                'state': TransitionState.ABSENT,
                'is_stable': False,
                'max_speed': 0.0,
                'speed_left': 0.0,
                'speed_right': 0.0,
                'consecutive_stable': 0
            }

        # Calculate delta time
        if self.last_timestamp is None:
            dt = 1.0 / 30.0  # Default assumption ~30 FPS on first frame
        else:
            dt = max(1e-4, timestamp - self.last_timestamp)

        speed_l = 0.0
        speed_r = 0.0

        # Calculate left wrist speed
        if wrist_left is not None:
            if self.last_pos_left is not None:
                dx = wrist_left[0] - self.last_pos_left[0]
                dy = wrist_left[1] - self.last_pos_left[1]
                inst_speed_l = np.sqrt(dx * dx + dy * dy) / dt
                self.smoothed_speed_left = (
                    self.smoothing_alpha * inst_speed_l +
                    (1.0 - self.smoothing_alpha) * self.smoothed_speed_left
                )
            speed_l = self.smoothed_speed_left
            self.last_pos_left = wrist_left
        else:
            self.last_pos_left = None
            self.smoothed_speed_left = 0.0

        # Calculate right wrist speed
        if wrist_right is not None:
            if self.last_pos_right is not None:
                dx = wrist_right[0] - self.last_pos_right[0]
                dy = wrist_right[1] - self.last_pos_right[1]
                inst_speed_r = np.sqrt(dx * dx + dy * dy) / dt
                self.smoothed_speed_right = (
                    self.smoothing_alpha * inst_speed_r +
                    (1.0 - self.smoothing_alpha) * self.smoothed_speed_right
                )
            speed_r = self.smoothed_speed_right
            self.last_pos_right = wrist_right
        else:
            self.last_pos_right = None
            self.smoothed_speed_right = 0.0

        self.last_timestamp = timestamp

        # Max speed of actively tracked hands
        max_speed = max(speed_l, speed_r)

        # Check stability
        if max_speed <= self.velocity_threshold:
            self.consecutive_stable_frames += 1
        else:
            self.consecutive_stable_frames = 0

        if self.consecutive_stable_frames >= self.stability_min_frames:
            self.current_state = TransitionState.STABLE
        else:
            self.current_state = TransitionState.TRANSITION

        return {
            'state': self.current_state,
            'is_stable': (self.current_state == TransitionState.STABLE),
            'max_speed': round(float(max_speed), 4),
            'speed_left': round(float(speed_l), 4),
            'speed_right': round(float(speed_r), 4),
            'consecutive_stable': self.consecutive_stable_frames
        }
