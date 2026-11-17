"""Find the Fruit's occlusion metric, used here purely as EVALUATION (no learning).

Copied in spirit from find_the_fruit_env.py / vision_utils.py:
  * target mask = mask_size x mask_size pixel box centred on the projected target fruit
  * plant mask  = depth (distance to image plane) inside plant_depth_band
  * occluded    = |target mask AND plant mask|
  * success     = occluded <= visible_px for sustain_steps consecutive control steps
"""

from __future__ import annotations

from collections import deque

import numpy as np

from .geometry import Camera


def target_box_mask(cam: Camera, fruit_pos, mask_size: int) -> np.ndarray:
    (u, v), = cam.project(np.asarray(fruit_pos)[None])[0]
    m = np.zeros((cam.height, cam.width), bool)
    h = mask_size // 2
    x0, x1 = int(np.clip(u - h, 0, cam.width)), int(np.clip(u + h + 1, 0, cam.width))
    y0, y1 = int(np.clip(v - h, 0, cam.height)), int(np.clip(v + h + 1, 0, cam.height))
    m[y0:y1, x0:x1] = True
    return m


def plant_mask(depth: np.ndarray, band) -> np.ndarray:
    d = np.squeeze(depth)
    return np.isfinite(d) & (d >= band[0]) & (d <= band[1])


def occluded_pixels(depth: np.ndarray, target_mask: np.ndarray, band) -> int:
    return int(np.sum(target_mask & plant_mask(depth, band)))


class SuccessTracker:
    """Their sustained-visibility rule, evaluated every control step."""

    def __init__(self, visible_px: int, sustain_steps: int):
        self.visible_px, self.window = visible_px, deque(maxlen=sustain_steps)
        self.history: list[int] = []

    def update(self, occluded: int) -> bool:
        self.history.append(int(occluded))
        self.window.append(occluded <= self.visible_px)
        return self.success

    @property
    def success(self) -> bool:
        return len(self.window) == self.window.maxlen and all(self.window)
