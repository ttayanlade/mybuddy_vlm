"""Step 3 — zero-shot visual grounding (2D) and lifting to 3D in the robot base frame.

Grounding DINO (open-vocabulary boxes, via HF transformers) -> SAM 2 (box-prompted
mask) -> mask centre + depth -> deprojection with the D435 intrinsics -> camera->base
extrinsic. Model loading is lazy so this module imports without torch.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from .geometry import (CameraIntrinsics, deproject_mask, deproject_pixel, mask_center_pixel,
                       robust_depth_at, transform_points)

log = logging.getLogger(__name__)


@dataclass
class Detection2D:
    label: str
    score: float
    box_xyxy: np.ndarray          # (4,) pixels
    mask: np.ndarray | None = None  # (H, W) bool


@dataclass
class Grounded3D:
    """Everything the motion primitives need about the obstacle, in the BASE frame."""
    label: str
    score: float
    center_px: tuple[int, int]
    center_base: np.ndarray       # (3,) deprojected mask centre  (the spec'd target point)
    top_z: float                  # highest surface point (robust 95th pct)
    extent_xy: np.ndarray         # (2,) half-size along base x / y (robust)
    box_xyxy: np.ndarray
    mask: np.ndarray
    n_points: int


class GroundingModel:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.device = cfg.get("device", "cuda")
        self._dino = None
        self._sam = None

    # -- lazy loaders -------------------------------------------------------
    def _load_dino(self):
        if self._dino is None:
            import torch  # noqa: F401
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

            mid = self.cfg["detector"]
            self._dino = (AutoProcessor.from_pretrained(mid),
                          AutoModelForZeroShotObjectDetection.from_pretrained(mid).to(self.device).eval())
            log.info("Loaded Grounding DINO: %s", mid)
        return self._dino

    def _load_sam(self):
        if self._sam is None:
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            self._sam = SAM2ImagePredictor.from_pretrained(self.cfg["segmenter"], device=self.device)
            log.info("Loaded SAM 2: %s", self.cfg["segmenter"])
        return self._sam

    # -- 2D ---------------------------------------------------------------
    def detect(self, rgb: np.ndarray, phrase: str) -> list[Detection2D]:
        import torch
        from PIL import Image

        proc, model = self._load_dino()
        text = phrase.lower().strip().rstrip(".") + "."  # GDINO expects lowercase, '.'-terminated
        img = Image.fromarray(rgb[..., :3])
        inputs = proc(images=img, text=text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = model(**inputs)
        try:
            res = proc.post_process_grounded_object_detection(
                out, inputs.input_ids, threshold=self.cfg["box_threshold"],
                text_threshold=self.cfg["text_threshold"], target_sizes=[img.size[::-1]])[0]
        except TypeError:  # transformers < 4.47 used box_threshold=
            res = proc.post_process_grounded_object_detection(
                out, inputs.input_ids, box_threshold=self.cfg["box_threshold"],
                text_threshold=self.cfg["text_threshold"], target_sizes=[img.size[::-1]])[0]
        labels = res.get("text_labels", res.get("labels"))
        dets = [Detection2D(str(l), float(s), b.cpu().numpy())
                for s, l, b in zip(res["scores"], labels, res["boxes"])]
        dets.sort(key=lambda d: -d.score)
        return dets

    def segment(self, rgb: np.ndarray, det: Detection2D) -> Detection2D:
        import torch

        sam = self._load_sam()
        with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16,
                                                    enabled=self.device.startswith("cuda")):
            sam.set_image(rgb[..., :3])
            masks, scores, _ = sam.predict(box=det.box_xyxy[None], multimask_output=False)
        det.mask = masks[0].astype(bool)
        return det

    def ground_2d(self, rgb: np.ndarray, phrase: str) -> Detection2D | None:
        dets = self.detect(rgb, phrase)
        if not dets:
            return None
        return self.segment(rgb, dets[0])


# ----------------------------------------------------------------------------
# 2D -> 3D (pure numpy; unit-tested)
# ----------------------------------------------------------------------------
def lift_to_base(det: Detection2D, depth_m: np.ndarray, K: CameraIntrinsics,
                 T_base_cam: np.ndarray, depth_window_px: int = 7) -> Grounded3D:
    if det.mask is None or not det.mask.any():
        raise ValueError(f"no mask for '{det.label}'")
    u, v = mask_center_pixel(det.mask)
    z = robust_depth_at(depth_m, det.mask, u, v, depth_window_px)
    p_cam = deproject_pixel(u, v, z, K)
    center_base = transform_points(T_base_cam, p_cam)[0]

    # Full mask cloud for geometry the primitives need (height, footprint).
    cloud_base = transform_points(T_base_cam, deproject_mask(det.mask, depth_m, K))
    if len(cloud_base) >= 10:
        lo, hi = np.percentile(cloud_base[:, :2], [5, 95], axis=0)
        extent = (hi - lo) / 2.0
        top_z = float(np.percentile(cloud_base[:, 2], 95))
    else:
        extent = np.array([0.02, 0.02])
        top_z = float(center_base[2])
    return Grounded3D(det.label, det.score, (u, v), center_base, top_z, extent,
                      det.box_xyxy, det.mask, len(cloud_base))
