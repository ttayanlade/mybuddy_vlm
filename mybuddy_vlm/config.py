"""Config loading + derived quantities shared by sim, reasoning and driver."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from .geometry import CameraIntrinsics, pose_to_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "pipeline.yaml"


def load_config(path: str | os.PathLike | None = None) -> dict[str, Any]:
    import yaml  # local import keeps the module importable without PyYAML for tests

    path = Path(path) if path else DEFAULT_CONFIG
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    cfg["_path"] = str(path)
    # Let Slurm jobs point at whichever VLM server is running without editing the file.
    for env, key in (("VLM_BASE_URL", "base_url"), ("VLM_MODEL", "model"), ("VLM_PROVIDER", "provider"),
                     ("VLM_STRUCTURED_OUTPUT", "structured_output")):
        if os.environ.get(env):
            cfg["vlm"][key] = os.environ[env]
    return cfg


def project_path(rel: str) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else PROJECT_ROOT / p


def intrinsics_from_cfg(cfg: dict) -> CameraIntrinsics:
    c = cfg["camera"]
    return CameraIntrinsics(fx=c["fx"], fy=c["fy"], cx=c["cx"], cy=c["cy"],
                            width=c["width"], height=c["height"])


def T_world_base(cfg: dict) -> np.ndarray:
    r = cfg["robot"]
    return pose_to_matrix(r["base_pos_world"], r["base_quat_world"])


def T_base_camera(cfg: dict, mode: str) -> np.ndarray:
    """4x4 transform mapping points in the camera optical frame to the robot base frame.

    ``mode`` is "sim" or "real". In sim it is derived exactly from the scene
    definition so the reasoning node and Isaac Lab can never disagree.
    """
    ext = cfg["ros"]["extrinsics"][mode]
    if ext == "from_scene":
        sc = cfg["scene"]["static_camera"]
        T_wc = pose_to_matrix(sc["pos"], sc["quat_ros"])
        return np.linalg.inv(T_world_base(cfg)) @ T_wc
    return pose_to_matrix(ext["pos"], ext["quat_ros"])
