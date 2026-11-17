"""Rigid transforms and pinhole camera maths. Quaternions are (w, x, y, z)."""

from __future__ import annotations

import numpy as np


def quat_to_matrix(q) -> np.ndarray:
    w, x, y, z = np.asarray(q, dtype=float) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def pose_to_matrix(pos, quat_wxyz) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = quat_to_matrix(quat_wxyz)
    T[:3, 3] = pos
    return T


def transform_points(T: np.ndarray, pts) -> np.ndarray:
    pts = np.atleast_2d(np.asarray(pts, float))
    return pts @ T[:3, :3].T + T[:3, 3]


class Camera:
    """Pinhole camera in the ROS optical convention (x right, y down, z forward).

    K: 3x3 intrinsics; T_world_cam: 4x4 pose of the camera in the world/base frame.
    """

    def __init__(self, K, T_world_cam, width: int, height: int):
        self.K = np.asarray(K, float).reshape(3, 3)
        self.T = np.asarray(T_world_cam, float)
        self.width, self.height = int(width), int(height)

    @property
    def R(self):
        return self.T[:3, :3]

    def project(self, p_world) -> np.ndarray:
        """World points (N,3) -> pixels (N,2) and depth (N,)."""
        pc = transform_points(np.linalg.inv(self.T), p_world)
        uv = (pc @ self.K.T)[:, :2] / pc[:, 2:3]
        return uv, pc[:, 2]

    def deproject(self, u: float, v: float, z: float) -> np.ndarray:
        """Pixel + depth (distance to the image plane) -> world point (3,)."""
        fx, fy, cx, cy = self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]
        pc = np.array([(u - cx) * z / fx, (v - cy) * z / fy, z])
        return transform_points(self.T, pc)[0]

    def image_direction(self, name: str) -> np.ndarray:
        """Unit world vector for an image-space direction (left/right/up/down)."""
        x, y = self.R[:, 0], self.R[:, 1]
        return {"right": x, "left": -x, "down": y, "up": -y}[name] / 1.0

    def scaled(self, width: int, height: int) -> "Camera":
        """Same pose/optics at another resolution."""
        S = np.diag([width / self.width, height / self.height, 1.0])
        return Camera(S @ self.K, self.T, width, height)


def snap_to_mask(mask: np.ndarray, u: float, v: float, radius: float):
    """Nearest True pixel of mask to (u, v) within radius; None if none."""
    vs, us = np.nonzero(mask)
    if len(us) == 0:
        return None
    d2 = (us - u) ** 2 + (vs - v) ** 2
    i = int(np.argmin(d2))
    return (int(us[i]), int(vs[i])) if d2[i] <= radius ** 2 else None


def median_depth(depth: np.ndarray, mask: np.ndarray, u: int, v: int, win: int = 4) -> float | None:
    H, W = depth.shape
    v0, v1, u0, u1 = max(0, v - win), min(H, v + win + 1), max(0, u - win), min(W, u + win + 1)
    d = depth[v0:v1, u0:u1][mask[v0:v1, u0:u1]]
    d = d[np.isfinite(d)]
    return float(np.median(d)) if d.size else None
