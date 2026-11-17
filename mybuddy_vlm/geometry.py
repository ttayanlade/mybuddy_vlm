"""Camera geometry: intrinsics, 2D->3D deprojection, rigid transforms."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ----------------------------------------------------------------------------
# Rigid transforms. Quaternions are (w, x, y, z) everywhere in this project,
# matching Isaac Lab's convention.
# ----------------------------------------------------------------------------
def quat_to_matrix(q) -> np.ndarray:
    w, x, y, z = np.asarray(q, dtype=float) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def matrix_to_quat(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=float)
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    else:
        i = int(np.argmax(np.diag(R)))
        if i == 0:
            s = np.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
        elif i == 1:
            s = np.sqrt(1 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
        else:
            s = np.sqrt(1 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    q = np.array(q)
    return q / np.linalg.norm(q) * (1 if q[0] >= 0 else -1)


def pose_to_matrix(pos, quat_wxyz) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = quat_to_matrix(quat_wxyz)
    T[:3, 3] = pos
    return T


def transform_points(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.atleast_2d(pts)
    return pts @ T[:3, :3].T + T[:3, 3]


# ----------------------------------------------------------------------------
# Pinhole camera
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1.0]])

    @classmethod
    def from_K(cls, K, width: int, height: int) -> "CameraIntrinsics":
        K = np.asarray(K, dtype=float).reshape(3, 3)
        return cls(K[0, 0], K[1, 1], K[0, 2], K[1, 2], width, height)


def deproject_pixel(u: float, v: float, z: float, K: CameraIntrinsics) -> np.ndarray:
    """Pixel (u, v) with metric depth z -> 3D point in the camera optical frame
    (ROS convention: x right, y down, z forward)."""
    return np.array([(u - K.cx) * z / K.fx, (v - K.cy) * z / K.fy, z])


def project_point(p_cam: np.ndarray, K: CameraIntrinsics) -> np.ndarray:
    x, y, z = p_cam
    return np.array([K.fx * x / z + K.cx, K.fy * y / z + K.cy])


def depth_to_meters(depth: np.ndarray, depth_scale: float) -> np.ndarray:
    """uint16 (real D435, mm) or float32 (sim, m) -> float32 metres; invalid -> NaN."""
    d = depth.astype(np.float32)
    if np.issubdtype(depth.dtype, np.integer):
        d = d * depth_scale
    d[~np.isfinite(d) | (d <= 0)] = np.nan
    return d


def deproject_mask(mask: np.ndarray, depth_m: np.ndarray, K: CameraIntrinsics,
                   stride: int = 2) -> np.ndarray:
    """All valid mask pixels -> Nx3 point cloud in the camera frame."""
    vs, us = np.nonzero(mask[::stride, ::stride])
    vs, us = vs * stride, us * stride
    z = depth_m[vs, us]
    ok = np.isfinite(z)
    us, vs, z = us[ok], vs[ok], z[ok]
    return np.stack([(us - K.cx) * z / K.fx, (vs - K.cy) * z / K.fy, z], axis=1)


def mask_center_pixel(mask: np.ndarray) -> tuple[int, int]:
    """Centroid of the mask; snapped to the nearest in-mask pixel so non-convex
    shapes (mug handle, draped towel) still give a point *on* the object."""
    vs, us = np.nonzero(mask)
    if len(us) == 0:
        raise ValueError("empty mask")
    uc, vc = us.mean(), vs.mean()
    if mask[int(round(vc)), int(round(uc))]:
        return int(round(uc)), int(round(vc))
    i = int(np.argmin((us - uc) ** 2 + (vs - vc) ** 2))
    return int(us[i]), int(vs[i])


def robust_depth_at(depth_m: np.ndarray, mask: np.ndarray, u: int, v: int, win: int) -> float:
    """Median of valid in-mask depths in a (2win+1)^2 window; widens until found.
    Handles D435 holes and the occasional sim dropout pixel."""
    H, W = depth_m.shape
    for w in (win, 2 * win, 4 * win, 8 * win):
        v0, v1, u0, u1 = max(0, v - w), min(H, v + w + 1), max(0, u - w), min(W, u + w + 1)
        patch = depth_m[v0:v1, u0:u1][mask[v0:v1, u0:u1]]
        patch = patch[np.isfinite(patch)]
        if patch.size >= 3:
            return float(np.median(patch))
    z = depth_m[mask]
    z = z[np.isfinite(z)]
    if z.size == 0:
        raise ValueError("no valid depth inside mask")
    return float(np.median(z))
