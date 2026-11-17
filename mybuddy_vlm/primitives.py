"""The one motion primitive: sweep a plant part out of the line of sight.

Given a 3D contact point P on the plant (from the VLM's pixel + depth) and a push
direction d (an image direction mapped through the camera pose), the tool
  approaches from the camera side, starts pre_dist behind P, sweeps through P to
  push_dist beyond it, and retreats toward the camera again.
Rejected (with a reason fed back to the VLM) if the path comes near a fruit or the floor.
"""

from __future__ import annotations

import numpy as np

from .kinematics import Waypoint


class PrimitiveError(RuntimeError):
    pass


def _seg_point_dist(a, b, p) -> float:
    ab = b - a
    t = np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-12), 0, 1)
    return float(np.linalg.norm(a + t * ab - p))


def sweep(P, d, view_dir, fruits, fruit_radius: float, c: dict) -> list[Waypoint]:
    P, d = np.asarray(P, float), np.asarray(d, float) / np.linalg.norm(d)
    back = -np.asarray(view_dir, float) / np.linalg.norm(view_dir)       # toward the camera / robot
    S = P - d * c["pre_dist"]
    E = P + d * c["push_dist"]
    A = S + back * c["approach_back"]
    R = E + back * c["approach_back"]
    wps = [Waypoint(A, c["free_speed"], "approach"), Waypoint(S, c["cart_speed"], "start"),
           Waypoint(E, c["cart_speed"], "sweep"), Waypoint(R, c["free_speed"], "retreat")]
    pts = [w.pos for w in wps]
    if min(p[2] for p in pts) < c["min_z"]:
        raise PrimitiveError(f"path goes below {c['min_z']} m (floor); choose a higher point or another direction")
    for f in fruits:
        gap = min(_seg_point_dist(a, b, np.asarray(f)) for a, b in zip(pts[:-1], pts[1:])) - fruit_radius
        if gap < c["fruit_clearance"]:
            raise PrimitiveError(f"path passes {max(gap, 0) * 100:.1f} cm from a fruit; touching fruit fails the "
                                 "task — choose another point or direction")
    return wps
