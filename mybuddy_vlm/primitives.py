"""Step 4 — parameterised motion primitives.

Each VLM ``primitive_action`` maps to a short sequence of Cartesian waypoints
(tool pointing down) plus optional gripper events, parameterised by the 3D
grounding of the obstacle. Waypoints are checked against the MyBuddy 280
reachable workspace before anything is sent to the arm.

push_left  : start on the +Y side of the obstacle, lower slightly below its top,
             sweep along -Y (spec). push_right mirrors it.
pull       : hook the far side and drag toward the base; if the far side is out
             of reach, press on top and drag (works well for cloth/towels).
lift       : with a gripper: pinch-lift-place aside. Without: edge-flip
             (slide under the near edge, then raise) — an approximation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .grounding import Grounded3D
from .kinematics import CartesianWaypoint


class PrimitiveError(RuntimeError):
    """Raised when a primitive cannot be executed safely (e.g. out of reach)."""


@dataclass
class Step:
    kind: str                                   # "move" | "gripper"
    waypoints: list[CartesianWaypoint] = field(default_factory=list)
    gripper: float | None = None                # 0 = closed, 1 = open
    label: str = ""


@dataclass
class PrimitivePlan:
    action: str
    steps: list[Step]
    notes: str = ""


# ----------------------------------------------------------------------------
def action_direction(action: str, frame: str, T_base_cam: np.ndarray | None,
                     obj_xy: np.ndarray) -> np.ndarray:
    """Unit xy direction (base frame) in which the obstacle should move."""
    if action in ("push_left", "push_right"):
        sign = -1.0 if action == "push_left" else 1.0
        if frame == "base":
            return np.array([0.0, sign])
        if frame == "camera":
            if T_base_cam is None:
                raise ValueError("camera direction frame needs T_base_cam")
            d = T_base_cam[:3, :3] @ np.array([sign, 0.0, 0.0])  # image +x = right
            d = d[:2]
            n = np.linalg.norm(d)
            if n < 1e-3:
                raise PrimitiveError("camera x-axis is vertical; cannot define left/right on the table")
            return d / n
        raise ValueError(f"unknown direction_frame '{frame}'")
    if action == "pull":
        n = np.linalg.norm(obj_xy)
        return -obj_xy / n if n > 1e-6 else np.array([-1.0, 0.0])  # radially toward base
    raise ValueError(action)


def half_extent_along(extent_xy: np.ndarray, d: np.ndarray) -> float:
    """Half-size of an axis-aligned footprint along direction d (support function)."""
    return float(abs(extent_xy[0] * d[0]) + abs(extent_xy[1] * d[1]))


class Workspace:
    def __init__(self, cfg: dict):
        w = cfg["workspace"]
        self.lo = np.array([w["x"][0], w["y"][0], max(w["z"][0], cfg["min_z"])])
        self.hi = np.array([w["x"][1], w["y"][1], w["z"][1]])
        self.r = w["max_radius"]
        self.r_min = w.get("min_radius", 0.0)   # horizontal dead zone around the base axis

    def clip(self, p: np.ndarray) -> np.ndarray:
        q = np.clip(p, self.lo, self.hi)
        r = np.linalg.norm(q)
        if r > self.r:
            q = q * (self.r / r)
            q[2] = max(q[2], self.lo[2])
        rh = np.linalg.norm(q[:2])
        if rh < self.r_min:
            q[:2] = q[:2] / max(rh, 1e-9) * self.r_min if rh > 1e-9 else [self.r_min, 0.0]
        return q

    def contains(self, p: np.ndarray, tol: float = 1e-6) -> bool:
        return bool(np.all(p >= self.lo - tol) and np.all(p <= self.hi + tol)
                    and np.linalg.norm(p) <= self.r + tol
                    and np.linalg.norm(p[:2]) >= self.r_min - tol)

    def max_travel(self, start_xy: np.ndarray, d: np.ndarray, z: float, want: float,
                   step: float = 0.005) -> float:
        """Longest distance <= want that a straight move from start_xy along d
        stays inside the workspace."""
        s = 0.0
        while s + step <= want + 1e-9 and self.contains(np.array([*(start_xy + d * (s + step)), z])):
            s += step
        return s


# ----------------------------------------------------------------------------
class PrimitiveLibrary:
    def __init__(self, prim_cfg: dict, gripper_enabled: bool, T_base_cam: np.ndarray | None = None):
        self.c = prim_cfg
        self.ws = Workspace(prim_cfg)
        self.gripper = gripper_enabled
        self.T_base_cam = T_base_cam

    # helpers -------------------------------------------------------------
    def _wp(self, xy, z, speed, label) -> CartesianWaypoint:
        return CartesianWaypoint(np.array([xy[0], xy[1], z], float), speed, label)

    def _check(self, plan: PrimitivePlan, contact_tol: float = 0.02) -> PrimitivePlan:
        """Clip free-space waypoints into the workspace; refuse if a contact
        waypoint would have to move more than ``contact_tol`` (the push would
        no longer touch the obstacle)."""
        for st in plan.steps:
            for wp in st.waypoints:
                clipped = self.ws.clip(wp.pos)
                shift = np.linalg.norm(clipped - wp.pos)
                if shift > contact_tol and wp.label in ("contact", "sweep", "drag", "grasp", "lift"):
                    raise PrimitiveError(
                        f"{plan.action}: '{wp.label}' waypoint {np.round(wp.pos, 3)} is outside the "
                        f"MyBuddy workspace (needs {shift * 100:.1f} cm correction)")
                wp.pos = clipped
        return plan

    def _contact_z(self, g: Grounded3D) -> float:
        # "lower slightly" below the top, but never above mid-height of tall
        # objects (pushing a mug near its rim tips it over) and never into the table.
        return float(max(self.c["min_z"], min(g.top_z - self.c["contact_drop"], 0.6 * g.top_z)))

    # primitives ------------------------------------------------------------
    def plan(self, action: str, g: Grounded3D) -> PrimitivePlan:
        if action in ("push_left", "push_right"):
            return self._check(self._push(action, g))
        if action == "pull":
            return self._check(self._pull(g))
        if action == "lift":
            return self._check(self._lift(g))
        raise PrimitiveError(f"unknown primitive '{action}'")

    def _push(self, action: str, g: Grounded3D) -> PrimitivePlan:
        c = self.c
        d = action_direction(action, c["direction_frame"], self.T_base_cam, g.center_base[:2])
        half = half_extent_along(g.extent_xy, d)
        pre = g.center_base[:2] - d * (half + c["standoff"])        # behind the obstacle
        end = pre + d * (c["standoff"] + c["push_distance"])          # obstacle moves push_distance
        z_safe = g.top_z + c["approach_height"]
        zc = self._contact_z(g)
        wps = [
            self._wp(pre, z_safe, c["free_speed"], "approach"),
            self._wp(pre, zc, c["cart_speed"], "contact"),            # lower slightly on Z
            self._wp(end, zc, c["cart_speed"], "sweep"),              # move along -Y / +Y
            self._wp(end, z_safe, c["free_speed"], "retreat"),
        ]
        return PrimitivePlan(action, [Step("move", wps, label=action)],
                             notes=f"dir={np.round(d, 2)} half={half:.3f} zc={zc:.3f}")

    def _pull(self, g: Grounded3D) -> PrimitivePlan:
        c = self.c
        d = action_direction("pull", c["direction_frame"], self.T_base_cam, g.center_base[:2])
        half = half_extent_along(g.extent_xy, d)
        z_safe = g.top_z + c["approach_height"]
        far = g.center_base[:2] - d * (half + c["standoff"])          # far side (away from base)
        far3 = np.array([*far, self._contact_z(g)])
        if self.ws.contains(far3):
            zc = far3[2]
            travel = self.ws.max_travel(far, d, zc, c["standoff"] + c["pull_distance"])
            if travel < c["standoff"] + 0.02:
                raise PrimitiveError("pull: no room to drag toward the base")
            end = far + d * travel
            wps = [self._wp(far, z_safe, c["free_speed"], "approach"),
                   self._wp(far, zc, c["cart_speed"], "contact"),
                   self._wp(end, zc, c["cart_speed"], "drag"),
                   self._wp(end, z_safe, c["free_speed"], "retreat")]
            note = "hook far side"
        else:
            # press-and-drag on top: friction does the work (towels, cloth, paper)
            zc = max(c["min_z"], g.top_z - c["contact_drop"])
            start = g.center_base[:2]
            travel = self.ws.max_travel(start, d, zc, c["pull_distance"])
            if travel < 0.02:
                raise PrimitiveError("pull: object already at the inner reach limit")
            end = start + d * travel
            wps = [self._wp(start, z_safe, c["free_speed"], "approach"),
                   self._wp(start, zc, c["cart_speed"] * 0.6, "contact"),
                   self._wp(end, zc, c["cart_speed"], "drag"),
                   self._wp(end, z_safe, c["free_speed"], "retreat")]
            note = "press-and-drag (far side out of reach)"
        return PrimitivePlan("pull", [Step("move", wps, label="pull")], notes=note)

    def _lift(self, g: Grounded3D) -> PrimitivePlan:
        c = self.c
        z_safe = g.top_z + c["approach_height"]
        ctr = g.center_base[:2]
        if self.gripper:
            zg = max(c["min_z"], g.top_z - 0.01)
            # put it down on whichever side has more room inside the workspace
            side = np.array([0.0, 1.0 if ctr[1] < 0 else -1.0])
            place = ctr + side * 0.08
            steps = [
                Step("gripper", gripper=1.0, label="open"),
                Step("move", [self._wp(ctr, z_safe, c["free_speed"], "approach"),
                              self._wp(ctr, zg, c["cart_speed"], "grasp")], label="descend"),
                Step("gripper", gripper=0.0, label="close"),
                Step("move", [self._wp(ctr, zg + c["lift_height"], c["cart_speed"], "lift"),
                              self._wp(place, zg + c["lift_height"], c["free_speed"], "carry"),
                              self._wp(place, zg + 0.01, c["cart_speed"], "place")], label="lift"),
                Step("gripper", gripper=1.0, label="release"),
                Step("move", [self._wp(place, z_safe, c["free_speed"], "retreat")], label="retreat"),
            ]
            return PrimitivePlan("lift", steps, notes="pinch-lift-place")
        # No gripper: edge flip. Enter at an edge at table height, slide under/into
        # it, then raise while continuing inward so the edge folds over. Try the
        # near edge first (flip away from base), then the side edges, and use the
        # first whose entry point is reachable (the near edge of a large towel is
        # often inside the dead zone around the base).
        radial = ctr / max(np.linalg.norm(ctr), 1e-6)
        side = np.array([-radial[1], radial[0]])
        for d_in in (radial, side, -side, -radial):
            half = half_extent_along(g.extent_xy, d_in)
            near = ctr - d_in * (half + c["standoff"])
            under = ctr - d_in * max(half - 0.02, 0.0)
            flip = under + d_in * 0.04
            pts = [np.array([*near, c["min_z"]]), np.array([*under, c["min_z"]]),
                   np.array([*flip, c["min_z"] + c["lift_height"]])]
            if all(self.ws.contains(p) for p in pts):
                break
        else:
            raise PrimitiveError("lift (edge flip): no reachable edge of the obstacle")
        wps = [self._wp(near, z_safe, c["free_speed"], "approach"),
               self._wp(near, c["min_z"], c["cart_speed"], "contact"),
               self._wp(under, c["min_z"], c["cart_speed"] * 0.6, "sweep"),
               self._wp(flip, c["min_z"] + c["lift_height"], c["cart_speed"], "lift"),
               self._wp(flip, z_safe + 0.03, c["free_speed"], "retreat")]
        return PrimitivePlan("lift", [Step("move", wps, label="edge_flip")],
                             notes=f"no gripper: edge-flip along {np.round(d_in, 2)}")
