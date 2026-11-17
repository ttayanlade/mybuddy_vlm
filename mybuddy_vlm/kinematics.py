"""MyBuddy 280 single-arm kinematics (== myCobot 280) — pure numpy.

* Forward kinematics from standard DH.
* Damped-least-squares IK for a 5-D task: tool position (3) + tool z-axis
  pointing straight down (2). Yaw about the tool axis is left free, which is
  what pushing primitives need and maximises reachability on a 6-DoF arm with
  a 28 cm reach.
* Joint-space trajectory construction from Cartesian waypoints with
  continuity checks and velocity-limited time parameterisation.

An optional MoveIt 2 backend (``ik_backend: moveit``) lives in ros_interface.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def dh_transform(d: float, a: float, alpha: float, theta: float) -> np.ndarray:
    ct, st, ca, sa = np.cos(theta), np.sin(theta), np.cos(alpha), np.sin(alpha)
    return np.array([[ct, -st * ca, st * sa, a * ct],
                     [st, ct * ca, -ct * sa, a * st],
                     [0, sa, ca, d],
                     [0, 0, 0, 1.0]])


class IKError(RuntimeError):
    pass


@dataclass
class ArmModel:
    dh: np.ndarray          # (6, 4) [d, a, alpha, theta_offset]
    lower: np.ndarray
    upper: np.ndarray
    tool_offset_z: float = 0.0

    @classmethod
    def from_cfg(cls, robot_cfg: dict) -> "ArmModel":
        return cls(np.asarray(robot_cfg["dh"], float), np.asarray(robot_cfg["joint_lower"], float),
                   np.asarray(robot_cfg["joint_upper"], float), float(robot_cfg.get("tool_offset_z", 0.0)))

    # ------------------------------------------------------------------ FK
    def fk(self, q) -> np.ndarray:
        T = np.eye(4)
        for (d, a, alpha, off), qi in zip(self.dh, q):
            T = T @ dh_transform(d, a, alpha, qi + off)
        if self.tool_offset_z:
            T = T @ np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, self.tool_offset_z], [0, 0, 0, 1.0]])
        return T

    # ------------------------------------------------------------------ IK
    def _task_error(self, q, p_des, z_des):
        T = self.fk(q)
        e_p = p_des - T[:3, 3]
        z = T[:3, 2]
        e_o = np.cross(z, z_des)          # zero when aligned (and != anti-aligned)
        if np.dot(z, z_des) < 0:          # anti-aligned: cross product vanishes, push away
            e_o = e_o + 0.5 * np.cross(z, np.array([1.0, 0, 0]))
        return np.concatenate([e_p, 0.1 * e_o])  # orientation weighted (m vs rad)

    def ik(self, p_des, q_seed, z_des=(0, 0, -1), tol=1e-3, max_iter=200,
           n_restarts=8, rng: np.random.Generator | None = None) -> np.ndarray:
        """Position + tool-axis-down IK. Raises IKError if unreachable within limits."""
        p_des, z_des = np.asarray(p_des, float), np.asarray(z_des, float)
        z_des = z_des / np.linalg.norm(z_des)
        rng = rng or np.random.default_rng(0)
        seeds = [np.asarray(q_seed, float)]
        seeds += [rng.uniform(self.lower * 0.7, self.upper * 0.7) for _ in range(n_restarts)]
        best, best_err = None, np.inf
        for q in seeds:
            q = np.clip(q.copy(), self.lower, self.upper)
            lam = 1e-2
            for _ in range(max_iter):
                e = self._task_error(q, p_des, z_des)
                err = np.linalg.norm(e[:3]) + np.linalg.norm(e[3:])
                if err < tol:
                    break
                J = self._num_jac(q, p_des, z_des, e)
                dq = J.T @ np.linalg.solve(J @ J.T + lam ** 2 * np.eye(len(e)), e)
                q = np.clip(q + np.clip(dq, -0.3, 0.3), self.lower, self.upper)
            pos_err = np.linalg.norm(self._task_error(q, p_des, z_des)[:3])
            ang_err = np.arccos(np.clip(np.dot(self.fk(q)[:3, 2], z_des), -1, 1))
            if pos_err < 3e-3 and ang_err < np.radians(5):
                # prefer the solution closest to the seed (smooth trajectories)
                score = np.linalg.norm(q - seeds[0])
                if score < best_err:
                    best, best_err = q, score
                if best_err < 0.5:  # close to seed: good enough, skip remaining restarts
                    return best
        if best is None:
            raise IKError(f"no IK solution for p={np.round(p_des, 3)} (tool down)")
        return best

    def _num_jac(self, q, p_des, z_des, e0, h=1e-6):
        # e = p_des - p(q): J_task = -de/dq so that dq = J^T(...)e reduces e.
        J = np.zeros((len(e0), len(q)))
        for i in range(len(q)):
            dq = np.zeros_like(q)
            dq[i] = h
            J[:, i] = -(self._task_error(q + dq, p_des, z_des) - e0) / h
        return J


# ----------------------------------------------------------------------------
# Trajectories
# ----------------------------------------------------------------------------
@dataclass
class CartesianWaypoint:
    pos: np.ndarray
    speed: float            # m/s to reach this waypoint from the previous one
    label: str = ""


@dataclass
class JointTrajectory:
    joint_names: list[str]
    positions: np.ndarray   # (N, 6)
    times: np.ndarray       # (N,) seconds from start, strictly increasing
    labels: list[str]


def densify(waypoints: list[CartesianWaypoint], start_pos: np.ndarray, spacing: float):
    """Linear interpolation in Cartesian space so the joint path tracks straight
    lines (important for pushes: a curved sweep can clip neighbouring clutter)."""
    out, prev = [], np.asarray(start_pos, float)
    for wp in waypoints:
        n = max(1, int(np.ceil(np.linalg.norm(wp.pos - prev) / spacing)))
        for k in range(1, n + 1):
            out.append(CartesianWaypoint(prev + (wp.pos - prev) * k / n, wp.speed, wp.label))
        prev = wp.pos
    return out


def plan_joint_trajectory(arm: ArmModel, joint_names: list[str], q_start: np.ndarray,
                          waypoints: list[CartesianWaypoint], spacing: float,
                          max_joint_vel: float, max_joint_jump: float = 0.35) -> JointTrajectory:
    start_pos = arm.fk(q_start)[:3, 3]
    dense = densify(waypoints, start_pos, spacing)
    qs, ts, labels = [np.asarray(q_start, float)], [0.0], ["start"]
    prev_p, t = start_pos, 0.0
    for i, wp in enumerate(dense):
        # first IK may jump (moving to approach pose); afterwards stay continuous
        q = arm.ik(wp.pos, qs[-1], n_restarts=8 if i == 0 else 2)
        jump = np.max(np.abs(q - qs[-1]))
        if i > 0 and jump > max_joint_jump:
            raise IKError(f"joint discontinuity {jump:.2f} rad at waypoint '{wp.label}' "
                          f"(p={np.round(wp.pos, 3)}); path crosses a singularity")
        dt = max(np.linalg.norm(wp.pos - prev_p) / max(wp.speed, 1e-3), jump / max_joint_vel, 0.02)
        t += dt
        qs.append(q)
        ts.append(t)
        labels.append(wp.label)
        prev_p = wp.pos
    return JointTrajectory(list(joint_names), np.array(qs), np.array(ts), labels)


if __name__ == "__main__":  # quick sanity print: python -m mybuddy_vlm.kinematics
    import argparse
    import sys

    from .config import load_config

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    arm = ArmModel.from_cfg(cfg["robot"])
    np.set_printoptions(precision=4, suppress=True)
    print("FK(zeros) flange position:", arm.fk(np.zeros(6))[:3, 3])
    print("FK(home)  flange position:", arm.fk(cfg["robot"]["home"])[:3, 3])
    for p in ([0.18, 0.0, 0.08], [0.22, -0.1, 0.05], [0.15, 0.12, 0.03]):
        try:
            q = arm.ik(p, cfg["robot"]["home"])
            print(f"IK{p} -> q={q}  check={arm.fk(q)[:3, 3]}")
        except IKError as e:
            print("IK failed:", e, file=sys.stderr)
