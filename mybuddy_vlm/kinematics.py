"""MyBuddy 280 single-arm kinematics — pure numpy.

* Forward kinematics straight from the URDF joint chain (default: the MyBuddy
  arm's link lengths differ from the myCobot 280's), or from a DH table.
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


def _rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def _axis_angle(axis, q):
    x, y, z = axis
    K = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(q) * K + (1 - np.cos(q)) * K @ K


def load_urdf_chain(path, base_link: str, ee_link: str, joint_names: list[str]):
    """[(T_origin 4x4, axis 3)] for the actuated joints base_link -> ee_link, with any
    fixed joints folded into the next origin; trailing fixed joints -> tool transform."""
    import xml.etree.ElementTree as ET

    root = ET.parse(path).getroot()
    by_child = {j.find("child").get("link"): j for j in root.findall("joint")}
    seq, link = [], ee_link
    while link != base_link:
        if link not in by_child:
            raise ValueError(f"{ee_link} is not below {base_link} in {path}")
        j = by_child[link]
        seq.append(j)
        link = j.find("parent").get("link")
    seq.reverse()
    chain, pending = [], np.eye(4)
    for j in seq:
        o = j.find("origin")
        T = np.eye(4)
        if o is not None:
            T[:3, :3] = _rpy(*map(float, (o.get("rpy") or "0 0 0").split()))
            T[:3, 3] = list(map(float, (o.get("xyz") or "0 0 0").split()))
        T = pending @ T
        if j.get("type") in ("revolute", "continuous"):
            ax = j.find("axis")
            axis = np.array((ax.get("xyz") if ax is not None else "1 0 0").split(), float)
            chain.append((j.get("name"), T, axis / np.linalg.norm(axis)))
            pending = np.eye(4)
        else:
            pending = T
    names = [c[0] for c in chain]
    if names != list(joint_names):
        raise ValueError(f"URDF chain joints {names} != config joint_names {joint_names}")
    return [(T, a) for _, T, a in chain], pending


@dataclass
class ArmModel:
    joints: list            # [(T_origin 4x4, axis(3))] per actuated joint
    tool: np.ndarray        # fixed 4x4 from last joint frame to the tool point
    lower: np.ndarray
    upper: np.ndarray

    @classmethod
    def from_cfg(cls, robot_cfg: dict) -> "ArmModel":
        lower = np.asarray(robot_cfg["joint_lower"], float)
        upper = np.asarray(robot_cfg["joint_upper"], float)
        tool_z = np.eye(4)
        tool_z[2, 3] = float(robot_cfg.get("tool_offset_z", 0.0))
        if robot_cfg.get("kinematics", "urdf") == "urdf":
            from .config import project_path

            joints, trailing = load_urdf_chain(project_path(robot_cfg["urdf_single_arm"]), robot_cfg["base_link"],
                                               robot_cfg["ee_link"], robot_cfg["joint_names"])
            return cls(joints, trailing @ tool_z, lower, upper)
        return cls.from_dh(np.asarray(robot_cfg["dh"], float), lower, upper, tool_z)

    @classmethod
    def from_dh(cls, dh: np.ndarray, lower, upper, tool=np.eye(4)) -> "ArmModel":
        """Standard DH [d, a, alpha, theta_offset] -> the same joint-chain form:
        T_i = Rz(q_i + off_i) * F_i, with F_i = Tz(d) Tx(a) Rx(alpha)."""
        joints, F_prev = [], np.eye(4)
        for d, a, alpha, off in dh:
            Rz = np.eye(4)
            Rz[:3, :3] = _axis_angle((0, 0, 1), off)
            joints.append((F_prev @ Rz, np.array([0.0, 0.0, 1.0])))
            F_prev = dh_transform(d, a, alpha, 0.0)
        return cls(joints, F_prev @ tool, np.asarray(lower), np.asarray(upper))

    # ------------------------------------------------------------------ FK
    def fk(self, q) -> np.ndarray:
        T = np.eye(4)
        R = np.eye(4)
        for (T_o, axis), qi in zip(self.joints, q):
            R[:3, :3] = _axis_angle(axis, qi)
            T = T @ T_o @ R
        return T @ self.tool

    # ------------------------------------------------------------------ IK
    W_ORI = 0.1  # orientation weight (m vs rad)

    def _fk_jac(self, q):
        """Tool pose + geometric Jacobian columns (world axis a_i, joint origin o_i)."""
        T = np.eye(4)
        R = np.eye(4)
        axes, origins = [], []
        for (T_o, axis), qi in zip(self.joints, q):
            T = T @ T_o
            axes.append(T[:3, :3] @ axis)
            origins.append(T[:3, 3].copy())
            R[:3, :3] = _axis_angle(axis, qi)
            T = T @ R
        T = T @ self.tool
        p, z = T[:3, 3], T[:3, 2]
        A, O = np.array(axes), np.array(origins)
        Jp = np.cross(A, p - O).T            # dp/dq   (3, n)
        Jz = np.cross(A, z).T                # dz/dq   (3, n)
        return T, Jp, Jz

    def _task(self, q, p_des, z_des):
        T, Jp, Jz = self._fk_jac(q)
        p, z = T[:3, 3], T[:3, 2]
        e_o = np.cross(z, z_des)             # zero when aligned (and when anti-aligned)
        J_o = -np.cross(Jz.T, z_des).T       # d(-e_o)/dq
        if np.dot(z, z_des) < 0:             # anti-aligned: push away from the saddle
            kick = np.array([1.0, 0, 0])
            e_o = e_o + 0.5 * np.cross(z, kick)
            J_o = J_o - 0.5 * np.cross(Jz.T, kick).T
        e = np.concatenate([p_des - p, self.W_ORI * e_o])
        J = np.vstack([Jp, self.W_ORI * J_o])  # J = -de/dq, so dq = J^T(JJ^T+λ²)^-1 e reduces e
        return e, J, z

    def _task_error(self, q, p_des, z_des):
        return self._task(q, p_des, z_des)[0]

    def ik(self, p_des, q_seed, z_des=(0, 0, -1), tol=1e-4, max_iter=150,
           n_restarts=8, rng: np.random.Generator | None = None) -> np.ndarray:
        """Position + tool-axis-down IK. Raises IKError if unreachable within limits."""
        p_des, z_des = np.asarray(p_des, float), np.asarray(z_des, float)
        z_des = z_des / np.linalg.norm(z_des)
        rng = rng or np.random.default_rng(0)
        seeds = [np.asarray(q_seed, float)]
        if n_restarts > 0:  # nearest tool-down configurations first, then random restarts
            seeds += list(self._bank_seeds(p_des, k=min(4, n_restarts)))
        seeds += [rng.uniform(self.lower * 0.9, self.upper * 0.9) for _ in range(n_restarts)]
        best, best_err = None, np.inf
        for q in seeds:
            q = np.clip(q.copy(), self.lower, self.upper)
            lam, prev = 1e-2, np.inf
            for it in range(max_iter):
                e, J, z = self._task(q, p_des, z_des)
                err = np.linalg.norm(e)
                if err < tol:
                    break
                if it > 20 and err > 0.999 * prev:   # stalled in a local minimum / at a limit
                    break
                prev = err
                dq = J.T @ np.linalg.solve(J @ J.T + lam ** 2 * np.eye(len(e)), e)
                q = np.clip(q + np.clip(dq, -0.3, 0.3), self.lower, self.upper)
            e, _, z = self._task(q, p_des, z_des)
            pos_err = np.linalg.norm(e[:3])
            ang_err = np.arccos(np.clip(np.dot(z, z_des), -1, 1))
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

    def _bank_seeds(self, p_des, k=4, n=6000):
        """Seeds from a lazily built bank of random configurations whose tool points
        roughly down, ranked by distance of their tool point to p_des. Escapes the
        local minima that DLS falls into from a single far-away seed."""
        if getattr(self, "_bank", None) is None:
            rng = np.random.default_rng(123)
            Q = rng.uniform(self.lower, self.upper, size=(n, len(self.lower)))
            keep_q, keep_p = [], []
            for q in Q:
                T = self.fk(q)
                if T[2, 2] < -0.6:
                    keep_q.append(q)
                    keep_p.append(T[:3, 3])
            object.__setattr__(self, "_bank", (np.array(keep_q), np.array(keep_p)))
        Qb, Pb = self._bank
        if len(Qb) == 0:
            return []
        idx = np.argsort(np.linalg.norm(Pb - p_des, axis=1))[:k]
        return Qb[idx]

    def _num_jac(self, q, p_des, z_des, e0, h=1e-6):
        """Finite-difference task Jacobian (kept for testing the analytic one)."""
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
