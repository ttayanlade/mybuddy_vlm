"""MyBuddy left-arm kinematics read straight from the robot USD (pure numpy + pxr).

Each USD physics joint stores the joint frame in its parent body (localPos0/localRot0)
and in its child body (localPos1/localRot1) plus an axis, so the child pose is

    T_parent_child(q) = T(lp0, lr0) @ Rot(axis, q) @ inv(T(lp1, lr1))

Chaining these from the world-fixed root gives exact FK for the same model PhysX
simulates. IK is position-only (the contact point matters, not the wrist angle, and
the 6 controlled joints include the waist): damped least squares with an analytic
Jacobian, warm-started from the previous solution and a bank of random seeds.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class IKError(RuntimeError):
    pass


def _axis_angle(axis, q):
    x, y, z = axis
    K = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(q) * K + (1 - np.cos(q)) * K @ K


def _T(pos, quat_wxyz):
    from .geometry import pose_to_matrix

    return pose_to_matrix(pos, quat_wxyz)


def _joint_frames(prim):
    from pxr import UsdPhysics

    j = UsdPhysics.Joint(prim)
    q0, q1 = j.GetLocalRot0Attr().Get(), j.GetLocalRot1Attr().Get()
    A = _T(tuple(j.GetLocalPos0Attr().Get()), (q0.GetReal(), *q0.GetImaginary()))
    B = _T(tuple(j.GetLocalPos1Attr().Get()), (q1.GetReal(), *q1.GetImaginary()))
    axis = {"X": (1.0, 0, 0), "Y": (0, 1.0, 0), "Z": (0, 0, 1.0)}.get(
        str(prim.GetAttribute("physics:axis").Get() or "X"), (1.0, 0, 0))
    b0 = [str(t) for t in j.GetBody0Rel().GetTargets()]
    b1 = [str(t) for t in j.GetBody1Rel().GetTargets()]
    return A, B, np.array(axis), (b0[0] if b0 else None), (b1[0] if b1 else None)


def load_usd_chain(usd_path, joint_names, fixed_tail: dict | None = None, tool_offset=(0, 0, 0)):
    """Returns (joints=[(T_origin, axis)], tool 4x4) in ArmModel form, rooted in the world.

    Fixed joints between the world and the first controlled joint are folded into its
    origin; `fixed_tail` joints (held at a constant angle) are folded into the tool.
    """
    from pxr import Usd, UsdPhysics

    stage = Usd.Stage.Open(str(usd_path))
    joints = {p.GetName(): p for p in stage.Traverse() if p.IsA(UsdPhysics.Joint)}
    by_child = {}
    for name, p in joints.items():
        A, B, axis, b0, b1 = _joint_frames(p)
        if b1:
            by_child[b1] = (name, A, B, axis, b0, p)

    # world -> body0 of the first controlled joint, via fixed joints
    first = _joint_frames(joints[joint_names[0]])
    prefix, body = np.eye(4), first[3]
    while body is not None:
        name, A, B, axis, b0, p = by_child[body]
        if not p.IsA(UsdPhysics.FixedJoint):
            raise ValueError(f"non-fixed joint {name} between world and {joint_names[0]}")
        prefix = A @ np.linalg.inv(B) @ prefix
        body = b0

    out, carry, prev_child = [], prefix, first[3]
    for name in joint_names:
        A, B, axis, b0, b1 = _joint_frames(joints[name])
        if b0 != prev_child:
            raise ValueError(f"joint {name} does not continue the chain ({b0} != {prev_child})")
        out.append((carry @ A, axis))
        carry, prev_child = np.linalg.inv(B), b1
    for name, angle in (fixed_tail or {}).items():
        A, B, axis, b0, b1 = _joint_frames(joints[name])
        if b0 != prev_child:
            raise ValueError(f"tail joint {name} does not continue the chain")
        R = np.eye(4)
        R[:3, :3] = _axis_angle(axis, angle)
        carry, prev_child = carry @ A @ R @ np.linalg.inv(B), b1
    tool = carry.copy()
    tool[:3, 3] = (carry @ np.array([*tool_offset, 1.0]))[:3]
    return out, tool, prev_child


def _hull(P):
    """Convex hull of points -> (vertices (k,3), planes (m,4) with n.x + d <= 0 inside)."""
    from scipy.spatial import ConvexHull

    h = ConvexHull(P)
    return P[h.vertices], h.equations


def load_collision_hulls(usd_path, body_names) -> dict:
    """{body: (vertices, planes)} in the world frame at the authored rest pose (q = 0).
    PhysX collides these bodies as convex hulls (MeshCollisionAPI approximation 'convexHull'),
    so the same hulls are what the planner checks."""
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.Open(str(usd_path))
    cache = UsdGeom.XformCache()
    bodies = {p.GetName(): p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)}
    out = {}
    for name in body_names:
        pts = []
        for c in Usd.PrimRange(bodies[name]):
            if c.HasAPI(UsdPhysics.CollisionAPI) and c.IsA(UsdGeom.Mesh):
                P = np.array(UsdGeom.Mesh(c).GetPointsAttr().Get() or [])
                if len(P):
                    pts.append((np.c_[P, np.ones(len(P))] @ np.array(cache.GetLocalToWorldTransform(c)))[:, :3])
        if not pts:
            raise ValueError(f"{name} has no collision mesh")
        out[name] = _hull(np.vstack(pts))
    return out


def _child_bodies(usd_path, joint_names) -> list:
    from pxr import Usd, UsdPhysics

    stage = Usd.Stage.Open(str(usd_path))
    joints = {p.GetName(): p for p in stage.Traverse() if p.IsA(UsdPhysics.Joint)}
    return [_joint_frames(joints[n])[4].rsplit("/", 1)[-1] for n in joint_names]


def _move(V, H, T):
    """Rigidly transform hull vertices and plane equations by 4x4 T."""
    R, t = T[:3, :3], T[:3, 3]
    n = H[:, :3] @ R.T
    return V @ R.T + t, np.c_[n, H[:, 3] - n @ t]


def _hull_gap(Va, Ha, Vb, Hb) -> float:
    """Conservative signed separation of two convex hulls (m): <0 means overlap.
    max over one hull's face planes of a vertex of the other is a lower bound on the true
    point-to-hull distance outside and the exact depth inside; checking both ways catches
    vertex-face contacts (edge-edge is covered by the margin)."""
    return float(min((Va @ Hb[:, :3].T + Hb[:, 3]).max(1).min(), (Vb @ Ha[:, :3].T + Ha[:, 3]).max(1).min()))


@dataclass
class ArmModel:
    joints: list            # [(T_origin 4x4, axis(3))]
    tool: np.ndarray        # last joint frame -> tool point
    lower: np.ndarray
    upper: np.ndarray
    ee_body: str = ""
    # Self-collision (left arm vs torso / right arm), all as convex hulls like PhysX:
    obstacles: list = None  # [(name, V, H)] in the frame after joint 0 (waist): they ride on it
    links: list = None      # [(name, frame_idx, V, H)] in the frame after joint `frame_idx` (len(joints) = tool)
    margin: float = 0.01    # minimum accepted separation
    path_margin: float = 0.003  # free joint-space moves may pass this close (contact is at 0)
    influence: float = 0.025  # IK pushes the arm away (in its null space) below this separation

    @classmethod
    def from_cfg(cls, cfg: dict) -> "ArmModel":
        from . import project_path

        r = cfg["robot"]
        usd = project_path(r["usd"])
        joints, tool, ee = load_usd_chain(usd, r["joints"], r.get("fixed_tail"), r.get("tool_offset", (0, 0, 0)))
        if r.get("ee_body") and not ee.endswith("/" + r["ee_body"]):
            raise ValueError(f"chain ends at {ee}, config says ee_body={r['ee_body']}")
        arm = cls(joints, tool, np.asarray(r["lower"], float), np.asarray(r["upper"], float), ee)
        sc = r.get("self_collision") or {}
        if sc.get("obstacles") and sc.get("links"):
            hulls = load_collision_hulls(usd, sc["obstacles"] + sc["links"])
            F0 = arm.frames(np.zeros(len(joints)))
            W = np.linalg.inv(F0[0])
            arm.obstacles = [(n, *_move(*hulls[n], W)) for n in sc["obstacles"]]
            idx = {b: i for i, b in enumerate(_child_bodies(usd, r["joints"]))}
            idx[r["ee_body"]] = len(joints)
            arm.links = []
            for n in sc["links"]:
                if n not in idx:
                    raise ValueError(f"self_collision link {n} is not a body of the controlled chain")
                arm.links.append((n, idx[n], *_move(*hulls[n], np.linalg.inv(F0[idx[n]]))))
            arm.margin, arm.influence = float(sc.get("margin", 0.01)), float(sc.get("influence", 0.025))
            arm.path_margin = float(sc.get("path_margin", 0.003))
        return arm

    # ---------------------------------------------------------- self-collision
    def frames(self, q) -> list:
        """World pose of the frame after each joint, then the tool frame."""
        T, R, out = np.eye(4), np.eye(4), []
        for (T_o, axis), qi in zip(self.joints, q):
            R[:3, :3] = _axis_angle(axis, qi)
            T = T @ T_o @ R
            out.append(T.copy())
        out.append(T @ self.tool)
        return out

    def clearance(self, q) -> tuple[float, str]:
        """Smallest separation (m, <0 = penetration) between the checked arm links and the
        torso / right arm, and the colliding pair's name."""
        if not self.obstacles:
            return np.inf, ""
        F = self.frames(q)
        W = np.linalg.inv(F[0])
        best, pair = np.inf, ""
        for name, i, V, H in self.links:
            Vl, Hl = _move(V, H, W @ F[i])
            for oname, Vo, Ho in self.obstacles:
                g = _hull_gap(Vl, Hl, Vo, Ho)
                if g < best:
                    best, pair = g, f"{name}-{oname}"
        return best, pair

    def self_collision(self, q) -> str | None:
        """'link-body' pair closer than `margin`, else None."""
        g, pair = self.clearance(q)
        return pair if g < self.margin else None

    def _clearance_grad(self, q, c0, h=1e-3) -> np.ndarray:
        g = np.zeros(len(q))
        for j in range(1, len(q)):          # the waist turns arm and torso together: no effect
            dq = q.copy()
            dq[j] += h
            g[j] = (self.clearance(dq)[0] - c0) / h
        return g

    # ------------------------------------------------------------------ FK
    def fk(self, q) -> np.ndarray:
        T, R = np.eye(4), np.eye(4)
        for (T_o, axis), qi in zip(self.joints, q):
            R[:3, :3] = _axis_angle(axis, qi)
            T = T @ T_o @ R
        return T @ self.tool

    def _fk_jac(self, q):
        T, R = np.eye(4), np.eye(4)
        A, O = [], []
        for (T_o, axis), qi in zip(self.joints, q):
            T = T @ T_o
            A.append(T[:3, :3] @ axis)
            O.append(T[:3, 3].copy())
            R[:3, :3] = _axis_angle(axis, qi)
            T = T @ R
        p = (T @ self.tool)[:3, 3]
        return p, np.cross(np.array(A), p - np.array(O)).T     # (3,), dp/dq (3,n)

    # ------------------------------------------------------------------ IK
    def ik(self, p_des, q_seed, tol=5e-4, max_iter=150, n_restarts=6, rng=None) -> np.ndarray:
        """Position IK. The 3 redundant DOF are used to keep the arm clear of the torso: when the
        separation drops below `influence`, a clearance gradient step is projected into the
        Jacobian's null space (the tool does not move), so the solution stays on the same
        branch instead of jumping to a distant collision-free one."""
        p_des = np.asarray(p_des, float)
        rng = rng or np.random.default_rng(0)
        seeds = [np.clip(np.asarray(q_seed, float), self.lower, self.upper)]
        if n_restarts:
            seeds += list(self._bank_seeds(p_des, k=min(4, n_restarts)))
            seeds += [rng.uniform(self.lower, self.upper) for _ in range(n_restarts)]
        I = np.eye(len(seeds[0]))
        best, best_d, hit = None, np.inf, None
        for q in seeds:
            q, lam, prev = q.copy(), 1e-2, np.inf
            for it in range(max_iter):
                p, J = self._fk_jac(q)
                e = p_des - p
                err = np.linalg.norm(e)
                c = self.clearance(q)[0] if self.obstacles else np.inf
                clear = c >= self.margin + 0.005 or (c >= self.margin and err < tol)
                if err < tol and clear:
                    break
                if it > 20 and err > 0.999 * prev and clear:
                    break
                prev = err
                Jp = J.T @ np.linalg.inv(J @ J.T + lam ** 2 * np.eye(3))
                dq = Jp @ e
                if c < self.influence:
                    g = self._clearance_grad(q, c)
                    gn = np.linalg.norm(g)
                    if gn > 1e-9:
                        dq = dq + (I - Jp @ J) @ (g / gn) * min(0.1, 2.0 * (self.influence - c))
                q = np.clip(q + np.clip(dq, -0.25, 0.25), self.lower, self.upper)
            if np.linalg.norm(p_des - self._fk_jac(q)[0]) < 3e-3:
                col = self.self_collision(q)
                if col:
                    hit = hit or col
                    continue
                d = np.linalg.norm(q - seeds[0])
                if d < best_d:
                    best, best_d = q, d
                if best_d < 0.4:
                    return best
        if best is None:
            raise IKError(f"{np.round(p_des, 3)} is reachable only with {hit} contact (the arm would hit the "
                          "robot's own body)" if hit else f"{np.round(p_des, 3)} is out of reach within the joint limits")
        return best

    def _bank_seeds(self, p_des, k=4, n=4000):
        if getattr(self, "_bank", None) is None:
            Q = np.random.default_rng(1).uniform(self.lower, self.upper, size=(n, len(self.lower)))
            P = np.array([self._fk_jac(q)[0] for q in Q])
            object.__setattr__(self, "_bank", (Q, P))
        Q, P = self._bank
        return Q[np.argsort(np.linalg.norm(P - p_des, axis=1))[:k]]


# ----------------------------------------------------------------------------
@dataclass
class Waypoint:
    pos: np.ndarray
    speed: float
    label: str = ""


@dataclass
class JointTrajectory:
    q: np.ndarray           # (N, n)
    t: np.ndarray           # (N,)
    labels: list

    def sample(self, t: float) -> np.ndarray:
        if t <= self.t[0]:
            return self.q[0]
        if t >= self.t[-1]:
            return self.q[-1]
        i = int(np.searchsorted(self.t, t))
        a = (t - self.t[i - 1]) / (self.t[i] - self.t[i - 1])
        return self.q[i - 1] + a * (self.q[i] - self.q[i - 1])


def plan_joint(arm: ArmModel, q_start, q_goal, max_joint_vel=1.5, step=0.05, min_z=None, label="") -> JointTrajectory:
    """Joint-space interpolation for free moves, checked every `step` rad: the arm never gets
    closer to its own body than min(path_margin, clearance at the start), and the tool stays above min_z."""
    q_start, q_goal = np.asarray(q_start, float), np.asarray(q_goal, float)
    n = max(1, int(np.ceil(np.max(np.abs(q_goal - q_start)) / step)))
    floor_c = min(arm.path_margin, arm.clearance(q_start)[0])
    qs, ts = [q_start], [0.0]
    for k in range(1, n + 1):
        q = q_start + (q_goal - q_start) * k / n
        g, pair = arm.clearance(q)
        if g < floor_c:
            raise IKError(f"joint-space move to '{label}' passes {pair} at {g * 1000:.0f} mm")
        if min_z is not None and arm.fk(q)[2, 3] < min_z:
            raise IKError(f"joint-space move to '{label}' takes the tool below {min_z} m")
        qs.append(q)
        ts.append(ts[-1] + max(np.max(np.abs(q - qs[-2])) / max_joint_vel, 1 / 30))
    return JointTrajectory(np.array(qs), np.array(ts), ["start"] + [label] * n)


def _concat(a: JointTrajectory, b: JointTrajectory) -> JointTrajectory:
    return JointTrajectory(np.r_[a.q, b.q[1:]], np.r_[a.t, a.t[-1] + b.t[1:]], a.labels + b.labels[1:])


def plan_sweep(arm: ArmModel, q_start, waypoints: list[Waypoint], spacing=0.01, max_joint_vel=1.5,
               min_z=None, via=None) -> JointTrajectory:
    """Free joint-space move to the first waypoint, then straight Cartesian segments through the rest.
    The free move avoids the large reconfigurations a straight tool line from the start pose can
    force near the torso; the contact-making segments stay Cartesian.
    If the direct free move comes too close to the body, it goes through `via` (a known-clear
    pose, e.g. the episode's start pose) instead."""
    first = waypoints[0]
    q_a = arm.ik(first.pos, q_start)
    try:
        free = plan_joint(arm, q_start, q_a, max_joint_vel, min_z=min_z, label=first.label)
    except IKError:
        if via is None:
            raise
        q_a = arm.ik(first.pos, via)
        free = _concat(plan_joint(arm, q_start, via, max_joint_vel, min_z=min_z, label="via"),
                       plan_joint(arm, via, q_a, max_joint_vel, min_z=min_z, label=first.label))
    if len(waypoints) == 1:
        return free
    return _concat(free, plan_cartesian(arm, q_a, waypoints[1:], spacing, max_joint_vel, warm=True))


def plan_cartesian(arm: ArmModel, q_start, waypoints: list[Waypoint], spacing=0.01,
                   max_joint_vel=1.5, max_jump=0.35, warm=False) -> JointTrajectory:
    """Straight-line Cartesian segments -> continuous, velocity-limited joint path."""
    q = np.asarray(q_start, float)
    p_prev = arm.fk(q)[:3, 3]
    qs, ts, labels, t = [q], [0.0], ["start"], 0.0
    first = not warm                      # warm: q_start already solves p_prev, check jumps from step 1
    for wp in waypoints:
        n = max(1, int(np.ceil(np.linalg.norm(wp.pos - p_prev) / spacing)))
        for k in range(1, n + 1):
            p = p_prev + (wp.pos - p_prev) * k / n
            qn = arm.ik(p, qs[-1], n_restarts=6 if first else 1)
            jump = float(np.max(np.abs(qn - qs[-1])))
            if not first and jump > max_jump:
                raise IKError(f"discontinuity {jump:.2f} rad near '{wp.label}' {np.round(p, 3)}")
            first = False
            t += max(np.linalg.norm(p - arm.fk(qs[-1])[:3, 3]) / wp.speed, jump / max_joint_vel, 1 / 30)
            qs.append(qn)
            ts.append(t)
            labels.append(wp.label)
        p_prev = wp.pos
    return JointTrajectory(np.array(qs), np.array(ts), labels)
