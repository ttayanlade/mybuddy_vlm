"""Decoupled reason -> ground -> act loop (~1 Hz decision rate).

    python -m mybuddy_vlm.reasoning_node --sim            # against Isaac Sim
    python -m mybuddy_vlm.reasoning_node --real           # against the physical MyBuddy
    python -m mybuddy_vlm.reasoning_node --sim --dry-run  # plan + log, never move

--sim / --real change ONLY the ROS 2 namespace, the clock source and the
camera extrinsic source. VLM, grounding, primitives and planning are identical.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np

from .config import PROJECT_ROOT, T_base_camera, intrinsics_from_cfg, load_config
from .geometry import depth_to_meters
from .grounding import GroundingModel, lift_to_base
from .kinematics import ArmModel, CartesianWaypoint, IKError, plan_joint_trajectory
from .primitives import PrimitiveError, PrimitiveLibrary
from .vlm_client import VLMClient, VLMResponseError

log = logging.getLogger("mybuddy_vlm")


def save_debug(run_dir: Path, it: int, rgb, det=None, g=None, decision=None, extra: dict | None = None):
    from PIL import Image, ImageDraw

    img = Image.fromarray(rgb[..., :3].astype(np.uint8)).convert("RGBA")
    if det is not None and det.mask is not None:
        overlay = np.zeros((*det.mask.shape, 4), np.uint8)
        overlay[det.mask] = (255, 60, 60, 110)
        img = Image.alpha_composite(img, Image.fromarray(overlay))
    d = ImageDraw.Draw(img)
    if det is not None:
        d.rectangle(det.box_xyxy.tolist(), outline=(255, 255, 0, 255), width=2)
        d.text((det.box_xyxy[0] + 3, det.box_xyxy[1] + 3), f"{det.label} {det.score:.2f}", fill=(255, 255, 0, 255))
    if g is not None:
        u, v = g.center_px
        d.ellipse([u - 4, v - 4, u + 4, v + 4], outline=(0, 255, 0, 255), width=2)
    img.convert("RGB").save(run_dir / f"iter{it:02d}.jpg")
    rec = {"iter": it, **(extra or {})}
    if decision is not None:
        rec["vlm"] = {"target_obstacle": decision.target_obstacle,
                      "primitive_action": decision.primitive_action, "reasoning": decision.reasoning}
    if g is not None:
        rec["grounded"] = {"center_base": g.center_base.round(4).tolist(), "top_z": round(g.top_z, 4),
                           "extent_xy": g.extent_xy.round(4).tolist(), "score": g.score}
    with open(run_dir / "log.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")


class SearchAgent:
    def __init__(self, cfg: dict, mode: str, dry_run: bool = False):
        from .ros_interface import MoveItArm, RobotIO, spin_in_background

        self.cfg, self.mode, self.dry = cfg, mode, dry_run
        self.io = RobotIO(cfg, mode)
        self.executor, _ = spin_in_background(self.io)
        self.K = intrinsics_from_cfg(cfg)
        self.T_base_cam = T_base_camera(cfg, mode)
        self.vlm = VLMClient(cfg["vlm"])
        self.grounder = GroundingModel(cfg["grounding"])
        self.prims = PrimitiveLibrary(cfg["primitives"], cfg["robot"]["gripper"]["enabled"], self.T_base_cam)
        arm = ArmModel.from_cfg(cfg["robot"])
        if cfg["primitives"]["ik_backend"] == "moveit":
            arm = MoveItArm(self.io, arm, ee_link=cfg["robot"]["ee_link"], base_frame=cfg["robot"]["base_link"])
        self.arm = arm
        self.history: list[dict] = []
        self.run_dir = PROJECT_ROOT / "runs" / time.strftime(f"%Y%m%d-%H%M%S-{mode}")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        log.info("debug output -> %s", self.run_dir)

    # ------------------------------------------------------------------ utils
    def _intrinsics(self):
        """Prefer the live camera_info (real D435 units differ slightly)."""
        K = self.io.camera_info_K
        if K is not None and (abs(K.fx - self.K.fx) > 5 or abs(K.cx - self.K.cx) > 5):
            log.warning("camera_info differs from config intrinsics; using camera_info")
        return K or self.K

    def _move_cartesian(self, waypoints: list[CartesianWaypoint]) -> bool:
        p = self.cfg["primitives"]
        q0 = self.io.get_joint_positions()
        traj = plan_joint_trajectory(self.arm, self.io.joint_names, q0, waypoints,
                                     p["waypoint_spacing"], self.cfg["robot"]["max_joint_vel"])
        log.info("  trajectory: %d pts, %.1f s", len(traj.times), traj.times[-1])
        if self.dry:
            return True
        return self.io.execute(traj)

    def go_home(self):
        from .kinematics import JointTrajectory

        q0 = self.io.get_joint_positions()
        qh = np.array(self.cfg["robot"]["home"])
        T = max(1.0, float(np.max(np.abs(qh - q0))) / self.cfg["robot"]["max_joint_vel"])
        n = 20
        s = np.linspace(0, 1, n + 1)
        s = 3 * s ** 2 - 2 * s ** 3  # smoothstep: zero velocity at both ends
        traj = JointTrajectory(self.io.joint_names, q0 + np.outer(s, qh - q0),
                               np.linspace(0, T, n + 1), ["home"] * (n + 1))
        if not self.dry:
            self.io.execute(traj)

    # ------------------------------------------------------------------ steps
    def check_target_visible(self, rgb, depth_m):
        target = self.cfg["task"]["target"]
        dets = self.grounder.detect(rgb, target)
        if dets and dets[0].score >= self.cfg["task"]["target_detect_threshold"]:
            det = self.grounder.segment(rgb, dets[0])
            g = lift_to_base(det, depth_m, self._intrinsics(), self.T_base_cam,
                             self.cfg["grounding"]["depth_window_px"])
            return det, g
        return None, None

    def step(self, it: int, last_motion_end: float) -> bool:
        """One reason->ground->act iteration. Returns True when the target is found."""
        target = self.cfg["task"]["target"]
        obs = self.io.get_observation(newer_than=last_motion_end)
        depth_m = depth_to_meters(obs.depth, self.cfg["camera"]["depth_scale"])

        # 0) termination check — is the target already uncovered?
        det, g = self.check_target_visible(obs.rgb, depth_m)
        if g is not None:
            log.info("TARGET '%s' FOUND at %s (score %.2f)", target, g.center_base.round(3), g.score)
            self.io.status(json.dumps({"found": True, "target": target,
                                       "position_base": g.center_base.round(4).tolist()}))
            save_debug(self.run_dir, it, obs.rgb, det, g, extra={"found": True})
            return True

        # 1+2) VLM spatial reasoning
        try:
            decision = self.vlm.decide(obs.rgb, target, self.history)
        except VLMResponseError as e:
            log.error("VLM error: %s", e)
            return False
        self.io.status(json.dumps({"iter": it, "vlm": decision.__dict__ | {"raw": None}}))

        # 3) zero-shot grounding 2D -> 3D (base frame)
        det = self.grounder.ground_2d(obs.rgb, decision.target_obstacle)
        if det is None:
            self._record(decision, "obstacle not detected by grounding model")
            save_debug(self.run_dir, it, obs.rgb, decision=decision, extra={"outcome": "not grounded"})
            return False
        try:
            g = lift_to_base(det, depth_m, self._intrinsics(), self.T_base_cam,
                             self.cfg["grounding"]["depth_window_px"])
        except ValueError as e:
            self._record(decision, f"no valid depth: {e}")
            return False
        log.info("  grounded '%s' -> base %s top_z=%.3f extent=%s", det.label,
                 g.center_base.round(3), g.top_z, g.extent_xy.round(3))

        # 4) motion primitive -> trajectory -> execute
        try:
            plan = self.prims.plan(decision.primitive_action, g)
            log.info("  primitive %s (%s)", plan.action, plan.notes)
            ok = True
            for st in plan.steps:
                if st.kind == "move":
                    ok &= self._move_cartesian(st.waypoints)
                elif st.kind == "gripper" and not self.dry:
                    self.io.send_gripper(st.gripper)
            self.go_home()
            outcome = "executed" if ok else "executed but arm was blocked (did not reach goal)"
        except (PrimitiveError, IKError) as e:
            log.warning("  cannot execute: %s", e)
            outcome = f"infeasible: {e}"
        self._record(decision, outcome)
        save_debug(self.run_dir, it, obs.rgb, det, g, decision, extra={"outcome": outcome})
        return False

    def _record(self, decision, outcome: str):
        self.history.append({"target_obstacle": decision.target_obstacle,
                             "primitive_action": decision.primitive_action, "outcome": outcome})

    def run(self):
        period = 1.0 / self.cfg["task"]["loop_hz"]
        self.io.get_joint_positions(timeout=30.0)
        self.go_home()
        last_motion_end = self.io.now_s()
        for it in range(self.cfg["task"]["max_iterations"]):
            t0 = time.monotonic()
            log.info("=== iteration %d ===", it)
            if self.step(it, last_motion_end):
                return True
            last_motion_end = self.io.now_s()
            time.sleep(max(0.0, period - (time.monotonic() - t0)))  # cap at ~1 Hz
        log.info("gave up after %d iterations", self.cfg["task"]["max_iterations"])
        self.io.status(json.dumps({"found": False}))
        return False


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    m = ap.add_mutually_exclusive_group(required=True)
    m.add_argument("--sim", action="store_true", help="Isaac Sim (namespace /sim, sim time)")
    m.add_argument("--real", action="store_true", help="physical MyBuddy + RealSense")
    ap.add_argument("--config", default=None)
    ap.add_argument("--target", default=None, help="override task.target, e.g. 'keys'")
    ap.add_argument("--dry-run", action="store_true", help="perceive + plan, never command the arm")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(args.config)
    if args.target:
        cfg["task"]["target"] = args.target
    mode = "sim" if args.sim else "real"

    import rclpy

    rclpy.init()
    agent = SearchAgent(cfg, mode, dry_run=args.dry_run)
    try:
        found = agent.run()
    finally:
        agent.executor.shutdown()
        agent.io.destroy_node()
        rclpy.shutdown()
    raise SystemExit(0 if found else 1)


if __name__ == "__main__":
    main()
