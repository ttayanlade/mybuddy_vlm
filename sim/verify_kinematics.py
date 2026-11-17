#!/usr/bin/env python3
"""Check that the numpy DH model (used for IK) matches the imported URDF/USD.

  ./isaaclab.sh -p sim/verify_kinematics.py --headless

Teleports the arm to random joint configurations, reads the ee_link pose from
PhysX and compares it with ArmModel.fk(). A constant offset means the URDF
base/flange frames differ from DH frame 0/6 (fix base_link/ee_link or add the
offset); a configuration-dependent error means joint signs/zeros differ.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--config", default=None)
parser.add_argument("--n", type=int, default=20)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402
from isaaclab.assets import Articulation, ArticulationCfg  # noqa: E402

from mybuddy_vlm.config import load_config, project_path  # noqa: E402
from mybuddy_vlm.geometry import quat_to_matrix  # noqa: E402
from mybuddy_vlm.kinematics import ArmModel  # noqa: E402


def main():
    cfg = load_config(args.config)
    r = cfg["robot"]
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.01, device=args.device, gravity=(0, 0, 0)))
    robot = Articulation(ArticulationCfg(
        prim_path="/World/Robot", spawn=sim_utils.UsdFileCfg(usd_path=str(project_path(r["usd_out"]))),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0, 0, 0)),
        actuators={"arm": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=1e4, damping=1e2)}))
    sim.reset()
    jids, _ = robot.find_joints(r["joint_names"], preserve_order=True)
    base_id = robot.find_bodies(r["base_link"])[0][0]
    ee_id = robot.find_bodies(r["ee_link"])[0][0]
    arm = ArmModel.from_cfg(r)
    rng = np.random.default_rng(0)
    errs = []
    for _ in range(args.n):
        q = rng.uniform(arm.lower * 0.6, arm.upper * 0.6)
        jp = robot.data.default_joint_pos.clone()
        jp[0, jids] = torch.tensor(q, dtype=jp.dtype, device=jp.device)
        robot.write_joint_state_to_sim(jp, torch.zeros_like(jp))
        sim.step(render=False)
        robot.update(0.01)
        pw = robot.data.body_pos_w[0].cpu().numpy()
        qw = robot.data.body_quat_w[0].cpu().numpy()
        R_b = quat_to_matrix(qw[base_id])
        p_sim = R_b.T @ (pw[ee_id] - pw[base_id])
        p_dh = arm.fk(q)[:3, 3]
        errs.append(p_sim - p_dh)
        print(f"q={np.round(q, 2)}  sim={np.round(p_sim, 4)}  dh={np.round(p_dh, 4)}  "
              f"|err|={np.linalg.norm(p_sim - p_dh) * 1000:.1f} mm")
    errs = np.array(errs)
    print(f"\nmean error vector: {np.round(errs.mean(0) * 1000, 1)} mm, "
          f"std: {np.round(errs.std(0) * 1000, 1)} mm, max |err| {np.linalg.norm(errs, axis=1).max() * 1000:.1f} mm")
    if np.linalg.norm(errs, axis=1).max() < 0.005:
        print("PASS: DH model matches the simulated arm")
    elif errs.std(0).max() < 0.003:
        print("Constant offset: base/flange frame mismatch — adjust robot.base_link/ee_link or tool offset")
    else:
        print("FAIL: joint sign/zero mismatch between DH table and URDF — fix robot.dh offsets/signs")


if __name__ == "__main__":
    main()
    app.close()
