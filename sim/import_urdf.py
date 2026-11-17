#!/usr/bin/env python3
"""Convert the single-arm MyBuddy URDF to USD with Isaac Sim's URDF importer,
then explicitly apply the ArticulationRoot and compliant joint-drive gains.

Run with Isaac Lab's python (it boots Isaac Sim headless):

  ./isaaclab.sh -p sim/import_urdf.py --headless
  # or: ${ISAACSIM_PATH}/python.sh sim/import_urdf.py --headless

What it does
  1. URDFCreateImportConfig / URDFParseAndImportFile (isaacsim.asset.importer.urdf,
     falling back to the older omni.importer.urdf module name) with fix_base=True.
  2. Applies UsdPhysics.ArticulationRootAPI (+ PhysxArticulationAPI) on the BASE
     link prim, removing any root API the importer put elsewhere, so there is
     exactly one articulation root at the robot base.
  3. Writes per-joint angular position drives (stiffness/damping/max force) from
     config/pipeline.yaml -> robot.drives. Low stiffness = compliant pushes.
  4. Saves to robot.usd_out. mybuddy_env.py loads this USD and *also* sets the same
     gains through Isaac Lab's ImplicitActuatorCfg, so either path is consistent.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--config", default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

# ---- everything below needs the running Kit app ---------------------------
import omni.kit.commands  # noqa: E402
import omni.usd  # noqa: E402
from pxr import PhysxSchema, Sdf, Usd, UsdPhysics  # noqa: E402

from mybuddy_vlm.config import load_config, project_path  # noqa: E402

try:  # Isaac Sim >= 4.5
    from isaacsim.asset.importer.urdf import _urdf
except ImportError:  # Isaac Sim 4.0 / 4.2
    from omni.importer.urdf import _urdf


def main():
    cfg = load_config(args.config)
    rc = cfg["robot"]
    urdf = project_path(rc["urdf_single_arm"])
    usd_out = project_path(rc["usd_out"])
    usd_out.parent.mkdir(parents=True, exist_ok=True)
    if not urdf.exists():
        raise SystemExit(f"{urdf} not found — run sim/extract_single_arm.py first")

    # 1) URDF importer --------------------------------------------------------
    _, ic = omni.kit.commands.execute("URDFCreateImportConfig")
    ic.merge_fixed_joints = False      # keep flange/camera frames as separate prims
    ic.convex_decomp = False           # convex hulls are fine for a small arm
    ic.import_inertia_tensor = True
    ic.fix_base = True                 # base bolted to the table
    ic.self_collision = False
    ic.make_default_prim = True
    ic.create_physics_scene = False
    ic.distance_scale = 1.0
    ic.density = 0.0                   # use URDF masses
    # Drive type/gains: attribute names changed across importer versions.
    for name, val in (("default_drive_type", _urdf.UrdfJointTargetType.JOINT_DRIVE_POSITION),
                      ("default_drive_strength", float(rc["drives"]["stiffness"][0])),
                      ("default_position_drive_damping", float(rc["drives"]["damping"][0]))):
        if hasattr(ic, name):
            setattr(ic, name, val)

    ok, prim_path = omni.kit.commands.execute(
        "URDFParseAndImportFile", urdf_path=str(urdf), import_config=ic, dest_path=str(usd_out),
        get_articulation_root=True)
    if not ok:
        raise SystemExit("URDF import failed")
    print(f"[import] URDF -> {usd_out} (importer reported root {prim_path})")

    # 2-3) open the written USD and enforce root + drives ----------------------
    stage = Usd.Stage.Open(str(usd_out))
    default = stage.GetDefaultPrim()
    base_name = rc["base_link"]
    base = next((p for p in Usd.PrimRange(default) if p.GetName() == base_name), None)
    if base is None:
        raise SystemExit(f"base link prim '{base_name}' not found under {default.GetPath()}")

    for p in Usd.PrimRange(default):
        if p != base and p.HasAPI(UsdPhysics.ArticulationRootAPI):
            p.RemoveAPI(UsdPhysics.ArticulationRootAPI)
            if p.HasAPI(PhysxSchema.PhysxArticulationAPI):
                p.RemoveAPI(PhysxSchema.PhysxArticulationAPI)
    UsdPhysics.ArticulationRootAPI.Apply(base)
    px = PhysxSchema.PhysxArticulationAPI.Apply(base)
    px.CreateEnabledSelfCollisionsAttr(False)
    px.CreateSolverPositionIterationCountAttr(16)   # small light links need more iterations
    px.CreateSolverVelocityIterationCountAttr(4)
    print(f"[import] ArticulationRoot on {base.GetPath()}")

    d = rc["drives"]
    found = set()
    for p in Usd.PrimRange(default):
        if not p.IsA(UsdPhysics.RevoluteJoint):
            continue
        name = p.GetName()
        if name not in rc["joint_names"]:
            continue
        i = rc["joint_names"].index(name)
        drv = UsdPhysics.DriveAPI.Apply(p, "angular")
        drv.CreateTypeAttr("force")
        # USD angular drives are per DEGREE: convert from N·m/rad.
        drv.CreateStiffnessAttr(d["stiffness"][i] * math.pi / 180.0)
        drv.CreateDampingAttr(d["damping"][i] * math.pi / 180.0)
        drv.CreateMaxForceAttr(d["effort_limit"][i])
        drv.CreateTargetPositionAttr(math.degrees(rc["home"][i]))
        pj = PhysxSchema.PhysxJointAPI.Apply(p)
        pj.CreateMaxJointVelocityAttr(math.degrees(d["velocity_limit"]))
        pj.CreateJointFrictionAttr(0.02)   # small servo friction
        found.add(name)
    missing = set(rc["joint_names"]) - found
    if missing:
        raise SystemExit(f"revolute joints not found in USD: {sorted(missing)} — "
                         "fix robot.joint_names or re-run extract_single_arm.py --rename-joints auto")
    stage.GetRootLayer().Save()
    print(f"[import] drives set on {sorted(found)}; saved {usd_out}")


if __name__ == "__main__":
    main()
    app.close()
