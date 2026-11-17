#!/usr/bin/env python3
"""Step 1 — Isaac Lab tabletop scene with a single MyBuddy 280 arm + ROS 2 bridge.

  ./isaaclab.sh -p sim/mybuddy_env.py                 # GUI
  ./isaaclab.sh -p sim/mybuddy_env.py --headless --enable_cameras

Scene
  * table (static collider) + ground + dome light (randomised on reset)
  * MyBuddy single arm from the USD made by sim/import_urdf.py, fixed base,
    ArticulationRoot at the base, compliant implicit PD drives (robot.drives)
  * eye-to-hand static RGB-D camera + eye-in-hand wrist RGB-D camera, both
    with RealSense D435 colour intrinsics (640x480, fx=fy=615)
  * clutter: mug + box (rigid, YCB), keys (small rigid target), towel
    (semi-rigid FEM soft body) draped over the keys. All masses 0.03–0.1 kg
    because the real arm has ~250 g payload.

Sim-to-real protocol
  * the robot is driven only through ROS 2 (sim/ros2_sim_bridge.py) with the
    same message types and relative topic names as the hardware;
  * camera intrinsics, extrinsics and drive gains come from config/pipeline.yaml;
  * D435-like depth noise/dropout; object pose + lighting randomisation on reset
    (publish std_msgs/Empty to /sim/sim/reset).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description="MyBuddy VLM search scene")
parser.add_argument("--config", default=None)
parser.add_argument("--no_ros", action="store_true", help="run the scene without the ROS 2 bridge")
parser.add_argument("--seed", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True  # RGB-D sensors need rendering even when headless
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ---- Isaac imports (after the app is up) ----------------------------------
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402
from isaaclab.assets import (ArticulationCfg, AssetBaseCfg, DeformableObjectCfg,  # noqa: E402
                             RigidObjectCfg)
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.sensors import CameraCfg  # noqa: E402
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR  # noqa: E402

from mybuddy_vlm.config import load_config, project_path  # noqa: E402

CFG = load_config(args.config)


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def base_to_world(p_base):
    """Object positions in config are in the robot base frame (base quat = identity)."""
    return tuple(float(a + b) for a, b in zip(p_base, CFG["robot"]["base_pos_world"]))


def make_actuator() -> ImplicitActuatorCfg:
    r, d = CFG["robot"], CFG["robot"]["drives"]
    kw = dict(joint_names_expr=r["joint_names"],
              stiffness={n: d["stiffness"][i] for i, n in enumerate(r["joint_names"])},
              damping={n: d["damping"][i] for i, n in enumerate(r["joint_names"])})
    eff = {n: d["effort_limit"][i] for i, n in enumerate(r["joint_names"])}
    try:  # Isaac Lab >= 2.1 separates sim limits from actuator-model limits
        return ImplicitActuatorCfg(**kw, effort_limit_sim=eff, velocity_limit_sim=d["velocity_limit"])
    except TypeError:
        return ImplicitActuatorCfg(**kw, effort_limit=eff, velocity_limit=d["velocity_limit"])


def d435_pinhole() -> sim_utils.PinholeCameraCfg:
    c = CFG["camera"]
    K = [c["fx"], 0.0, c["cx"], 0.0, c["fy"], c["cy"], 0.0, 0.0, 1.0]
    return sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
        intrinsic_matrix=K, width=c["width"], height=c["height"],
        focal_length=c["focal_length_mm"], clipping_range=tuple(c["clipping_range"]))


def light_rigid_props():
    # small, light objects: more solver iterations + a little damping keeps them from jittering
    return sim_utils.RigidBodyPropertiesCfg(solver_position_iteration_count=16, solver_velocity_iteration_count=2,
                                            max_depenetration_velocity=0.5, linear_damping=0.05,
                                            angular_damping=0.05)


FRICTION = sim_utils.RigidBodyMaterialCfg(static_friction=0.8, dynamic_friction=0.6, restitution=0.0)


def make_object(name: str, o: dict):
    pos = base_to_world(o["pos"])
    kind = o["kind"]
    if kind == "ycb":
        s = o.get("scale", 1.0)
        spawn = sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/{o['usd']}", scale=(s, s, s),
                                     rigid_props=light_rigid_props(),
                                     mass_props=sim_utils.MassPropertiesCfg(mass=o["mass"]),
                                     collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002,
                                                                                      rest_offset=0.0))
        return RigidObjectCfg(prim_path=f"{{ENV_REGEX_NS}}/{name}", spawn=spawn,
                              init_state=RigidObjectCfg.InitialStateCfg(pos=pos))
    if kind == "cuboid" or (kind == "towel" and o.get("mode") == "rigid_thin"):
        spawn = sim_utils.CuboidCfg(
            size=tuple(o["size"]), rigid_props=light_rigid_props(),
            mass_props=sim_utils.MassPropertiesCfg(mass=o["mass"]),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
            physics_material=FRICTION,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=tuple(o["color"]),
                                                        metallic=o.get("metallic", 0.0),
                                                        roughness=0.4 if o.get("metallic") else 0.9))
        return RigidObjectCfg(prim_path=f"{{ENV_REGEX_NS}}/{name}", spawn=spawn,
                              init_state=RigidObjectCfg.InitialStateCfg(pos=pos))
    if kind == "towel":
        # Semi-rigid towel as a thin FEM soft body (PhysX deformable; GPU physics only).
        spawn = sim_utils.MeshCuboidCfg(
            size=tuple(o["size"]),
            deformable_props=sim_utils.DeformableBodyPropertiesCfg(
                rest_offset=0.0, contact_offset=0.002, simulation_hexahedral_resolution=12,
                solver_position_iteration_count=20, self_collision=False),
            physics_material=sim_utils.DeformableBodyMaterialCfg(
                youngs_modulus=o["youngs_modulus"], poissons_ratio=o["poissons_ratio"],
                dynamic_friction=0.8, elasticity_damping=0.02, damping_scale=1.0),
            mass_props=sim_utils.MassPropertiesCfg(mass=o["mass"]),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=tuple(o["color"]), roughness=1.0))
        return DeformableObjectCfg(prim_path=f"{{ENV_REGEX_NS}}/{name}", spawn=spawn,
                                   init_state=DeformableObjectCfg.InitialStateCfg(pos=pos))
    raise ValueError(f"unknown object kind '{kind}' for {name}")


def build_scene_cfg() -> InteractiveSceneCfg:
    r, sc = CFG["robot"], CFG["scene"]
    cfg = InteractiveSceneCfg(num_envs=1, env_spacing=3.0, replicate_physics=False)

    cfg.ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    cfg.light = AssetBaseCfg(prim_path="/World/light",
                             spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(1.0, 1.0, 1.0)))
    t = sc["table"]
    cfg.table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/table",
        spawn=sim_utils.CuboidCfg(size=tuple(t["size"]),
                                  collision_props=sim_utils.CollisionPropertiesCfg(),
                                  physics_material=FRICTION,
                                  visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.42, 0.30),
                                                                              roughness=0.8)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=tuple(t["pos"])))

    usd = project_path(r["usd_out"])
    if not usd.exists():
        raise SystemExit(f"{usd} missing — run sim/extract_single_arm.py then sim/import_urdf.py")
    cfg.robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(usd),
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False, max_depenetration_velocity=1.0),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False, solver_position_iteration_count=16,
                solver_velocity_iteration_count=4, fix_root_link=True)),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=tuple(r["base_pos_world"]), rot=tuple(r["base_quat_world"]),
            joint_pos={n: float(q) for n, q in zip(r["joint_names"], r["home"])}),
        actuators={"arm": make_actuator()},
        soft_joint_pos_limit_factor=0.95)

    cam_dt = 1.0 / CFG["camera"]["publish_hz"]
    s = sc["static_camera"]
    cfg.static_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/static_cam", update_period=cam_dt,
        height=CFG["camera"]["height"], width=CFG["camera"]["width"],
        data_types=["rgb", "distance_to_image_plane"], spawn=d435_pinhole(),
        offset=CameraCfg.OffsetCfg(pos=tuple(s["pos"]), rot=tuple(s["quat_ros"]), convention="ros"))
    w = sc["wrist_camera"]
    cfg.wrist_cam = CameraCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/{r['ee_link']}/wrist_cam", update_period=cam_dt,
        height=CFG["camera"]["height"], width=CFG["camera"]["width"],
        data_types=["rgb", "distance_to_image_plane"], spawn=d435_pinhole(),
        offset=CameraCfg.OffsetCfg(pos=tuple(w["pos"]), rot=tuple(w["quat_ros"]), convention="ros"))

    for name, o in sc["objects"].items():
        setattr(cfg, name, make_object(name, o))
    return cfg


# ----------------------------------------------------------------------------
# randomisation
# ----------------------------------------------------------------------------
def reset_scene(scene: InteractiveScene, rng: np.random.Generator):
    rnd = CFG["scene"]["randomize"]
    robot = scene["robot"]
    robot.write_joint_state_to_sim(robot.data.default_joint_pos.clone(), torch.zeros_like(robot.data.default_joint_vel))
    robot.set_joint_position_target(robot.data.default_joint_pos.clone())
    def jitter():
        return rng.uniform(-1, 1, 2) * rnd["object_xy_jitter"] if rnd["enabled"] else np.zeros(2)

    hidden_jitter = jitter()  # keys and towel move together so the towel stays ON the keys
    for name, o in CFG["scene"]["objects"].items():
        asset = scene[name]
        j = hidden_jitter if name in ("keys", "towel") else jitter()
        off = torch.tensor([j[0], j[1], 0.0], device=scene.device)
        if isinstance(asset.cfg, DeformableObjectCfg):
            nodal = asset.data.default_nodal_state_w.clone()
            nodal[..., :3] += off
            asset.write_nodal_state_to_sim(nodal)
        else:
            root = asset.data.default_root_state.clone()
            root[:, :3] += scene.env_origins + off
            asset.write_root_pose_to_sim(root[:, :7])
            asset.write_root_velocity_to_sim(torch.zeros_like(root[:, 7:]))
    if rnd["enabled"]:
        import omni.usd
        from pxr import Gf, UsdLux

        light = UsdLux.DomeLight(omni.usd.get_context().get_stage().GetPrimAtPath("/World/light"))
        light.GetIntensityAttr().Set(float(rng.uniform(*rnd["light_intensity"])))
        j = rnd["light_color_temp_jitter"]
        light.GetColorAttr().Set(Gf.Vec3f(*[float(1.0 - j * rng.random()) for _ in range(3)]))
    scene.reset()


# ----------------------------------------------------------------------------
def main():
    rng = np.random.default_rng(args.seed)
    needs_gpu = any(o["kind"] == "towel" and o.get("mode", "deformable") == "deformable"
                    for o in CFG["scene"]["objects"].values())
    if needs_gpu and not args.device.startswith("cuda"):
        raise SystemExit("deformable towel needs GPU physics (--device cuda:0) or set towel.mode: rigid_thin")

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1.0 / 120.0, device=args.device,
                                                              render_interval=4))
    sim.set_camera_view(eye=(0.9, 0.7, 1.3), target=(0.2, 0.0, 0.75))
    scene = InteractiveScene(build_scene_cfg())
    sim.reset()

    robot = scene["robot"]
    jn = CFG["robot"]["joint_names"]
    jids, _ = robot.find_joints(jn, preserve_order=True)
    gj = CFG["robot"]["gripper"]["joint_names"] if CFG["robot"]["gripper"]["enabled"] else []
    gids = robot.find_joints(gj, preserve_order=True)[0] if gj else []
    reset_scene(scene, rng)

    bridge = None
    if not args.no_ros:
        from isaacsim.core.utils.extensions import enable_extension

        enable_extension("isaacsim.ros2.bridge")  # brings Isaac Sim's bundled rclpy onto sys.path
        simulation_app.update()
        import rclpy

        from ros2_sim_bridge import SimBridge, spin_bridge

        rclpy.init()
        bridge = SimBridge(CFG, jn, gj)
        spin_bridge(bridge)

    dt = sim.get_physics_dt()
    cam_period = 1.0 / CFG["camera"]["publish_hz"]
    js_period = 1.0 / 50.0
    t_sim, next_cam, next_js = 0.0, 0.0, 0.0
    print("[mybuddy_env] running — Ctrl+C to quit")
    while simulation_app.is_running():
        if bridge is not None:
            if bridge.reset_requested:
                bridge.reset_requested = False
                bridge.clear_trajectory()
                reset_scene(scene, rng)
            q = robot.data.joint_pos[0, jids].cpu().numpy()
            tgt = bridge.joint_targets(t_sim, q)
            if tgt is not None:
                robot.set_joint_position_target(torch.tensor(tgt, dtype=torch.float32, device=sim.device)[None],
                                                joint_ids=jids)
            g = bridge.gripper_target()
            if g is not None and gids:
                lo = robot.data.soft_joint_pos_limits[0, gids, 0]
                hi = robot.data.soft_joint_pos_limits[0, gids, 1]
                robot.set_joint_position_target((lo + g * (hi - lo))[None], joint_ids=gids)

        scene.write_data_to_sim()
        sim.step()
        t_sim += dt
        scene.update(dt)

        if bridge is not None:
            bridge.publish_clock(t_sim)
            if t_sim >= next_js:
                next_js += js_period
                bridge.publish_joint_states(t_sim, robot.data.joint_pos[0, jids].cpu().numpy(),
                                            robot.data.joint_vel[0, jids].cpu().numpy())
            if t_sim >= next_cam:
                next_cam += cam_period
                for cam_name, wrist in (("static_cam", False), ("wrist_cam", True)):
                    out = scene[cam_name].data.output
                    bridge.publish_camera(t_sim, out["rgb"][0].cpu().numpy(),
                                          out["distance_to_image_plane"][0].cpu().numpy(), wrist=wrist)

    if bridge is not None:
        bridge.destroy_node()
        import rclpy

        rclpy.shutdown()


if __name__ == "__main__":
    main()
    simulation_app.close()
