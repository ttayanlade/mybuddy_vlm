"""Find the Fruit scene as an Isaac Lab DirectRLEnv — used as a SIMULATOR, not for RL.

Adapted from Find the Fruit (Subedi et al. 2025, github.com/nitesh-subedi/Find_the_Fruit:
env_cfg.py, find_the_fruit_env.py, scene_builder.py; BSD-3 Isaac Lab headers). Kept as in
the original: MyBuddy USD, stiff weightless drives, deformable plant + ground attachments,
2 goal spheres with x jitter and a random target, their camera, contact sensors, reset.
Changed for the VLM method:
  * actions are ABSOLUTE joint targets (our planner's trajectory), not policy deltas;
  * no auto-reset / termination inside the env (the runner decides), long episodes;
  * a second, higher-resolution camera with the same pose/optics for the VLM;
  * debug visualisation off (markers would appear in the images the VLM sees);
  * Isaac Lab 2.3 / Isaac Sim 5.1 API names (effort_limit_sim, velocity_limit_sim).
Import only after AppLauncher has started Isaac Sim.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg, DeformableObject, DeformableObjectCfg, RigidObject, \
    RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import Camera, CameraCfg, ContactSensor, ContactSensorCfg, TiledCamera, TiledCameraCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass
from isaaclab.utils.math import sample_uniform

from mybuddy_vlm import load_config, project_path

CFG = load_config()
LINK_SENSORS = ["left_arm_l1", "left_arm_l3", "left_arm_l4", "left_arm_l5", "left_arm_l6"]  # as theirs
# Diagnostics only: each link sensor also reports per-partner forces against these rigid bodies.
# Whatever net force they don't explain comes from something unlisted (plant, ground).
ROBOT_BODIES = ["base_footprint", "base_cylinder", "base_link", "left_arm_l1", "left_arm_l2", "left_arm_l3",
                "left_arm_l4", "left_arm_l5", "left_arm_l6", "right_arm_l1", "right_arm_l2", "right_arm_l3",
                "right_arm_l4", "right_arm_l5", "right_arm_l6"]


def contact_partners(link: str, num_fruits: int) -> list[str]:
    return [b for b in ROBOT_BODIES if b != link] + [f"GoalCube_{i}" for i in range(num_fruits)]
FRUIT_COLORS = [((1.0, 0.0, 0.0), "red"), ((0.0, 1.0, 0.0), "green"), ((0.0, 0.0, 1.0), "blue")]


def _pinhole(c):
    return sim_utils.PinholeCameraCfg(focal_length=c["focal_length"], focus_distance=c["focus_distance"],
                                      horizontal_aperture=c["horizontal_aperture"],
                                      vertical_aperture=c["vertical_aperture"],
                                      clipping_range=tuple(c["clipping_range"]), f_stop=240.0)


def _actuator(r):
    kw = dict(joint_names_expr=[".*"], stiffness=r["stiffness"], damping=r["damping"])
    try:
        return ImplicitActuatorCfg(**kw, effort_limit_sim=r["effort_limit"], velocity_limit_sim=r["velocity_limit"])
    except TypeError:
        return ImplicitActuatorCfg(**kw, effort_limit=r["effort_limit"], velocity_limit=r["velocity_limit"])


@configclass
class FtFEnvCfg(DirectRLEnvCfg):
    decimation = CFG["sim"]["decimation"]
    episode_length_s = 3600.0                         # runner decides when an episode ends
    action_scale = 1.0
    state_space = 0
    sim: SimulationCfg = SimulationCfg(
        dt=CFG["sim"]["dt"], render_interval=CFG["sim"]["decimation"],
        physx=PhysxCfg(gpu_temp_buffer_capacity=2 ** 26, gpu_found_lost_pairs_capacity=2 ** 27,
                       min_position_iteration_count=128),
        gravity=(0.0, 0.0, -9.81))
    robot_cfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/mybuddy",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(project_path(CFG["robot"]["usd"])), activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True, max_depenetration_velocity=5.0),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True, solver_position_iteration_count=12, solver_velocity_iteration_count=1)),
        actuators={".*": _actuator(CFG["robot"])})
    arm_dof_name = CFG["robot"]["joints"]
    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Camera",
        offset=TiledCameraCfg.OffsetCfg(pos=tuple(CFG["camera"]["pos"]), rot=tuple(CFG["camera"]["rot_opengl"]),
                                        convention="opengl"),
        data_types=["rgb", "depth"], spawn=_pinhole(CFG["camera"]),
        width=CFG["camera"]["metric_res"], height=CFG["camera"]["metric_res"])
    vlm_camera: CameraCfg = CameraCfg(
        prim_path="/World/envs/env_.*/VLMCamera",
        offset=CameraCfg.OffsetCfg(pos=tuple(CFG["camera"]["pos"]), rot=tuple(CFG["camera"]["rot_opengl"]),
                                   convention="opengl"),
        data_types=["rgb", "distance_to_image_plane"], spawn=_pinhole(CFG["camera"]),
        width=CFG["camera"]["vlm_res"], height=CFG["camera"]["vlm_res"])
    num_goal_cubes: int = CFG["scene"]["num_fruits"]
    action_space = gym.spaces.Box(low=-np.pi, high=np.pi, shape=(len(CFG["robot"]["joints"]),))
    observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(len(CFG["robot"]["joints"]),))
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=False)


# ----------------------------------------------------------------------------
# Deformable plant (scene_builder.py, unchanged physics)
# ----------------------------------------------------------------------------
def spawn_plant(prim_path, translation, usd_path, scale):
    cfg = sim_utils.UsdFileCfg(usd_path=usd_path, scale=(scale, scale, scale))
    cfg.func(prim_path, cfg, translation=translation, orientation=(1.0, 0.0, 0.0, 0.0))


def _make_deformable(stage, material_path, prim_dict, resolution=10):
    from omni.physx.scripts import deformableUtils, physicsUtils

    deformableUtils.add_deformable_body_material(stage, material_path, youngs_modulus=7.5e8, poissons_ratio=0.49,
                                                 damping_scale=1.0, dynamic_friction=0.5, density=2.0)
    for _, prim in prim_dict.items():
        deformableUtils.add_physx_deformable_body(stage, prim.GetPath(), collision_simplification=True,
                                                  simulation_hexahedral_resolution=resolution,
                                                  solver_position_iteration_count=128, self_collision=False)
        physicsUtils.add_physics_material_to_prim(stage, prim.GetPrim(), material_path)


def _attach(stage, prim_dict, plant_path):
    from pxr import PhysxSchema

    items = list(prim_dict.items())
    key, prim = items[0]                                  # main stalk -> ground
    att = PhysxSchema.PhysxPhysicsAttachment.Define(stage, prim.GetPath().AppendElementString(f"attachment_{key}"))
    att.GetActor0Rel().SetTargets([prim.GetPath()])
    att.GetActor1Rel().SetTargets(["/World/ground/GroundPlane/CollisionPlane"])
    api = PhysxSchema.PhysxAutoAttachmentAPI.Apply(att.GetPrim())
    for name, val in (("deformableVertexOverlapOffset", 0.05), ("enableDeformableVertexAttachments", True),
                      ("enableRigidSurfaceAttachments", True), ("enableCollisionFiltering", True),
                      ("collisionFilteringOffset", 0.01), ("enableDeformableFilteringPairs", True)):
        api.GetPrim().GetAttribute(f"physxAutoAttachment:{name}").Set(val)
    for key, prim in items[1:]:                           # sub-stalks -> main stalk
        att = PhysxSchema.PhysxPhysicsAttachment.Define(stage, prim.GetPath().AppendElementString(f"attachment_{key}"))
        att.GetActor0Rel().SetTargets([prim.GetPath()])
        att.GetActor1Rel().SetTargets([f"{plant_path}/stalk/plant_023"])
        PhysxSchema.PhysxAutoAttachmentAPI.Apply(att.GetPrim()).GetPrim().GetAttribute(
            "physxAutoAttachment:deformableVertexOverlapOffset").Set(0.005)


def add_plant(stage, env_prim_paths, material_path):
    objs = [DeformableObject(DeformableObjectCfg(prim_path="/World/envs/env_.*/Plant1/stalk", debug_vis=False))]
    objs += [DeformableObject(DeformableObjectCfg(prim_path=f"/World/envs/env_.*/Plant1/stalk{i}", debug_vis=False))
             for i in range(1, 6)]
    for env_path in env_prim_paths:
        plant = stage.GetPrimAtPath(f"{env_path}/Plant1")
        meshes = [m.GetAllChildren()[0] for m in plant.GetAllChildren()][1:]
        prim_dict = dict(zip(plant.GetAllChildrenNames()[1:], meshes))
        _make_deformable(stage, material_path, prim_dict)
        _attach(stage, prim_dict, f"{env_path}/Plant1")
    return objs


# ----------------------------------------------------------------------------
class FtFEnv(DirectRLEnv):
    cfg: FtFEnvCfg

    def __init__(self, cfg: FtFEnvCfg, render_mode=None, **kw):
        super().__init__(cfg, render_mode, **kw)
        self.arm_idx, _ = self._robot.find_joints(self.cfg.arm_dof_name, preserve_order=True)
        self.ee_idx = self._robot.find_bodies([CFG["robot"]["ee_body"]])[0][0]
        self.init_angles = torch.deg2rad(torch.tensor(CFG["robot"]["init_deg"], device=self.device))[None]
        self.root_fruit_states = torch.stack([c.data.root_com_state_w.clone() for c in self.goal_cubes], dim=1)
        self.target_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.targets = self.init_angles.clone()

    def _setup_scene(self):
        import isaacsim.core.utils.stage as stage_utils
        import omni.usd

        s = CFG["scene"]
        self.stage = stage_utils.get_current_stage()
        self._robot = Articulation(self.cfg.robot_cfg)
        self._tiled_camera = TiledCamera(self.cfg.tiled_camera)
        self._vlm_camera = Camera(self.cfg.vlm_camera)
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        spawn_plant("/World/envs/env_.*/Plant1", tuple(s["plant_pos"]), str(project_path(s["plant_usd"])),
                    s["plant_scale"])
        light = sim_utils.SphereLightCfg(intensity=3000.0, color=(1.0, 1.0, 1.0), radius=1.0)
        light.func("/World/envs/env_.*/Light", light, translation=(0.0, -0.4, 2.0))
        bg = sim_utils.UsdFileCfg(usd_path=str(project_path(s["background_usd"])), scale=(1.0, 1.0, 1.0))
        bg.func("/World/envs/env_.*/BackgroundPlane", bg, translation=(0.0, 0.25, 0.0), orientation=(1.0, 0, 0, 0))
        g2 = sim_utils.CuboidCfg(size=(1.25, 0.25, 0.01), visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.6, 0.6)))
        g2.func("/World/envs/env_.*/GroundPlane2", g2, translation=(0.0, -0.35, 0.0))
        self.goal_cubes = []
        for i in range(self.cfg.num_goal_cubes):
            color = FRUIT_COLORS[i % len(FRUIT_COLORS)][0]
            self.goal_cubes.append(RigidObject(RigidObjectCfg(
                prim_path=f"/World/envs/env_.*/GoalCube_{i}",
                spawn=sim_utils.SphereCfg(radius=s["fruit_radius"],
                                          visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
                                          rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
                                          mass_props=sim_utils.MassPropertiesCfg(mass=1.0)),
                init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(s["fruit_base_pos"])))))
        def _partner_path(p):
            return f"/World/envs/env_.*/{p}" if p.startswith("GoalCube") else f"/World/envs/env_.*/mybuddy/{p}"

        sensors = [ContactSensor(ContactSensorCfg(
            prim_path=f"/World/envs/env_.*/mybuddy/{n}", update_period=0.0, history_length=6, debug_vis=False,
            filter_prim_paths_expr=[_partner_path(p) for p in contact_partners(n, self.cfg.num_goal_cubes)]))
            for n in LINK_SENSORS]
        material = omni.usd.get_stage_next_free_path(self.stage, "/plant_material", True)
        self.scene.clone_environments(copy_from_source=True)
        self.scene.filter_collisions(global_prim_paths=[])
        self.plant_objs = add_plant(self.stage, self.scene.env_prim_paths, material)
        self.scene.articulations["robot"] = self._robot
        self.scene.sensors["tiled_camera"] = self._tiled_camera
        self.scene.sensors["vlm_camera"] = self._vlm_camera
        for i, sns in enumerate(sensors):
            self.scene.sensors[f"contact_sensor_{i}"] = sns
        for i, c in enumerate(self.goal_cubes):
            self.scene.rigid_objects[f"goal_cube_{i}"] = c
        for i, o in enumerate(self.plant_objs):
            self.scene.deformable_objects[f"stalk_{i}"] = o

    # ------------------------------------------------------------ stepping
    def _pre_physics_step(self, actions: torch.Tensor):
        self.targets = actions.clone()
        self.targets[:, 0].clamp_(CFG["robot"]["lower"][0], CFG["robot"]["upper"][0])   # their waist clamp

    def _apply_action(self):
        self._robot.set_joint_position_target(self.targets, joint_ids=self.arm_idx)

    def _get_observations(self):
        return {"policy": self._robot.data.joint_pos[:, self.arm_idx]}

    def _get_rewards(self):
        return torch.zeros(self.num_envs, device=self.device)

    def _get_dones(self):
        z = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return z, z

    # --------------------------------------------------------------- reset
    def _reset_idx(self, env_ids):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        super()._reset_idx(env_ids)
        n = len(env_ids)
        q = self._robot.data.default_joint_pos[env_ids][:, self.arm_idx] + self.init_angles.repeat(n, 1)
        lo, hi = CFG["robot"]["init_j2_ranges"][0 if np.random.uniform() < 0.5 else 1]
        q[:, 2] = sample_uniform(lo, hi, q[:, 2].shape, self.device)
        self.targets[env_ids] = q
        self._robot.set_joint_position_target(q, self.arm_idx, env_ids=env_ids)
        self._robot.write_joint_state_to_sim(q, torch.zeros_like(q), self.arm_idx, env_ids)
        for o in self.plant_objs:
            o.write_nodal_state_to_sim(o.data.default_nodal_state_w.clone()[env_ids], env_ids)
        self.target_idx[env_ids] = torch.randint(0, self.cfg.num_goal_cubes, (n,), device=self.device)
        base = self.root_fruit_states[env_ids]
        pos = base[:, :, :3].clone()
        jlo, jhi = CFG["scene"]["fruit_x_jitter"]
        pos[..., 0] += sample_uniform(jlo, jhi, pos[..., 0].shape, self.device)
        for k, c in enumerate(self.goal_cubes):
            st = base[:, k, :].clone()
            st[:, :3] = pos[:, k]
            c.write_root_pose_to_sim(st[:, :7], env_ids)
            c.write_root_velocity_to_sim(torch.zeros_like(st[:, 7:]), env_ids)

    # ------------------------------------------------------------ accessors (env 0)
    def joint_pos(self) -> np.ndarray:
        return self._robot.data.joint_pos[0, self.arm_idx].cpu().numpy()

    def fruit_positions(self) -> np.ndarray:
        o = self.scene.env_origins[0]
        return np.stack([(c.data.root_pos_w[0] - o).cpu().numpy() for c in self.goal_cubes])

    def target(self) -> tuple[int, str]:
        i = int(self.target_idx[0])
        return i, FRUIT_COLORS[i % len(FRUIT_COLORS)][1]

    def in_contact(self) -> bool:
        f = torch.cat([self.scene.sensors[f"contact_sensor_{i}"].data.net_forces_w for i in range(len(LINK_SENSORS))], 1)
        return bool(torch.any(f[0] != 0))

    def contact_report(self) -> dict:
        """Which sensed links touch something, plus geometry hints (fruit/floor distance)."""
        out = {}
        o = self.scene.env_origins[0]
        fruits = self.fruit_positions()
        for i, name in enumerate(LINK_SENSORS):
            data = self.scene.sensors[f"contact_sensor_{i}"].data
            f = data.net_forces_w[0]
            mag = float(torch.linalg.norm(f, dim=-1).max())
            if mag > 0:
                bid = self._robot.find_bodies([name])[0][0]
                p = (self._robot.data.body_pos_w[0, bid] - o).cpu().numpy()
                out[name] = {"force": round(mag, 3), "pos": p.round(3).tolist(),
                             "fruit_dist": np.linalg.norm(fruits - p, axis=1).round(3).tolist()}
                if data.force_matrix_w is not None:
                    fm = data.force_matrix_w[0, 0]                       # (num_partners, 3)
                    names = contact_partners(name, self.cfg.num_goal_cubes)
                    by = {n: round(float(v), 3) for n, v in zip(names, torch.linalg.norm(fm, dim=-1).tolist()) if v > 0}
                    out[name]["partners"] = by
                    out[name]["unexplained_force"] = round(float(torch.linalg.norm(f[0] - fm.sum(0))), 3)
        return out

    def plant_deviation(self) -> float:
        """Mean node displacement (m) of the plant from its default (rest) state."""
        d = [float(torch.linalg.norm(o.data.nodal_pos_w[0] - o.data.default_nodal_state_w[0, :, :3], dim=-1).mean())
             for o in self.plant_objs]
        return float(np.mean(d))

    def ee_pos(self) -> np.ndarray:
        return (self._robot.data.body_pos_w[0, self.ee_idx] - self.scene.env_origins[0]).cpu().numpy()

    def camera_model(self, which: str):
        """Pinhole model (ROS optical convention) in the env frame + latest images."""
        from mybuddy_vlm.geometry import Camera as Cam, pose_to_matrix

        cam = self._tiled_camera if which == "metric" else self._vlm_camera
        d = cam.data
        T = pose_to_matrix((d.pos_w[0] - self.scene.env_origins[0]).cpu().numpy(), d.quat_w_ros[0].cpu().numpy())
        K = d.intrinsic_matrices[0].cpu().numpy()
        rgb = d.output["rgb"][0].cpu().numpy()[..., :3].astype(np.uint8)
        depth = d.output["depth" if which == "metric" else "distance_to_image_plane"][0].cpu().numpy()
        depth = np.squeeze(depth).astype(np.float32)
        depth[~np.isfinite(depth)] = 0.0
        return Cam(K, T, rgb.shape[1], rgb.shape[0]), rgb, depth
