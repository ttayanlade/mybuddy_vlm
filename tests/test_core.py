"""CPU tests (no Isaac Sim): kinematics from the USD, camera maths, metric, primitive, agents.

    .venv/bin/python -m pytest -q
"""

import numpy as np
import pytest

from mybuddy_vlm import load_config
from mybuddy_vlm.agents import DecisionError, HeuristicAgent, parse_decision
from mybuddy_vlm.geometry import Camera, quat_to_matrix, snap_to_mask
from mybuddy_vlm.kinematics import ArmModel, IKError, plan_cartesian, plan_sweep
from mybuddy_vlm.metric import SuccessTracker, occluded_pixels, target_box_mask
from mybuddy_vlm.primitives import PrimitiveError, sweep


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def arm(cfg):
    return ArmModel.from_cfg(cfg)


def their_camera(cfg, res):
    """Their camera: opengl-convention offset -> ROS optical model, square pixels."""
    c = cfg["camera"]
    R_gl = quat_to_matrix(c["rot_opengl"])
    R_ros = R_gl @ np.diag([1.0, -1.0, -1.0])           # opengl (x right, y up, -z fwd) -> ROS
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R_ros, c["pos"]
    f = c["focal_length"] / c["horizontal_aperture"] * res
    return Camera([[f, 0, res / 2], [0, f, res / 2], [0, 0, 1]], T, res, res)


# ------------------------------------------------------------------ kinematics
def test_fk_matches_usd_rest_pose(arm):
    """At q=0 the chain must reproduce the authored world pose of left_arm_l6."""
    assert arm.ee_body.endswith("left_arm_l6")
    assert np.allclose(arm.fk(np.zeros(6))[:3, 3], [0.17, 0.0311, 0.081], atol=5e-4)


def test_analytic_jacobian(arm):
    rng = np.random.default_rng(0)
    for _ in range(5):
        q = rng.uniform(arm.lower, arm.upper)
        p, J = arm._fk_jac(q)
        Jn = np.stack([(arm._fk_jac(q + 1e-6 * np.eye(6)[i])[0] - p) / 1e-6 for i in range(6)], 1)
        assert np.allclose(J, Jn, atol=1e-5)


def test_ik_reaches_the_plant_region(arm, cfg):
    q0 = np.radians(cfg["robot"]["init_deg"])
    for p in ([0.06, -0.25, 0.20], [0.10, -0.30, 0.25], [0.0, -0.22, 0.15]):
        q = arm.ik(p, q0)
        assert np.linalg.norm(arm.fk(q)[:3, 3] - p) < 3e-3
        assert np.all(q >= arm.lower - 1e-9) and np.all(q <= arm.upper + 1e-9)


def test_ik_rejects_fruit_row_far_side(arm, cfg):
    with pytest.raises(IKError):
        arm.ik([0.0, -0.60, 0.28], np.radians(cfg["robot"]["init_deg"]), n_restarts=2)


# ------------------------------------------------------------------ camera
def test_their_camera_looks_forward_and_down(cfg):
    cam = their_camera(cfg, 256)
    fwd = cam.R[:, 2]
    assert fwd[1] < -0.85 and fwd[2] < -0.3             # toward -y (plant/fruit), tilted down ~25 deg
    # image directions: right = world -x, up = mostly world +z
    assert np.allclose(cam.image_direction("right"), [-1, 0, 0], atol=1e-3)
    assert cam.image_direction("up")[2] > 0.85


def test_fruits_project_inside_image_and_outside_plant_band(cfg):
    cam = their_camera(cfg, 256)
    s = cfg["scene"]
    for dx in (0.0, 0.15):
        uv, z = cam.project(np.array(s["fruit_base_pos"])[None] + [dx, 0, 0])
        assert 0 < uv[0, 0] < 256 and 0 < uv[0, 1] < 256
        assert z[0] > cfg["metric"]["plant_depth_band"][1]   # fruit itself never counts as plant


def test_deproject_inverts_project(cfg):
    cam = their_camera(cfg, 768)
    p = np.array([0.05, -0.27, 0.22])
    (uv,), (z,) = cam.project(p[None])
    assert np.allclose(cam.deproject(uv[0], uv[1], z), p, atol=1e-9)


# ------------------------------------------------------------------ metric
def test_metric_and_success_rule(cfg):
    cam = their_camera(cfg, 256)
    m = cfg["metric"]
    fruit = np.array(cfg["scene"]["fruit_base_pos"])
    tm = target_box_mask(cam, fruit, m["mask_size"])
    assert tm.sum() == m["mask_size"] ** 2 + 2 * m["mask_size"] + 1 or tm.sum() >= m["mask_size"] ** 2
    plant_depth = np.full((256, 256), 0.25, np.float32)     # everything in the plant band
    assert occluded_pixels(plant_depth, tm, m["plant_depth_band"]) == tm.sum()
    clear = np.full((256, 256), 0.6, np.float32)
    assert occluded_pixels(clear, tm, m["plant_depth_band"]) == 0
    tr = SuccessTracker(m["visible_px"], m["sustain_steps"])
    assert not any(tr.update(0) for _ in range(m["sustain_steps"] - 1))
    assert tr.update(0)
    assert not tr.update(500)                                 # broken streak


# ------------------------------------------------------------------ primitive
def test_sweep_geometry_and_fruit_safety(cfg):
    cam = their_camera(cfg, 768)
    c = cfg["agent"]
    P = np.array([0.06, -0.26, 0.22])
    left = cam.image_direction("left")
    wps = sweep(P, left, cam.R[:, 2], [], cfg["scene"]["fruit_radius"], c)
    a, s, e, r = (w.pos for w in wps)
    assert np.allclose(e - s, left * (c["pre_dist"] + c["push_dist"]))
    assert np.dot(a - s, cam.R[:, 2]) < 0                    # approach comes from the camera side
    fruit_on_path = P + left * 0.05
    with pytest.raises(PrimitiveError, match="fruit"):
        sweep(P, left, cam.R[:, 2], [fruit_on_path], cfg["scene"]["fruit_radius"], c)
    with pytest.raises(PrimitiveError, match="floor"):
        sweep(np.array([0.06, -0.26, 0.02]), cam.image_direction("down"), cam.R[:, 2], [], 0.03, c)


@pytest.mark.parametrize("direction", ["left", "right", "up", "down"])
def test_sweep_is_executable_from_their_start_pose(cfg, arm, direction):
    cam = their_camera(cfg, 768)
    c = cfg["agent"]
    P = np.array([0.06, -0.25, 0.20])
    wps = sweep(P, cam.image_direction(direction), cam.R[:, 2], [], cfg["scene"]["fruit_radius"], c)
    q0 = np.radians(cfg["robot"]["init_deg"])
    traj = plan_cartesian(arm, q0, wps, c["waypoint_spacing"], cfg["robot"]["max_joint_vel"])
    assert np.all(np.diff(traj.t) > 0)
    assert np.linalg.norm(arm.fk(traj.q[-1])[:3, 3] - wps[-1].pos) < 3e-3


# ------------------------------------------------------------------ agents
def test_parse_decision_formats():
    d = parse_decision('{"point": [500, 250], "direction": "LEFT", "reasoning": "stem"}', 768, 768, "norm1000")
    assert d.direction == "left" and np.allclose(d.point_px, (384, 192))
    d = parse_decision('<think>hmm</think>```json\n{"point": [100, 50], "direction": "up", "reasoning": ""}\n```',
                       768, 768, "pixels")
    assert d.point_px == (100.0, 50.0)
    for bad in ('{"point": [5000, 1], "direction": "left"}', '{"point": [1], "direction": "left"}',
                '{"point": [1, 2], "direction": "sideways"}', "push the leaf"):
        with pytest.raises(DecisionError):
            parse_decision(bad, 768, 768, "norm1000")


def test_heuristic_agent_and_snap():
    plant = np.zeros((100, 100), bool)
    plant[40:60, 40:45] = True                               # a stem left of the box centre (x=50)
    ag = HeuristicAgent()
    ag.set_context(plant, (40, 40, 60, 60))
    d = ag.decide(np.zeros((100, 100, 3), np.uint8), "red", [])
    assert plant[int(d.point_px[1]), int(d.point_px[0])] and d.direction == "left"
    assert snap_to_mask(plant, 50, 50, 10) == (44, 50)
    assert snap_to_mask(plant, 90, 90, 5) is None


def test_vlm_agent_request_offline(cfg, monkeypatch):
    from mybuddy_vlm.agents import VLMAgent

    sent = []
    replies = iter(["not json", '{"point": [520, 400], "direction": "right", "reasoning": "leaf over box"}'])

    def fake_post(self, payload):
        sent.append(payload)
        return next(replies)

    monkeypatch.setattr(VLMAgent, "_post", fake_post)
    d = VLMAgent(cfg["vlm"]).decide(np.zeros((768, 768, 3), np.uint8), "red", [])
    assert d.direction == "right" and len(sent) == 2
    assert sent[0]["response_format"]["json_schema"]["schema"]["properties"]["direction"]["enum"] == \
        ["left", "right", "up", "down"]
    assert "red fruit" in sent[0]["messages"][1]["content"][0]["text"]


def test_self_collision_hulls_match_physx_contacts(arm, cfg):
    """Poses where Isaac reported left_arm_l3 touching base_link (runs 1117-145945, 1117-150933;
    the latter with per-partner force filtering) overlap in the hull model; all initial poses
    (both j2 branches) are clear."""
    for q in ([0.066, -2.536, 0.022, 2.241, -2.006, 3.072], [0.486, -2.609, 0.02, 2.429, -1.94, 3.086],
              [0.28, -2.616, 0.008, 2.448, -1.951, 3.073], [0.364, -2.606, 0.023, 2.448, -1.931, 3.089],
              [0.391, -2.615, -0.002, 2.471, -1.934, 3.074]):
        g, pair = arm.clearance(np.array(q))
        assert g < 0 and pair.endswith("base_link")
    q0 = np.radians(cfg["robot"]["init_deg"])
    for j2 in (q0[2], -0.52, -0.17, 0.2, 0.3):
        q = q0.copy()
        q[2] = j2
        assert arm.clearance(q)[0] > 0


@pytest.mark.parametrize("direction", ["left", "right", "up", "down"])
@pytest.mark.parametrize("j2", [-0.52, -0.3, 0.2, 0.3])
def test_sweep_keeps_clear_of_torso(cfg, arm, direction, j2):
    """Sweeps like the ones that hit the torso in sim: IK solutions keep >= margin, free moves >= path_margin."""
    cam = their_camera(cfg, 768)
    c = cfg["agent"]
    q0 = np.radians(cfg["robot"]["init_deg"])
    q0[2] = j2
    for P in ([0.06, -0.25, 0.20], [0.03, -0.20, 0.21], [0.09, -0.27, 0.19]):
        wps = sweep(np.array(P), cam.image_direction(direction), cam.R[:, 2], [], cfg["scene"]["fruit_radius"], c)
        traj = plan_sweep(arm, q0, wps[:3], c["waypoint_spacing"], cfg["robot"]["max_joint_vel"], c["min_z"], via=q0)
        start_c = arm.clearance(q0)[0]
        assert min(arm.clearance(q)[0] for q in traj.q[1:]) >= min(arm.path_margin, start_c)
        cart = [q for q, lab in zip(traj.q, traj.labels) if lab in ("start", "sweep")][1:]
        assert min(arm.clearance(q)[0] for q in cart) >= arm.margin
        assert np.all(np.diff(traj.t) > 0)
        assert np.linalg.norm(arm.fk(traj.q[-1])[:3, 3] - wps[2].pos) < 3e-3
        back = plan_cartesian(arm, traj.q[-1], wps[3:], c["waypoint_spacing"], cfg["robot"]["max_joint_vel"],
                              warm=True)
        assert min(arm.clearance(q)[0] for q in back.q[1:]) >= arm.margin
