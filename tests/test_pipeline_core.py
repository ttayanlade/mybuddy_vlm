"""Unit tests for the parts of the pipeline that need no GPU, ROS or Isaac Sim.

    pytest -q tests/
"""

import numpy as np
import pytest

from mybuddy_vlm.config import T_base_camera, intrinsics_from_cfg, load_config
from mybuddy_vlm.geometry import (deproject_pixel, depth_to_meters, mask_center_pixel, matrix_to_quat,
                                  project_point, quat_to_matrix, robust_depth_at, transform_points)
from mybuddy_vlm.grounding import Detection2D, lift_to_base
from mybuddy_vlm.kinematics import ArmModel, IKError, plan_joint_trajectory
from mybuddy_vlm.primitives import PrimitiveError, PrimitiveLibrary, action_direction
from mybuddy_vlm.vlm_client import VLMResponseError, parse_decision


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def arm(cfg):
    return ArmModel.from_cfg(cfg["robot"])


# ---------------------------------------------------------------- geometry
def test_quat_roundtrip():
    rng = np.random.default_rng(1)
    for _ in range(20):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        q *= np.sign(q[0])
        assert np.allclose(matrix_to_quat(quat_to_matrix(q)), q, atol=1e-8)


def test_deproject_project_roundtrip(cfg):
    K = intrinsics_from_cfg(cfg)
    p = deproject_pixel(400, 100, 0.6, K)
    assert np.allclose(project_point(p, K), [400, 100])
    assert np.allclose(deproject_pixel(K.cx, K.cy, 0.5, K), [0, 0, 0.5])


def test_static_camera_looks_at_workspace(cfg):
    """Principal ray of the eye-to-hand camera must hit the table inside the arm workspace."""
    T = T_base_camera(cfg, "sim")
    origin, ray = T[:3, 3], T[:3, :3] @ np.array([0, 0, 1.0])
    s = -origin[2] / ray[2]
    hit = origin + s * ray
    ws = cfg["primitives"]["workspace"]
    assert ws["x"][0] <= hit[0] <= ws["x"][1] + 0.05 and abs(hit[1]) < 0.05
    # image right (+x_cam) maps to base +Y for this camera => push_left == -Y in both conventions
    assert np.allclose(T[:3, 0], [0, 1, 0], atol=1e-3)


def test_depth_conversion():
    d16 = np.array([[0, 500], [1000, 65535]], np.uint16)
    m = depth_to_meters(d16, 0.001)
    assert np.isnan(m[0, 0]) and m[0, 1] == pytest.approx(0.5)
    f32 = np.array([[np.inf, 0.4]], np.float32)
    m = depth_to_meters(f32, 0.001)
    assert np.isnan(m[0, 0]) and m[0, 1] == pytest.approx(0.4)


def test_mask_center_snaps_onto_ring():
    m = np.zeros((50, 50), bool)
    m[10:40, 10:40] = True
    m[15:35, 15:35] = False  # ring: centroid is in the hole
    u, v = mask_center_pixel(m)
    assert m[v, u]


def test_robust_depth_ignores_holes():
    d = np.full((40, 40), 0.55, np.float32)
    d[18:22, 18:22] = np.nan
    m = np.ones_like(d, bool)
    assert robust_depth_at(d, m, 20, 20, 1) == pytest.approx(0.55)


def synthetic_scene(cfg, p_obj_base, half=(0.03, 0.02), height=0.04):
    """Render a box-shaped depth patch for an object at p_obj_base as seen by the static camera."""
    K = intrinsics_from_cfg(cfg)
    T_bc = T_base_camera(cfg, "sim")
    T_cb = np.linalg.inv(T_bc)
    xs = np.linspace(-half[0], half[0], 40)
    ys = np.linspace(-half[1], half[1], 40)
    pts = np.array([[p_obj_base[0] + x, p_obj_base[1] + y, height] for x in xs for y in ys])
    pc = transform_points(T_cb, pts)
    depth = np.full((K.height, K.width), np.nan, np.float32)
    mask = np.zeros((K.height, K.width), bool)
    for p in pc:
        u, v = np.round(project_point(p, K)).astype(int)
        for du in (-1, 0, 1):
            for dv in (-1, 0, 1):
                depth[v + dv, u + du] = p[2]
                mask[v + dv, u + du] = True
    return K, T_bc, depth, mask


def test_lift_to_base_recovers_object_position(cfg):
    p_true = np.array([0.20, -0.05, 0.0])
    K, T_bc, depth, mask = synthetic_scene(cfg, p_true)
    vs, us = np.nonzero(mask)
    det = Detection2D("blue towel", 0.9, np.array([us.min(), vs.min(), us.max(), vs.max()], float), mask)
    g = lift_to_base(det, depth, K, T_bc)
    assert np.allclose(g.center_base[:2], p_true[:2], atol=0.006)
    assert g.top_z == pytest.approx(0.04, abs=0.003)
    assert np.allclose(g.extent_xy, [0.03, 0.02], atol=0.006)


# --------------------------------------------------------------------- VLM
def test_parse_valid_and_fenced():
    d = parse_decision('{"target_obstacle": "blue towel", "primitive_action": "LIFT", "reasoning": "flat cloth"}')
    assert d.primitive_action == "lift" and d.target_obstacle == "blue towel"
    d = parse_decision('```json\n{"target_obstacle": "mug", "primitive_action": "push_left", "reasoning": "r"}\n```')
    assert d.primitive_action == "push_left"


@pytest.mark.parametrize("bad", [
    "I think you should move the towel",
    '{"target_obstacle": "mug", "primitive_action": "throw", "reasoning": "x"}',
    '{"target_obstacle": "mug", "primitive_action": "pull"}',
    '{"target_obstacle": "", "primitive_action": "pull", "reasoning": "x"}',
    '{"target_obstacle": "mug", "primitive_action": "pull", "reasoning": "x", "extra": 1}',
    '["mug", "pull"]',
])
def test_parse_rejects(bad):
    with pytest.raises(VLMResponseError):
        parse_decision(bad)


# -------------------------------------------------------------- kinematics
def test_ik_fk_consistency(arm, cfg):
    for p in ([0.18, 0.0, 0.08], [0.22, -0.10, 0.04], [0.14, 0.14, 0.03]):
        q = arm.ik(p, cfg["robot"]["home"])
        T = arm.fk(q)
        assert np.allclose(T[:3, 3], p, atol=3e-3)
        assert T[2, 2] < -0.99  # tool pointing down
        assert np.all(q >= arm.lower - 1e-9) and np.all(q <= arm.upper + 1e-9)


def test_ik_rejects_unreachable(arm, cfg):
    with pytest.raises(IKError):
        arm.ik([0.6, 0.0, 0.05], cfg["robot"]["home"], n_restarts=3)


# -------------------------------------------------------------- primitives
def make_grounded(center, top_z=0.03, extent=(0.04, 0.03)):
    from mybuddy_vlm.grounding import Grounded3D

    return Grounded3D("obj", 0.9, (0, 0), np.array(center, float), top_z, np.array(extent, float),
                      np.zeros(4), np.zeros((1, 1), bool), 100)


def test_push_left_geometry(cfg):
    lib = PrimitiveLibrary(cfg["primitives"], False)
    g = make_grounded([0.19, 0.0, 0.02])
    wps = lib.plan("push_left", g).steps[0].waypoints
    approach, contact, sweep, retreat = wps
    pc = cfg["primitives"]
    assert approach.pos[1] > g.center_base[1] + g.extent_xy[1]          # starts on +Y side
    assert contact.pos[2] < g.top_z and contact.pos[2] >= pc["min_z"]     # lowered slightly
    assert np.allclose(contact.pos[:2], approach.pos[:2])
    assert sweep.pos[1] < contact.pos[1] and np.isclose(sweep.pos[0], contact.pos[0])  # moves -Y only
    assert sweep.pos[2] == contact.pos[2]
    assert retreat.pos[2] > g.top_z


def test_push_right_mirrors(cfg):
    lib = PrimitiveLibrary(cfg["primitives"], False)
    wps = lib.plan("push_right", make_grounded([0.19, 0.0, 0.02])).steps[0].waypoints
    assert wps[2].pos[1] > wps[1].pos[1]


def test_camera_direction_frame_agrees_for_default_camera(cfg):
    T = T_base_camera(cfg, "sim")
    d = action_direction("push_left", "camera", T, np.array([0.2, 0.0]))
    assert np.allclose(d, [0, -1], atol=1e-3)


def test_pull_falls_back_to_press_drag_when_far_side_unreachable(cfg):
    lib = PrimitiveLibrary(cfg["primitives"], False)
    plan = lib.plan("pull", make_grounded([0.25, 0.0, 0.005], top_z=0.012, extent=(0.08, 0.06)))
    assert "press" in plan.notes
    drag = [w for w in plan.steps[0].waypoints if w.label == "drag"][0]
    assert np.linalg.norm(drag.pos[:2]) < 0.25  # dragged toward base


def test_out_of_reach_is_refused(cfg):
    lib = PrimitiveLibrary(cfg["primitives"], False)
    with pytest.raises(PrimitiveError):
        lib.plan("push_left", make_grounded([0.40, 0.0, 0.02]))


def test_lift_with_gripper_has_grasp_events(cfg):
    lib = PrimitiveLibrary(cfg["primitives"], True)
    kinds = [s.kind for s in lib.plan("lift", make_grounded([0.18, -0.02, 0.01], top_z=0.015)).steps]
    assert kinds.count("gripper") == 3


@pytest.mark.parametrize("action", ["push_left", "push_right", "pull", "lift"])
def test_every_primitive_plans_a_feasible_joint_trajectory(cfg, arm, action):
    """End to end (minus ROS): grounded towel over the keys -> primitive -> IK trajectory."""
    o = cfg["scene"]["objects"]["towel"]
    g = make_grounded([o["pos"][0], o["pos"][1], 0.01], top_z=0.015,
                      extent=(o["size"][0] / 2, o["size"][1] / 2))
    lib = PrimitiveLibrary(cfg["primitives"], cfg["robot"]["gripper"]["enabled"])
    plan = lib.plan(action, g)
    q = np.array(cfg["robot"]["home"])
    for st in plan.steps:
        if st.kind != "move":
            continue
        traj = plan_joint_trajectory(arm, cfg["robot"]["joint_names"], q, st.waypoints,
                                     cfg["primitives"]["waypoint_spacing"], cfg["robot"]["max_joint_vel"])
        assert np.all(np.diff(traj.times) > 0)
        assert np.max(np.abs(np.diff(traj.positions, axis=0))) < 0.36
        assert np.allclose(arm.fk(traj.positions[-1])[:3, 3], st.waypoints[-1].pos, atol=3e-3)
        q = traj.positions[-1]


def test_workspace_dead_zone_respected(cfg):
    from mybuddy_vlm.primitives import Workspace

    ws = Workspace(cfg["primitives"])
    assert not ws.contains(np.array([0.10, 0.0, 0.05]))
    assert np.linalg.norm(ws.clip(np.array([0.10, 0.0, 0.05]))[:2]) >= ws.r_min - 1e-9
    lib = PrimitiveLibrary(cfg["primitives"], False)
    for action in ("pull", "lift", "push_left", "push_right"):
        for st in lib.plan(action, make_grounded([0.18, -0.01, 0.01], top_z=0.015, extent=(0.08, 0.065))).steps:
            for w in st.waypoints:
                assert ws.contains(w.pos), (action, w.label, w.pos)


# ------------------------------------------------------- MyBuddy URDF / kinematics
def test_extracted_urdf_is_physics_ready(cfg):
    import xml.etree.ElementTree as ET
    from mybuddy_vlm.config import project_path

    path = project_path(cfg["robot"]["urdf_single_arm"])
    root = ET.parse(path).getroot()
    joints = {j.get("name"): j for j in root.findall("joint")}
    assert [n for n in joints if joints[n].get("type") == "revolute"] == cfg["robot"]["joint_names"]
    for n in cfg["robot"]["joint_names"]:
        assert float(joints[n].find("limit").get("velocity")) > 0          # upstream had 0
        assert float(joints[n].find("limit").get("effort")) < 10           # upstream had 1000
    for link in root.findall("link"):
        assert link.find("inertial") is not None, link.get("name")          # upstream had none
        for mesh in link.iter("mesh"):
            assert (path.parent / mesh.get("filename")).exists()           # relative, resolvable
    j1 = joints["joint1"].find("origin")
    assert j1.get("xyz").split()[:2] == ["0", "0"]                         # mounted upright


def test_dh_conversion_matches_classic_dh(cfg):
    """from_dh() must reproduce the textbook DH product exactly."""
    from mybuddy_vlm.kinematics import dh_transform

    dh = np.asarray(cfg["robot"]["dh"], float)
    arm = ArmModel.from_dh(dh, -np.pi * np.ones(6), np.pi * np.ones(6))
    rng = np.random.default_rng(3)
    for _ in range(10):
        q = rng.uniform(-2, 2, 6)
        T = np.eye(4)
        for (d, a, al, off), qi in zip(dh, q):
            T = T @ dh_transform(d, a, al, qi + off)
        assert np.allclose(arm.fk(q), T, atol=1e-10)


def test_analytic_jacobian_matches_numeric(arm):
    rng = np.random.default_rng(4)
    for _ in range(5):
        q = rng.uniform(arm.lower, arm.upper)
        e, J, _ = arm._task(q, np.array([0.2, 0.0, 0.05]), np.array([0, 0, -1.0]))
        assert np.allclose(J, arm._num_jac(q, np.array([0.2, 0.0, 0.05]), np.array([0, 0, -1.0]), e), atol=1e-5)


def test_workspace_front_is_reachable(arm, cfg):
    """Every point of the configured workspace front (where clutter lives) is solvable."""
    for x in (0.14, 0.20, 0.25):
        for y in (-0.10, 0.0, 0.10):
            for z in (0.012, 0.05, 0.10):
                if np.hypot(x, y) <= cfg["primitives"]["workspace"]["max_radius"]:
                    arm.ik([x, y, z], cfg["robot"]["home"])


# ------------------------------------------------------- open-source / reasoning VLMs
def test_parse_strips_reasoning_model_think_block():
    txt = ('<think>The towel is flat and blue; keys are likely beneath it. Lifting exposes the most area.'
           '</think>\n{"target_obstacle": "blue towel", "primitive_action": "lift", "reasoning": "covers keys"}')
    assert parse_decision(txt).primitive_action == "lift"
    # some chat templates drop the opening tag
    txt2 = 'thinking...</think>{"target_obstacle": "mug", "primitive_action": "push_right", "reasoning": "r"}'
    assert parse_decision(txt2).target_obstacle == "mug"


def test_schema_enum_matches_primitives():
    from mybuddy_vlm.vlm_client import DECISION_SCHEMA, PRIMITIVES

    assert DECISION_SCHEMA["properties"]["primitive_action"]["enum"] == list(PRIMITIVES)
    assert set(DECISION_SCHEMA["required"]) == {"target_obstacle", "primitive_action", "reasoning"}


def test_env_overrides_vlm_endpoint(monkeypatch):
    monkeypatch.setenv("VLM_BASE_URL", "http://nova-gpu:8123/v1")
    monkeypatch.setenv("VLM_MODEL", "nvidia/Cosmos-Reason1-7B")
    c = load_config()
    assert c["vlm"]["base_url"] == "http://nova-gpu:8123/v1" and c["vlm"]["model"] == "nvidia/Cosmos-Reason1-7B"


def test_vlm_client_request_and_retry(cfg, monkeypatch):
    """Request shape for vLLM structured outputs + retry on a malformed first reply (no server)."""
    pytest.importorskip("openai")
    from types import SimpleNamespace as NS
    from mybuddy_vlm.vlm_client import VLMClient

    replies = iter(["Sure! I would move the towel.",
                    '{"target_obstacle": "blue towel", "primitive_action": "pull", "reasoning": "on top"}'])
    calls = []

    def fake_create(**kw):
        calls.append(kw)
        return NS(choices=[NS(message=NS(content=next(replies), reasoning_content=None))])

    client = VLMClient(cfg["vlm"])
    monkeypatch.setattr(client.client.chat.completions, "create", fake_create)
    d = client.decide(np.zeros((48, 64, 3), np.uint8), "keys",
                      history=[{"target_obstacle": "mug", "primitive_action": "push_left", "outcome": "executed"}])
    assert d.primitive_action == "pull" and len(calls) == 2
    first = calls[0]
    assert first["model"] == cfg["vlm"]["model"]
    assert first["response_format"]["type"] == "json_schema"
    text = first["messages"][1]["content"][0]["text"]
    assert "finding keys" in text and "push_left on 'mug'" in text
    assert first["messages"][1]["content"][1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert "Invalid response" in calls[1]["messages"][-1]["content"]   # error fed back on retry
