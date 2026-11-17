#!/usr/bin/env python3
"""Find the Fruit with a VLM (no RL). One entry point, three modes.

  ./isaaclab.sh -p sim/run.py preview  --headless --enable_cameras --out runs/preview/x
  ./isaaclab.sh -p sim/run.py verify   --headless --enable_cameras
  ./isaaclab.sh -p sim/run.py episodes --headless --enable_cameras --agent vlm --episodes 10 --out runs/eval/x

Episode loop (per decision, at most agent.max_actions):
  observe (VLM camera, target fruit boxed) -> agent picks {point on plant, image direction}
  -> point snapped onto the plant + depth -> 3D; direction mapped through the camera pose
  -> sweep primitive (approach, start behind the point, push through, HOLD) -> IK trajectory
  -> executed at 30 Hz with Find the Fruit's occlusion metric checked every control step.
Success: occluded <= 160 px for 10 consecutive steps (theirs). Failure: rigid contact on an
arm link (theirs), or out of decisions.
"""

from __future__ import annotations

import argparse
import functools
import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from isaaclab.app import AppLauncher  # noqa: E402

from safe_exit import close_and_exit  # noqa: E402

print = functools.partial(print, flush=True)  # noqa: A001

ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=["preview", "verify", "episodes"])
ap.add_argument("--agent", choices=["vlm", "heuristic"], default="heuristic")
ap.add_argument("--episodes", type=int, default=1)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="runs/latest")
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()
args.enable_cameras = True
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from ftf_env import CFG, FtFEnv, FtFEnvCfg  # noqa: E402
from mybuddy_vlm.agents import HeuristicAgent, VLMAgent, annotate_target, draw_decision  # noqa: E402
from mybuddy_vlm.geometry import median_depth, snap_to_mask  # noqa: E402
from mybuddy_vlm.kinematics import ArmModel, IKError, plan_cartesian, plan_sweep  # noqa: E402
from mybuddy_vlm.metric import SuccessTracker, occluded_pixels, plant_mask, target_box_mask  # noqa: E402
from mybuddy_vlm.primitives import PrimitiveError, sweep  # noqa: E402

OUT = Path(args.out)
OUT.mkdir(parents=True, exist_ok=True)
M = CFG["metric"]


def colorize(depth, lo=0.05, hi=0.8):
    t = np.clip((depth - lo) / (hi - lo), 0, 1)
    rgb = np.stack([np.clip(1.5 - abs(4 * t - 3), 0, 1), np.clip(1.5 - abs(4 * t - 2), 0, 1),
                    np.clip(1.5 - abs(4 * t - 1), 0, 1)], -1)
    rgb[depth <= 0] = 0
    return (rgb * 255).astype(np.uint8)


class Runner:
    def __init__(self):
        self.env = FtFEnv(FtFEnvCfg())
        self.arm = ArmModel.from_cfg(CFG)
        self.ctrl_dt = CFG["sim"]["dt"] * CFG["sim"]["decimation"]

    # ------------------------------------------------------------------ basics
    def step(self, q):
        a = torch.tensor(np.asarray(q), dtype=torch.float32, device=self.env.device)[None]
        self.env.step(a)

    def reset(self, seed):
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.env.reset(seed=seed)
        q = self.env.joint_pos()
        for _ in range(int(CFG["sim"]["settle_s"] / self.ctrl_dt)):
            self.step(q)
        return q

    def metric_now(self):
        cam, _, depth = self.env.camera_model("metric")
        i, _ = self.env.target()
        tmask = target_box_mask(cam, self.env.fruit_positions()[i], M["mask_size"])
        return occluded_pixels(depth, tmask, M["plant_depth_band"]), int(tmask.sum())

    def vlm_view(self):
        """VLM camera image with the target fruit's region boxed (same rule as their mask, scaled)."""
        cam, rgb, depth = self.env.camera_model("vlm")
        i, color = self.env.target()
        (u, v), = cam.project(self.env.fruit_positions()[i][None])[0]
        h = M["mask_size"] / 2 * cam.width / CFG["camera"]["metric_res"]
        box = (u - h, v - h, u + h, v + h)
        return cam, rgb, depth, box, color

    # ------------------------------------------------------------------ preview
    def preview(self):
        from pxr import UsdGeom

        report = {}
        self.reset(args.seed)
        cam, rgb, depth, box, color = self.vlm_view()
        Image.fromarray(annotate_target(rgb, box)).save(OUT / "vlm_view.png")
        Image.fromarray(colorize(depth)).save(OUT / "vlm_depth.png")
        mcam, mrgb, mdepth = self.env.camera_model("metric")
        pm = plant_mask(mdepth, M["plant_depth_band"])
        i, _ = self.env.target()
        tm = target_box_mask(mcam, self.env.fruit_positions()[i], M["mask_size"])
        over = mrgb.copy()
        over[pm] = (0.5 * over[pm] + 0.5 * np.array([0, 255, 0])).astype(np.uint8)
        over[tm] = (0.5 * over[tm] + 0.5 * np.array([255, 230, 0])).astype(np.uint8)
        Image.fromarray(over).resize((512, 512), Image.NEAREST).save(OUT / "metric_masks.png")
        report["occluded_px_with_plant"] = occluded_pixels(mdepth, tm, M["plant_depth_band"])
        report["target"], report["fruits"] = color, self.env.fruit_positions().round(4).tolist()
        report["ee_pos"], report["joint_pos"] = self.env.ee_pos().round(4).tolist(), self.env.joint_pos().round(4).tolist()
        report["fk_vs_sim_mm"] = float(np.linalg.norm(self.arm.fk(self.env.joint_pos())[:3, 3] - self.env.ee_pos()) * 1000)
        report["in_contact"] = self.env.in_contact()
        # hide the plant: the target box should then sit on a clearly visible fruit
        stage = self.env.stage
        plant = stage.GetPrimAtPath("/World/envs/env_0/Plant1")
        UsdGeom.Imageable(plant).MakeInvisible()
        for _ in range(4):
            self.step(self.env.joint_pos())
        _, rgb_np, _, box_np, _ = self.vlm_view()
        Image.fromarray(annotate_target(rgb_np, box_np)).save(OUT / "vlm_view_plant_hidden.png")
        _, _, mdepth2 = self.env.camera_model("metric")
        report["occluded_px_plant_hidden"] = occluded_pixels(mdepth2, tm, M["plant_depth_band"])
        UsdGeom.Imageable(plant).MakeVisible()
        (OUT / "report.json").write_text(json.dumps(report, indent=2))
        print("[preview]", json.dumps(report))

    # ------------------------------------------------------------------ verify
    def verify(self):
        self.reset(args.seed)
        rng = np.random.default_rng(1)
        errs = []
        for _ in range(15):
            q = rng.uniform(self.arm.lower * 0.6, self.arm.upper * 0.6)
            jp = self.env._robot.data.joint_pos.clone()
            jp[:, self.env.arm_idx] = torch.tensor(q, dtype=jp.dtype, device=jp.device)
            self.env._robot.write_joint_state_to_sim(jp, torch.zeros_like(jp))
            for _ in range(3):
                self.step(q)
            qs = self.env.joint_pos()
            e = float(np.linalg.norm(self.arm.fk(qs)[:3, 3] - self.env.ee_pos()) * 1000)
            errs.append(e)
            print(f"[verify] q={np.round(qs, 2)}  |fk - sim| = {e:.2f} mm")
        print(f"[verify] max {max(errs):.2f} mm, mean {np.mean(errs):.2f} mm ->",
              "PASS" if max(errs) < 3 else "FAIL")

    # ------------------------------------------------------------------ episodes
    def execute(self, traj, tracker, hold_steps):
        """Run a joint trajectory, then hold its last point; metric every control step."""
        t, T = 0.0, traj.t[-1]
        while t <= T + hold_steps * self.ctrl_dt:
            self.step(traj.sample(t))
            t += self.ctrl_dt
            occ, _ = self.metric_now()
            if tracker.update(occ):
                return "success"
            if M["contact_is_failure"] and self.env.in_contact():
                self.last_contact = {"t": round(t, 2), "label": traj.labels[min(int(np.searchsorted(traj.t, t)), len(traj.labels) - 1)],
                                     "q": self.env.joint_pos().round(3).tolist(), "links": self.env.contact_report()}
                print("[contact]", json.dumps(self.last_contact))
                return "contact"
        return "done"

    def episode(self, k, agent):
        ep_dir = OUT / f"ep{k:03d}"
        ep_dir.mkdir(exist_ok=True)
        seed = args.seed + k
        q = self.reset(seed)
        self.q_home = q.copy()                # known-clear pose: free moves may route through it
        tracker = SuccessTracker(M["visible_px"], M["sustain_steps"])
        history, rec = [], {"seed": seed, "decisions": []}
        occ0, area = self.metric_now()
        tracker.update(occ0)
        rec["occluded_start"], rec["mask_area"] = occ0, area
        rec["target"] = self.env.target()[1]
        rec["plant_deviation_after_reset_m"] = round(self.env.plant_deviation(), 4)
        rec["contact_at_start"] = self.env.in_contact()
        self.last_contact = None
        outcome = "out_of_decisions"
        ac = CFG["agent"]
        for n in range(ac["max_actions"]):
            cam, rgb, depth, box, color = self.vlm_view()
            shown = annotate_target(rgb, box)
            pmask = plant_mask(depth, M["plant_depth_band"])
            if isinstance(agent, HeuristicAgent):
                agent.set_context(pmask, box)
            t0 = time.time()
            try:
                d = agent.decide(shown, color, history)
            except Exception as e:  # VLM down / invalid replies
                rec["decisions"].append({"error": repr(e)})
                outcome = "agent_error"
                break
            dec = {"n": n, "point_px": [round(v, 1) for v in d.point_px], "direction": d.direction,
                   "reasoning": d.reasoning, "latency_s": round(time.time() - t0, 2)}
            snapped = snap_to_mask(pmask, *d.point_px, ac["point_snap_px"])
            h = {"direction": d.direction, "point_shown": [int(v) for v in d.point_px]}
            if snapped is None:
                h["outcome"] = "that point is not on the plant (nothing to push there)"
                dec["outcome"] = h["outcome"]
            else:
                z = median_depth(depth, pmask, *snapped)
                P = cam.deproject(snapped[0], snapped[1], z)
                dvec = cam.image_direction(d.direction)
                try:
                    wps = sweep(P, dvec, cam.R[:, 2], self.env.fruit_positions(), CFG["scene"]["fruit_radius"], ac)
                    traj = plan_sweep(self.arm, self.env.joint_pos(), wps[:3], ac["waypoint_spacing"],
                                      CFG["robot"]["max_joint_vel"], min_z=ac["min_z"], via=self.q_home)
                    res = self.execute(traj, tracker, hold_steps=M["sustain_steps"] + 5)
                    occ, _ = self.metric_now()
                    dec.update(point_3d=P.round(4).tolist(), snapped_px=list(snapped), result=res,
                               occluded_after=occ, traj_s=round(float(traj.t[-1]), 2))
                    h["outcome"] = (f"fruit {100 * (1 - occ / area):.0f}% visible while holding"
                                    if res != "contact" else "the arm touched something rigid (failure)")
                    if res in ("success", "contact"):
                        outcome = res
                    else:  # release: retreat toward the camera before the next decision
                        back = plan_cartesian(self.arm, self.env.joint_pos(), wps[3:], ac["waypoint_spacing"],
                                              CFG["robot"]["max_joint_vel"])
                        if self.execute(back, tracker, hold_steps=3) == "contact":
                            outcome = "contact"
                except (PrimitiveError, IKError) as e:
                    h["outcome"], h["blocked"] = f"not executable: {e}", True
                    dec["outcome"] = h["outcome"]
            history.append(h)
            rec["decisions"].append(dec)
            Image.fromarray(draw_decision(shown, d, h.get("outcome", ""))).save(ep_dir / f"decision{n}.png")
            print(f"[ep {k}] decision {n}: {d.direction} @ {dec['point_px']} -> {h.get('outcome')}")
            if outcome in ("success", "contact"):
                break
        _, rgb_end, _, box_end, _ = self.vlm_view()
        Image.fromarray(annotate_target(rgb_end, box_end)).save(ep_dir / "final.png")
        rec.update(outcome=outcome, success=outcome == "success", occluded_history=tracker.history,
                   contact=self.last_contact)
        (ep_dir / "episode.json").write_text(json.dumps(rec, indent=2))
        return rec

    def episodes(self):
        agent = VLMAgent(CFG["vlm"]) if args.agent == "vlm" else HeuristicAgent()
        recs = []
        for k in range(args.episodes):
            r = self.episode(k, agent)
            recs.append(r)
            print(f"[ep {k}] {r['outcome']}  decisions={len(r['decisions'])}  target={r['target']}  "
                  f"plant_dev_after_reset={r['plant_deviation_after_reset_m']} m  contact_at_start={r['contact_at_start']}")
        summary = {"agent": args.agent, "seed": args.seed, "model": CFG["vlm"]["model"] if args.agent == "vlm" else None,
                   "episodes": len(recs), "success_rate": float(np.mean([r["success"] for r in recs])),
                   "outcomes": {o: sum(r["outcome"] == o for r in recs) for o in {r["outcome"] for r in recs}},
                   "mean_decisions": float(np.mean([len(r["decisions"]) for r in recs]))}
        (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
        print("[summary]", json.dumps(summary))


if __name__ == "__main__":
    rc = 1
    try:
        r = Runner()
        getattr(r, args.mode)()
        rc = 0
    except Exception:
        traceback.print_exc()
    finally:
        close_and_exit(app, rc)
