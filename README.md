# Find the Fruit with a VLM (MyBuddy, Isaac Lab, no RL)

The **task, scene, robot and success metric** are the ones from
[Find the Fruit](https://arxiv.org/abs/2505.16547) (Subedi, Yang, Jha, Sarkar, 2025;
code: `github.com/nitesh-subedi/Find_the_Fruit`). A MyBuddy left arm must push a deformable
plant's stems and leaves so that a hidden target fruit becomes visible to the camera.

**The method is different.** Find the Fruit trains a PPO visuomotor policy. Here, nothing is
trained: a vision-language model looks at the camera image and decides *where on the plant
to push and in which direction*. A sweep primitive then executes that decision with IK.

```
camera image (target fruit boxed) ──► VLM ──► {"point": [x, y], "direction": "left|right|up|down"}
         │                                                │
         └─ depth ──► point snapped onto the plant ──► 3D contact point; direction mapped through the camera pose
                                                          │
                     sweep: approach from the camera side → start behind the point → push through it → HOLD
                                                          │
            30 Hz joint trajectory (IK on the USD kinematics) ──► Find the Fruit's occlusion metric every step
```

## What is copied from Find the Fruit and what is ours

| | From Find the Fruit (unchanged) | Ours |
|---|---|---|
| Robot | `mybuddy_rotate.usd`: full MyBuddy; waist + left j1–j5 controlled; stiff drives (kp 800, kd 4), weightless robot | Kinematics read from the same USD (`mybuddy_vlm/kinematics.py`), position-only IK that avoids self-collision (below) |
| Scene | Deformable plant `plant_v21.usd` with ground/stalk attachments, background, 2 fruit spheres with x-jitter and a random target, their initial pose and j2 randomisation | — |
| Camera | Their pose and optics (looking forward and ~25° down at 0.5 m height), 256×256 | A second camera at the same pose, 768×768, for the VLM |
| Success | Target 30×30 px mask, plant = depth in [0.1, 0.4] m, success = ≤160 occluded px for 10 consecutive steps; rigid arm-link contact = failure | Used as **evaluation only** |
| Decisions | PPO policy, 30 Hz joint deltas | VLM (Qwen3-VL-8B on vLLM by default) choosing one push per decision, ≤5 decisions per episode; a geometric heuristic baseline |

`sim/ftf_env.py` is adapted from their `env_cfg.py`, `find_the_fruit_env.py` and
`scene_builder.py`: actions became absolute joint targets, there is no auto-reset, the VLM
camera was added, debug markers were turned off, and the API names follow Isaac Lab 2.3.


## Notes

- **The plant is elastic.** A stem pushed aside springs back once released, which is why the
  success rule asks for 10 consecutive visible steps. The sweep therefore ends by *holding*
  the push. It retreats only if the fruit isn't visible yet.
- **The metric counts anything at plant depth as plant**, the arm included, exactly as in
  Find the Fruit. A push must therefore not leave the arm in front of the target.
- **Contact:** as in their env, rigid contact on an arm link (another fruit, the ground,
  self-collision) fails the episode. PhysX doesn't report deformable-body contacts, so
  touching the plant is allowed.
