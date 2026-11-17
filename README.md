# MyBuddy single-arm zero-shot VLM search pipeline

Find a hidden object in clutter ("find the keys under the towel") with one 6-DoF arm of an
Elephant Robotics MyBuddy 280. The pipeline is decoupled:

```
 eye-to-hand RGB ──► open-source VLM (Qwen3-VL on vLLM; GPT-4o optional) ──► {"target_obstacle", "primitive_action", "reasoning"}
                                                            │
 RGB + aligned depth ──► Grounding DINO ─► SAM 2 mask ─► mask centre + depth ─► (x,y,z) in base frame
                                                            │
                     parameterised primitive (push_left/right, pull, lift) ─► tool-down IK ─► JointTrajectory
                                                            │
                         ROS 2  <ns>/mybuddy/joint_trajectory_controller/joint_cmd
                            ├── --sim : Isaac Sim bridge  (ns /sim, /clock)
                            └── --real: pymycobot driver  (ns /)
```

Everything is configured in **`config/pipeline.yaml`**: intrinsics, extrinsics, drive gains,
masses, topics, and workspace. Isaac Lab, the reasoning node, and the hardware driver all read
this file, so they can't drift apart.

## Layout

| Path | What |
|---|---|
| `config/pipeline.yaml` | the one config file |
| `urdf/mybuddy/` | official MyBuddy URDF + meshes (Elephant Robotics, BSD-3, unmodified) |
| `assets/urdf/mybuddy_left_arm.urdf` | generated single left arm, upright, physics-ready (committed) |
| `reference/elephantrobotics/` | upstream SRDF + driver scripts kept for reference (servo API, joint convention) |
| `sim/extract_single_arm.py` | step 1a: cut the left arm out of the MyBuddy URDF and fix it for physics (pure Python) |
| `sim/mesh_utils.py` | COLLADA/STL bounds (used for inertials) |
| `sim/import_urdf.py` | step 1b: Isaac Sim URDF importer → USD, **ArticulationRoot on base**, compliant drives |
| `sim/mybuddy_env.py` | step 1c: Isaac Lab scene (table, clutter, D435 eye-to-hand + eye-in-hand cameras) + ROS 2 bridge |
| `sim/ros2_sim_bridge.py` | makes the sim look like the real robot on ROS 2 (images, joint states, trajectories, /clock) |
| `sim/verify_kinematics.py` | checks that the numpy kinematics match what PhysX simulates after import |
| `mybuddy_vlm/vlm_client.py` | step 2: VLM prompt + strict JSON validation with retry |
| `mybuddy_vlm/grounding.py` | step 3: Grounding DINO + SAM 2, 2D → 3D deprojection to base frame |
| `mybuddy_vlm/primitives.py` | step 4: push_left / push_right / pull / lift → Cartesian waypoints, workspace checks |
| `mybuddy_vlm/kinematics.py` | FK from the URDF chain, tool-down DLS IK (analytic Jacobian + seed bank), trajectory timing |
| `mybuddy_vlm/ros_interface.py` | step 5: ROS 2 I/O (namespace-agnostic), optional MoveIt 2 IK backend |
| `mybuddy_vlm/reasoning_node.py` | the ~1 Hz reason → ground → act loop, `--sim` / `--real` / `--dry-run` |
| `real/mybuddy_driver_node.py` | JointTrajectory → pymycobot `MyBuddy.send_angles` for the physical arm |
| `scripts/slurm/` | Apptainer builds, model download, vLLM server job, sim+reasoning job |
| `tests/` | 35 unit tests covering everything that needs no GPU/ROS/Isaac |

## Status / what has been verified

* **Verified here, CPU only:** `pytest -q` passes 35/35. That covers deprojection and extrinsics, a
  synthetic-depth 2D→3D recovery (±6 mm), VLM JSON validation (including reasoning-model output and
  the request sent to vLLM, against a fake server), FK/IK on the real MyBuddy arm geometry, every
  primitive's geometry, and feasible joint trajectories for all four primitives on the default towel
  layout. The extracted MyBuddy URDF is checked for limits, inertials and resolvable meshes.
* **Not run yet (no GPU, Isaac Sim, or ROS 2 on this node):** `sim/*.py`, `ros_interface.py`,
  `reasoning_node.py`, `real/*.py`. They pass a syntax check only. Expect small API fixes for your
  exact Isaac Lab / Isaac Sim version. The code targets **Isaac Lab 2.x on Isaac Sim 4.5/5.0**
  (`isaaclab.*` namespace, `isaacsim.asset.importer.urdf`).

## Setup

```bash
cd /work/mech-ai/ayanlade/mybuddy_vlm
source .venv/bin/activate          # CPU venv for tests (numpy, pyyaml, pytest)
pytest -q
```

On the GPU / ROS 2 machine:

```bash
pip install -r requirements.txt     # inside the ROS 2 python env
vllm serve Qwen/Qwen3-VL-8B-Instruct --port 8000 --limit-mm-per-prompt '{"image": 1}'   # open-source VLM
# (or: vlm.provider: openai + export OPENAI_API_KEY=... to use GPT-4o instead)
```

### Models (all open source by default)

| Role | Default | Alternatives |
|---|---|---|
| Spatial reasoner (VLM) | `Qwen/Qwen3-VL-8B-Instruct` via vLLM | Qwen3-VL-32B, Qwen2.5-VL-7B, InternVL3-8B; **reasoning:** Qwen3-VL-8B-Thinking, NVIDIA Cosmos-Reason1-7B |
| Open-vocabulary detector | Grounding DINO (`IDEA-Research/grounding-dino-base`) | OWLv2, Florence-2 |
| Segmenter | SAM 2.1 (`facebook/sam2.1-hiera-large`) | SAM 2.1 base/small for speed |

Model choice is in `config/pipeline.yaml` (`vlm:`), or set `VLM_MODEL` / `VLM_BASE_URL` per run. The
model's output is constrained to the JSON schema with vLLM structured outputs, so it can't produce a
malformed decision. Reasoning models think before answering, which constrained decoding would
suppress. For those, set `structured_output: none` and `max_tokens: 2048`: the `<think>` block is
stripped and the JSON is still validated strictly. The newest open models change quickly; anything
vLLM can serve with image input works without code changes.

### Step 1: robot asset

`assets/urdf/mybuddy_left_arm.urdf` is already generated and committed. To regenerate it:

```bash
python sim/extract_single_arm.py --urdf urdf/mybuddy/mybuddy.urdf \
  --root link1 --tip link6_L --mount-upright 0.062 --mount-yaw 3.14159 \
  --package mycobot_description=. --relative-meshes --rename-joints auto \
  --collision-from-visual --velocity-limit 2.8 --effort-limit 1.5 \
  --link-masses 0.13,0.12,0.10,0.07,0.06,0.03 --out assets/urdf/mybuddy_left_arm.urdf
${ISAACLAB_PATH}/isaaclab.sh -p sim/import_urdf.py --headless                # -> assets/usd/
${ISAACLAB_PATH}/isaaclab.sh -p sim/verify_kinematics.py --headless          # numpy FK vs PhysX
```

What the extraction fixes in the upstream URDF, which is built for display in RViz:
* **No `<inertial>` on any link.** Masses are added (0.51 kg for the arm), with box inertia from each
  link's mesh bounds.
* **All joint velocity limits are 0**, which would freeze PhysX joints. They become 2.8 rad/s.
  Effort becomes 1.5 N·m instead of 1000.
* **`link4_L` / `link5_L` collision shapes are offset from their visual meshes.** Collision is set to
  match the visual, since collision is what pushes the clutter.
* **The arm attaches to the side of the torso**, with joint 1 horizontal. It's stood upright on the
  table at the mesh-derived height (62 mm), and turned 180° (`--mount-yaw`): at q1 = 0 the MyBuddy
  arm bends toward −x, so without the turn the workspace in front of it would sit at joint 1's limit.
* **This arm is not a myCobot 280.** Its link lengths differ (110 / 97 / 35 / 35+37.5 mm), so IK
  uses the URDF chain directly (`robot.kinematics: urdf`). The myCobot DH table remains as an
  option if you swap in a myCobot 280.

### Run in simulation

```bash
scripts/run_sim_stack.sh sim                 # terminal 1 (Isaac Sim, GPU)
scripts/run_sim_stack.sh reason --dry-run    # terminal 2: perceive + plan only
scripts/run_sim_stack.sh reason              # terminal 2: closed loop
scripts/run_sim_stack.sh reset               # randomised scene reset
```

Every iteration writes an annotated image plus a JSON line (VLM output, grounded 3D point,
outcome) to `runs/<timestamp>-sim/`.

### Run on the Nova Slurm cluster

Isaac Sim needs an **RTX GPU (RT cores)** for camera rendering. On Nova that means the `nova`
partition's **A40** (`nova21-frost-8`), **L40S**, **Quadro RTX 6000** (`nova20-frost-3..7`), or
**RTX PRO 6000** (Blackwell, needs Isaac Sim 5.x) nodes. A100/H100/V100 nodes can't render the
cameras. RHEL 9's glibc 2.34 is too old for the Isaac Sim pip wheels, so the sim runs from
NVIDIA's Isaac Lab container via Apptainer.

```bash
# one-time (~25-30 GB in containers/; note /work/mech-ai is nearly full)
srun -p nova -c 8 --mem=32G -t 2:00:00 --pty scripts/slurm/build_containers.sh

scripts/slurm/download_models.sh                    # on the LOGIN node: VLM + GDINO + SAM2 weights (~19 GB)

sbatch scripts/slurm/sim_pipeline.sbatch test       # unit tests in the reasoning container
sbatch scripts/slurm/sim_pipeline.sbatch assets     # URDF -> USD, numpy-FK-vs-PhysX check
sbatch scripts/slurm/sim_pipeline.sbatch scene      # sim smoke test: topics, camera Hz, joint_states

sbatch scripts/slurm/vlm_server.sbatch              # open-source VLM on an A100 node (recommended)
sbatch scripts/slurm/sim_pipeline.sbatch dryrun     # picks up runs/vlm_endpoint.txt automatically
sbatch scripts/slurm/sim_pipeline.sbatch pipeline
```

The VLM doesn't need RT cores, so the recommended setup runs it on an A100/H100 node with
`vlm_server.sbatch`, keeping the RTX node for Isaac Sim alone. Without a server running,
`sim_pipeline.sbatch` starts vLLM on the same GPU as the sim. That fits on a 48 GB A40/L40S but
not on a 24 GB RTX 6000.

Both containers run on the same GPU node and talk ROS 2 over localhost. Each job gets its own
`ROS_DOMAIN_ID`, so jobs from other users on the same node don't interfere. Change the GPU type
with `sbatch --gres=gpu:l40s:1 ...`. Logs go to `runs/slurm-<job>.out` and `runs/sim-<job>.log`.

### Run on the real MyBuddy

```bash
scripts/run_real_stack.sh camera     # D435 colour + depth aligned to colour
scripts/run_real_stack.sh driver     # pymycobot driver, left arm
scripts/run_real_stack.sh reason --dry-run   # check the 3D points before allowing motion
scripts/run_real_stack.sh reason
```

Before the first real run:
1. **Hand-eye calibration.** Paste the camera→base pose into `ros.extrinsics.real`. This is the
   one number the sim can't give you.
2. **Joint convention.** Command `robot.home` in sim and on the robot. If the poses differ, set
   `robot.real_joint_signs` / `real_joint_offsets_deg`.
3. Start with `--speed 20` on the driver and keep the arm within reach of a hand.

## Design notes and deliberate deviations

* **`--sim` / `--real` changes only the namespace, the clock source, and the extrinsic source.** All
  topics are relative (`camera/color/image_raw`, …). In sim they resolve under `/sim`, so a live sim
  and a live robot can share a ROS domain. On the robot they resolve under `/`, which gives exactly
  `/camera/color/image_raw`, `/camera/depth/image_rect_raw`, and
  `/mybuddy/joint_trajectory_controller/joint_cmd`.
* **Depth must be aligned to colour.** On a real D435, `depth/image_rect_raw` is *not* aligned with
  the colour image, and deprojecting a colour-image mask with it lands centimetres off. The camera
  script relays `aligned_depth_to_color` onto the spec'd topic name. In sim, colour and depth come
  from one camera and are aligned by construction.
* **Mask centre → 3D.** The spec'd mask-centre deprojection is used for the target point. The centroid
  is snapped onto the mask (a mug handle or draped towel can put it off-object), and depth is a
  windowed median (D435 holes). The full mask point cloud also gives the object's top height and
  footprint, which the primitives need for standoff and contact height.
* **push_left = −Y in the base frame (as specified).** With the default camera, which faces the robot,
  image-left is also −Y, so the VLM's "left" agrees (a test checks this). If you move the camera, set
  `primitives.direction_frame: camera` to take left/right from the image.
* **Pushes start beside the object, not at its centre.** The literal spec would drive the tool into
  the object. Instead the tool goes to the +Y side (half-width + 3 cm standoff), lowers 1.5 cm below
  the top (never above 60% of the object's height, so mugs aren't tipped), then sweeps −Y.
* **Reach (measured on the MyBuddy URDF).** Tool-down reach is 6–28 cm at table height and about
  26 cm at 10 cm height. A small pocket near the base on the shoulder-offset side (r < 13 cm at −45°)
  can't be reached. The workspace uses `max_radius: 0.26` and `min_radius: 0.13`, and pull/lift adapt
  (shortened drags, choosing a reachable edge). Unreachable requests are refused, and the reason goes
  back into the VLM prompt history for the next decision.
* **Hardware joint convention.** Elephant's own MyBuddy driver sends these URDF joint angles to the
  servos unchanged (`send_radians`, `reference/elephantrobotics/slider_control.py`), so the driver
  does the same. Our base frame is joint 1's frame turned 180°, which only matters for the camera
  calibration.
* **Termination.** Each tick first runs Grounding DINO on the target name. The loop stops when the
  target is detected above `task.target_detect_threshold`. This keeps the VLM's JSON schema exactly
  as specified.
* **Compliance and masses.** Joint stiffness is 6–40 N·m/rad with effort limits near the servo stall
  torque, so the arm yields instead of fighting clutter. All props weigh 0.03–0.1 kg (payload is about
  250 g). A blocked push is detected (final joint error > 0.05 rad) and reported to the VLM.

## Known limitations

* **`lift` without a gripper is an approximation.** The tool slides under an edge and raises it, so
  the edge folds over. Set `robot.gripper.enabled: true` and list the gripper joints for a real
  pinch-lift-place.
* **The towel is a thin FEM soft body (semi-rigid), not cloth.** It needs GPU physics. Set
  `towel.mode: rigid_thin` to fall back to a thin rigid sheet. Isaac Sim 5.x's new deformable API
  may need changes to `make_object()`.
* **The keys are a stand-in:** a small metallic block. Swap in a keys USD under
  `scene.objects.keys` for realistic detection.
* `ik_backend: moveit` assumes a running `move_group` with a group named `arm`. It is untested.
  Upstream's MoveIt config is for the dual-arm robot, with group `left_arm` including the waist.
* **Link masses are estimates** (0.51 kg for the arm; Elephant publishes no per-link values). Their
  inertia is box-approximated from the meshes.
