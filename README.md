# MyBuddy single-arm zero-shot VLM search pipeline

Find a hidden object in clutter ("find the keys under the towel") with one 6-DoF arm of an
Elephant Robotics MyBuddy 280. The pipeline is decoupled:

```
 eye-to-hand RGB ──► VLM (GPT-4o / local VILA)  ──► {"target_obstacle", "primitive_action", "reasoning"}
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
| `sim/extract_single_arm.py` | step 1a: cut one 6-DoF arm out of the dual-arm MyBuddy URDF (pure Python) |
| `sim/import_urdf.py` | step 1b: Isaac Sim URDF importer → USD, **ArticulationRoot on base**, compliant drives |
| `sim/mybuddy_env.py` | step 1c: Isaac Lab scene (table, clutter, D435 eye-to-hand + eye-in-hand cameras) + ROS 2 bridge |
| `sim/ros2_sim_bridge.py` | makes the sim look like the real robot on ROS 2 (images, joint states, trajectories, /clock) |
| `sim/verify_kinematics.py` | checks that the DH model used for IK matches the imported URDF |
| `mybuddy_vlm/vlm_client.py` | step 2: VLM prompt + strict JSON validation with retry |
| `mybuddy_vlm/grounding.py` | step 3: Grounding DINO + SAM 2, 2D → 3D deprojection to base frame |
| `mybuddy_vlm/primitives.py` | step 4: push_left / push_right / pull / lift → Cartesian waypoints, workspace checks |
| `mybuddy_vlm/kinematics.py` | myCobot-280 DH FK, tool-down DLS IK, joint trajectory timing |
| `mybuddy_vlm/ros_interface.py` | step 5: ROS 2 I/O (namespace-agnostic), optional MoveIt 2 IK backend |
| `mybuddy_vlm/reasoning_node.py` | the ~1 Hz reason → ground → act loop, `--sim` / `--real` / `--dry-run` |
| `real/mybuddy_driver_node.py` | JointTrajectory → pymycobot `MyBuddy.send_angles` for the physical arm |
| `tests/` | 27 unit tests covering everything that needs no GPU/ROS/Isaac |

## Status / what has been verified

* **Verified here, CPU only:** `pytest -q` passes 27/27. That covers deprojection and extrinsics, a
  synthetic-depth 2D→3D recovery (±6 mm), VLM JSON validation, IK/FK, every primitive's geometry, and
  feasible joint trajectories for all four primitives on the default towel layout.
  The URDF extractor was tested on a synthetic dual-arm URDF.
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
export OPENAI_API_KEY=...           # or set vlm.provider: openai_compatible + vlm.base_url for local VILA
```

### Step 1: robot asset

```bash
# Get the MyBuddy description (Elephant Robotics mycobot_ros / mycobot_ros2 repo, mybuddy URDF)
python sim/extract_single_arm.py --urdf <path>/mybuddy.urdf --list          # find link names
python sim/extract_single_arm.py --urdf <path>/mybuddy.urdf \
    --root <left_arm_shoulder_link> --tip <left_arm_6th_link> \
    --package mycobot_description=<path>/mycobot_description \
    --rename-joints auto --out assets/urdf/mybuddy_left_arm.urdf
# then set robot.ee_link in the config to the flange link printed by --list
${ISAACLAB_PATH}/isaaclab.sh -p sim/import_urdf.py --headless                # -> assets/usd/
${ISAACLAB_PATH}/isaaclab.sh -p sim/verify_kinematics.py --headless          # DH vs URDF check
```

A myCobot 280 URDF works the same way: it is kinematically identical and already single-arm.

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

sbatch scripts/slurm/sim_pipeline.sbatch test       # unit tests in the reasoning container
sbatch scripts/slurm/sim_pipeline.sbatch assets     # URDF -> USD, DH-vs-URDF check
sbatch scripts/slurm/sim_pipeline.sbatch scene      # sim smoke test: topics, camera Hz, joint_states
OPENAI_API_KEY=... sbatch scripts/slurm/sim_pipeline.sbatch dryrun
OPENAI_API_KEY=... sbatch scripts/slurm/sim_pipeline.sbatch pipeline
```

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
* **Reach dead zone.** The 280's wrist offsets make tool-down poses unreachable within about 13 cm of
  the base axis at table height. `workspace.min_radius: 0.145` is enforced, and pull/lift adapt
  (shortened drags, choosing a reachable edge). Unreachable requests are refused, and the reason goes
  back into the VLM prompt history for the next decision.
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
