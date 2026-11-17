Copied from https://github.com/elephantrobotics/mycobot_ros (commit c0d6fbe, BSD-3-Clause, see
../../urdf/mybuddy/LICENSE.elephantrobotics). Reference only — not imported by the pipeline.

- firefighter.srdf    MoveIt groups: left_arm = chain base->link6_L (includes waist joint base_link1).
- slider_control.py   Shows URDF joint radians go to the servos UNCHANGED: mb.send_radians(1, q_left, speed),
                      ids 1=left arm, 2=right arm, 3=waist; mb.set_fresh_mode(1) for streaming.
                      => real_joint_signs = 1, real_joint_offsets_deg = 0 in config/pipeline.yaml.
- test_control.py     Current pymycobot MyBuddy API: get_angles(id), send_angles(id, deg[6], speed).

The URDF/meshes in ../../urdf/mybuddy are identical to mycobot_description/urdf/mybuddy upstream.
Notes on that URDF (fixed by sim/extract_single_arm.py, upstream left untouched):
- no <inertial> on any link; all joint velocity limits are 0; effort 1000
- link4_L / link5_L collision origins disagree with their visual origins
- the 280 URDFs' inertials are placeholders (0.1 kg, I=0.03 kg m^2) and are not used
