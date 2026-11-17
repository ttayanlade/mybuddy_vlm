#!/usr/bin/env bash
# Physical MyBuddy + RealSense D435. Run each part in its own terminal.
#   scripts/run_real_stack.sh camera   # D435, colour+ALIGNED depth on the topics the node expects
#   scripts/run_real_stack.sh driver   # pymycobot <-> ROS 2 trajectory driver (left arm)
#   scripts/run_real_stack.sh reason   # identical reasoning code, real namespace
set -euo pipefail
cd "$(dirname "$0")/.."
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-42}
case "${1:-}" in
  camera)
    # The node deprojects COLOUR pixels, so depth must be aligned to colour.
    # realsense2_camera publishes that on aligned_depth_to_color; relay it onto the
    # spec'd /camera/depth/image_rect_raw so the reasoning code is unchanged.
    ros2 launch realsense2_camera rs_launch.py camera_namespace:=/ camera_name:=camera \
        align_depth.enable:=true rgb_camera.color_profile:=640x480x30 \
        depth_module.depth_profile:=640x480x30 &
    sleep 3
    ros2 run topic_tools relay /camera/aligned_depth_to_color/image_raw /camera/depth/image_rect_raw ;;
  driver)
    python real/mybuddy_driver_node.py --port "${MYBUDDY_PORT:-/dev/ttyACM0}" --arm left --speed 25 ;;
  reason)
    python -m mybuddy_vlm.reasoning_node --real "${@:2}" ;;
  *) echo "usage: $0 {camera|driver|reason} [extra args]"; exit 1 ;;
esac
