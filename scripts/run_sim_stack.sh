#!/usr/bin/env bash
# Terminal 1: Isaac Sim scene + ROS 2 bridge.  Terminal 2: reasoning node.
# Usage: scripts/run_sim_stack.sh sim      (needs ISAACLAB_PATH, ISAACSIM_PATH)
#        scripts/run_sim_stack.sh reason   (needs a sourced ROS 2 + OPENAI_API_KEY)
set -euo pipefail
cd "$(dirname "$0")/.."
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-42}
case "${1:-}" in
  sim)
    # Isaac Sim's bundled ROS 2 (no system ROS needed inside the sim process).
    export ROS_DISTRO=${ROS_DISTRO:-humble}
    export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
    export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}:${ISAACSIM_PATH}/exts/isaacsim.ros2.bridge/${ROS_DISTRO}/lib
    "${ISAACLAB_PATH}/isaaclab.sh" -p sim/mybuddy_env.py --device cuda:0 "${@:2}" ;;
  reason)
    python -m mybuddy_vlm.reasoning_node --sim "${@:2}" ;;
  reset)
    ros2 topic pub --once /sim/sim/reset std_msgs/msg/Empty "{}" ;;
  *) echo "usage: $0 {sim|reason|reset} [extra args]"; exit 1 ;;
esac
