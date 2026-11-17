#!/usr/bin/env bash
#SBATCH --job-name=mybuddy-vlm
#SBATCH --partition=nova
#SBATCH --gres=gpu:a40:1          # RTX-capable: a40 | l40s | rtx_6000. NOT a100/h100/v100 (no RT cores -> no cameras)
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=runs/slurm-%j.out
#
# One GPU node runs both containers; they talk ROS 2 over shared memory/localhost.
#
#   sbatch scripts/slurm/sim_pipeline.sbatch assets     # URDF -> USD + DH/URDF check
#   sbatch scripts/slurm/sim_pipeline.sbatch scene      # sim only, 10 min smoke test (topics + a frame dump)
#   sbatch scripts/slurm/sim_pipeline.sbatch pipeline   # sim + reasoning loop   (needs OPENAI_API_KEY in env)
#   sbatch scripts/slurm/sim_pipeline.sbatch dryrun     # sim + reasoning, never moves the arm
#   sbatch scripts/slurm/sim_pipeline.sbatch test       # unit tests inside the reasoning container
set -euo pipefail
MODE=${1:-pipeline}
PROJ=${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}
cd "$PROJ"
mkdir -p runs .isaac_home .hf_cache
module load apptainer

SIF_DIR=${SIF_DIR:-$PROJ/containers}
SIM_SIF=$SIF_DIR/isaaclab.sif
ROS_SIF=$SIF_DIR/ros2_reasoner.sif

# Isolate this job's ROS graph from other jobs/users sharing the node.
export ROS_DOMAIN_ID=$(( ${SLURM_JOB_ID:-0} % 200 + 10 ))
export ROS_LOCALHOST_ONLY=1
export HF_HOME=$PROJ/.hf_cache
echo "job ${SLURM_JOB_ID:-local} on $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1), ROS_DOMAIN_ID=$ROS_DOMAIN_ID"

# Isaac Sim writes caches under $HOME and /isaac-sim/kit; redirect both into the project.
SIM_RUN=(apptainer exec --nv --cleanenv --home "$PROJ/.isaac_home" --bind "$PROJ" --pwd "$PROJ"
         --env ACCEPT_EULA=Y,PRIVACY_CONSENT=Y,OMNI_KIT_ACCEPT_EULA=YES
         --env ROS_DOMAIN_ID=$ROS_DOMAIN_ID,ROS_LOCALHOST_ONLY=1
         --env ISAACLAB_PATH=/workspace/isaaclab,ISAACSIM_PATH=/isaac-sim
         "$SIM_SIF")
# Clean env so host conda/python on PATH can't shadow the container's; the API key
# is forwarded via APPTAINERENV_ (works with --cleanenv, never printed).
[[ -n ${OPENAI_API_KEY:-} ]] && export APPTAINERENV_OPENAI_API_KEY=$OPENAI_API_KEY
ROS_RUN=(apptainer exec --nv --cleanenv --bind "$PROJ" --pwd "$PROJ"
         --env ROS_DOMAIN_ID=$ROS_DOMAIN_ID,ROS_LOCALHOST_ONLY=1,HF_HOME=$HF_HOME
         "$ROS_SIF")
isaaclab() { "${SIM_RUN[@]}" /workspace/isaaclab/isaaclab.sh -p "$@"; }

wait_for_topic() {  # $1 topic, $2 timeout s  (first Isaac start compiles shaders: several minutes)
  local t0=$SECONDS
  until "${ROS_RUN[@]}" bash -c "ros2 topic list 2>/dev/null | grep -qx '$1'"; do
    (( SECONDS - t0 > $2 )) && { echo "timeout waiting for $1"; return 1; }
    sleep 10
  done
  echo "$1 is up after $(( SECONDS - t0 ))s"
}

start_sim() {
  "${SIM_RUN[@]}" bash scripts/run_sim_stack.sh sim --headless > "runs/sim-${SLURM_JOB_ID:-local}.log" 2>&1 &
  SIM_PID=$!
  trap 'kill $SIM_PID 2>/dev/null || true' EXIT
  wait_for_topic /sim/camera/color/image_raw 1200
}

case "$MODE" in
  assets)
    isaaclab sim/import_urdf.py --headless
    isaaclab sim/verify_kinematics.py --headless ;;
  scene)
    start_sim
    "${ROS_RUN[@]}" bash -c "ros2 topic list; ros2 topic hz /sim/camera/color/image_raw --window 30 & sleep 20; kill %1;
                             timeout 10 ros2 topic echo --once /sim/mybuddy/joint_states"
    sleep 540 ;;
  pipeline)
    start_sim
    "${ROS_RUN[@]}" python3 -m mybuddy_vlm.reasoning_node --sim ;;
  dryrun)
    start_sim
    "${ROS_RUN[@]}" python3 -m mybuddy_vlm.reasoning_node --sim --dry-run ;;
  test)
    "${ROS_RUN[@]}" python3 -m pytest -q ;;
  *) echo "unknown mode $MODE"; exit 1 ;;
esac
