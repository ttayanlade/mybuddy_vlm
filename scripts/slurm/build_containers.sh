#!/usr/bin/env bash
#SBATCH --job-name=mybuddy-build
#SBATCH --partition=nova
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=runs/build-%j.out
#
# Fetch/build the Apptainer images used on the Nova cluster. Pass the ones you want
# (default: all), e.g.
#   sbatch scripts/slurm/build_containers.sh isaaclab        # only what assets/render/scene-sim need
#   sbatch scripts/slurm/build_containers.sh vllm
#
#   isaaclab  Isaac Lab 2.3.2 / Isaac Sim 5.1 (+ bundled ROS 2 libs) from NGC   ~25 GB
#   vllm      vLLM OpenAI-compatible server for the open-source VLM              ~10 GB
#
# Isaac Sim needs glibc >= 2.35 for the pip wheels; RHEL9 has 2.34, hence containers.
# Public NGC images pull anonymously; if NGC asks for auth:
#   export APPTAINER_DOCKER_USERNAME='$oauthtoken' APPTAINER_DOCKER_PASSWORD=<NGC key>
set -euo pipefail
PROJ=${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}
cd "$PROJ"
mkdir -p runs
module load apptainer

SIF_DIR=${SIF_DIR:-$PROJ/containers}
ISAACLAB_IMAGE=${ISAACLAB_IMAGE:-docker://nvcr.io/nvidia/isaac-lab:2.3.2}   # newest 2.x on NGC (3.0 changes APIs)
VLLM_IMAGE=${VLLM_IMAGE:-docker://vllm/vllm-openai:v0.11.0}                 # Qwen3-VL needs vLLM >= 0.11
# Always use project-local dirs: Nova presets APPTAINER_TMPDIR=/tmp, which is small and shared.
export APPTAINER_CACHEDIR=${BUILD_CACHEDIR:-$PROJ/.apptainer_cache}
export APPTAINER_TMPDIR=${BUILD_TMPDIR:-$PROJ/.apptainer_tmp/${SLURM_JOB_ID:-$$}}
mkdir -p "$SIF_DIR" "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"
WANT=("$@"); [[ ${#WANT[@]} -eq 0 ]] && WANT=(isaaclab vllm)

for w in "${WANT[@]}"; do
  t0=$SECONDS
  case "$w" in
    isaaclab) [[ -f $SIF_DIR/isaaclab.sif ]] || apptainer pull "$SIF_DIR/isaaclab.sif" "$ISAACLAB_IMAGE" ;;
    vllm)     [[ -f $SIF_DIR/vllm.sif ]]     || apptainer pull "$SIF_DIR/vllm.sif" "$VLLM_IMAGE" ;;
    *) echo "unknown image '$w'"; exit 1 ;;
  esac
  echo ">> $w done in $(( SECONDS - t0 ))s"
done
# only ever delete the per-job temp dir we created inside the project
[[ $APPTAINER_TMPDIR == "$PROJ"/.apptainer_tmp/* ]] && rm -rf "$APPTAINER_TMPDIR"
ls -lh "$SIF_DIR"/*.sif
df -h "$PROJ" | tail -1
