#!/usr/bin/env bash
# Fetch/build the two Apptainer images used on the Nova cluster.
#   1. Isaac Lab (includes Isaac Sim + its bundled ROS 2 libs) from NGC
#   2. ROS 2 Humble + VLM/grounding deps for the reasoning node
#
# Run from a CPU job (pulling/building on the login node may be killed):
#   srun -p nova -c 8 --mem=32G -t 2:00:00 --pty scripts/slurm/build_containers.sh
#
# Isaac Sim needs glibc >= 2.35 for the pip wheels; RHEL9 has 2.34, hence containers.
# NGC may require a (free) API key:
#   export APPTAINER_DOCKER_USERNAME='$oauthtoken' APPTAINER_DOCKER_PASSWORD=<NGC key>
set -euo pipefail
cd "$(dirname "$0")/../.."
PROJ=$(pwd)
module load apptainer

SIF_DIR=${SIF_DIR:-$PROJ/containers}
ISAACLAB_IMAGE=${ISAACLAB_IMAGE:-docker://nvcr.io/nvidia/isaac-lab:2.1.0}   # check NGC for newer tags
export APPTAINER_CACHEDIR=${APPTAINER_CACHEDIR:-$PROJ/.apptainer_cache}
export APPTAINER_TMPDIR=${APPTAINER_TMPDIR:-$PROJ/.apptainer_tmp}
mkdir -p "$SIF_DIR" "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"

if [[ ! -f $SIF_DIR/isaaclab.sif ]]; then
  echo ">> pulling $ISAACLAB_IMAGE (~20 GB)"
  apptainer pull "$SIF_DIR/isaaclab.sif" "$ISAACLAB_IMAGE"
fi
if [[ ! -f $SIF_DIR/ros2_reasoner.sif ]]; then
  echo ">> building ros2_reasoner.sif"
  apptainer build --fakeroot "$SIF_DIR/ros2_reasoner.sif" containers/ros2_reasoner.def
fi
rm -rf "$APPTAINER_TMPDIR"
ls -lh "$SIF_DIR"/*.sif
