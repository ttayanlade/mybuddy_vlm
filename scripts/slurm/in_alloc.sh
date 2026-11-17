#!/usr/bin/env bash
# Run scripts/slurm/run.sbatch as a step inside an existing allocation without ending it:
#   scripts/slurm/in_alloc.sh <jobid> preview
#   scripts/slurm/in_alloc.sh <jobid> episodes --agent vlm --episodes 5
# Log: runs/<mode>-<RUN_ID>.log ; outputs: runs/<mode>/<RUN_ID>/
# Paths are passed explicitly: OnDemand allocations do not forward the caller env to steps.
set -euo pipefail
JOB=$1; MODE=$2; shift 2
cd "$(dirname "$0")/../.."
PROJ=$PWD; RUN_ID="${JOB}-$(date +%m%d-%H%M%S)"
echo "RUN_ID=$RUN_ID  log: runs/$MODE-$RUN_ID.log  out: runs/$MODE/$RUN_ID"
exec srun --jobid="$JOB" --overlap -n1 -c "${CPUS:-14}" /bin/bash -lc \
  "cd '$PROJ' && PROJ='$PROJ' RUN_ID='$RUN_ID' VLM_BASE_URL='${VLM_BASE_URL:-}' VLM_MODEL='${VLM_MODEL:-}' bash scripts/slurm/run.sbatch $MODE $*" \
  > "runs/$MODE-$RUN_ID.log" 2>&1
