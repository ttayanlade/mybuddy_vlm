#!/usr/bin/env bash
#SBATCH --job-name=ftf-vlm
#SBATCH --partition=nova
#SBATCH --gres=gpu:a40:1          # RTX-class GPU needed for Isaac cameras: a40 | l40s | rtx_pro_6000 | rtx_6000
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=runs/slurm-%j.out
#
#   sbatch scripts/slurm/run.sbatch preview
#   sbatch scripts/slurm/run.sbatch verify
#   sbatch scripts/slurm/run.sbatch episodes --agent heuristic --episodes 10
#   sbatch scripts/slurm/run.sbatch episodes --agent vlm --episodes 10
# Inside an existing allocation: scripts/slurm/in_alloc.sh <jobid> <same args>
#
# --agent vlm: uses $VLM_BASE_URL if set, else runs/vlm_endpoint.txt (vlm_server.sbatch),
# else starts vLLM on THIS GPU next to Isaac Sim (fits on 48 GB a40/l40s).
set -euo pipefail
MODE=${1:?mode: preview|verify|episodes}; shift
PROJ=${PROJ:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}}
cd "$PROJ"
RUN_ID=${RUN_ID:-${SLURM_JOB_ID:-local}-$(date +%m%d-%H%M%S)}
mkdir -p runs .isaac_home .hf_cache .isaac_cache/kit_cache .isaac_cache/kit_data .isaac_cache/kit_logs
module load apptainer
SIF=${SIF_DIR:-$PROJ/containers}
echo "run $RUN_ID on $(hostname): $(nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader | head -1)"

VLM_PID=
if [[ " $* " == *" --agent vlm "* ]]; then
  if [[ -z ${VLM_BASE_URL:-} && -f runs/vlm_endpoint.txt ]]; then read -r VLM_BASE_URL VLM_MODEL_F < runs/vlm_endpoint.txt; VLM_MODEL=${VLM_MODEL:-$VLM_MODEL_F}; fi
  if [[ -z ${VLM_BASE_URL:-} ]]; then
    PORT=$(( 18000 + RANDOM % 1000 ))
    VLM_MODEL=${VLM_MODEL:-Qwen/Qwen3-VL-8B-Instruct}; VLM_BASE_URL=http://127.0.0.1:$PORT/v1
    apptainer exec --nv --cleanenv --bind "$PROJ" --env HF_HOME="$PROJ/.hf_cache",HF_HUB_OFFLINE=1 "$SIF/vllm.sif" \
      vllm serve "$VLM_MODEL" --host 127.0.0.1 --port $PORT --max-model-len 8192 \
        --gpu-memory-utilization ${VLM_GPU_MEM_UTIL:-0.45} --limit-mm-per-prompt '{"image": 1}' \
        > "runs/vllm-$RUN_ID.log" 2>&1 &
    VLM_PID=$!
    trap 'kill $VLM_PID 2>/dev/null || true' EXIT
  fi
  echo "VLM ${VLM_MODEL:-<config>} at $VLM_BASE_URL"
  t0=$SECONDS
  until curl -sf "$VLM_BASE_URL/models" >/dev/null; do
    (( SECONDS - t0 > 900 )) && { echo "VLM not reachable"; exit 1; }
    [[ -n $VLM_PID ]] && ! kill -0 $VLM_PID 2>/dev/null && { echo "vLLM died:"; tail -20 "runs/vllm-$RUN_ID.log"; exit 1; }
    sleep 10
  done
  echo "VLM ready after $(( SECONDS - t0 ))s"
fi

apptainer exec --nv --cleanenv --writable-tmpfs --home "$PROJ/.isaac_home" --bind "$PROJ" --pwd "$PROJ" \
  --bind "$PROJ/.isaac_cache/kit_cache:/isaac-sim/kit/cache" \
  --bind "$PROJ/.isaac_cache/kit_data:/isaac-sim/kit/data" \
  --bind "$PROJ/.isaac_cache/kit_logs:/isaac-sim/kit/logs" \
  --env ACCEPT_EULA=Y,PRIVACY_CONSENT=Y,OMNI_KIT_ACCEPT_EULA=YES,PYTHONUNBUFFERED=1 \
  --env "VLM_BASE_URL=${VLM_BASE_URL:-},VLM_MODEL=${VLM_MODEL:-}" \
  "$SIF/isaaclab.sif" /workspace/isaaclab/isaaclab.sh -p sim/run.py "$MODE" \
    --headless --enable_cameras --device cuda:0 --out "runs/$MODE/$RUN_ID" "$@"
echo "outputs: runs/$MODE/$RUN_ID"
