#!/usr/bin/env bash
#SBATCH --job-name=mybuddy-vlm-server
#SBATCH --partition=nova
#SBATCH --gres=gpu:a100:1         # no RT cores needed for the VLM: a100 | h200 | v100(small models) all fine
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=runs/vlm-server-%j.out
#
# Open-source VLM behind vLLM's OpenAI-compatible API, on a non-RTX GPU node, so the
# RTX node only runs Isaac Sim. Writes its URL to runs/vlm_endpoint.txt, which
# sim_pipeline.sbatch picks up automatically.
#
#   sbatch scripts/slurm/vlm_server.sbatch                                   # Qwen3-VL-8B-Instruct
#   VLM_MODEL=Qwen/Qwen3-VL-32B-Instruct sbatch --gres=gpu:h200:1 scripts/slurm/vlm_server.sbatch
#   VLM_MODEL=nvidia/Cosmos-Reason1-7B VLLM_EXTRA="--reasoning-parser deepseek_r1" sbatch scripts/slurm/vlm_server.sbatch
set -euo pipefail
PROJ=${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}
cd "$PROJ"; mkdir -p runs .hf_cache
module load apptainer

MODEL=${VLM_MODEL:-Qwen/Qwen3-VL-8B-Instruct}
PORT=${VLM_PORT:-$(( 8000 + ${SLURM_JOB_ID:-0} % 1000 ))}
MEM_UTIL=${VLM_GPU_MEM_UTIL:-0.90}
VLLM_SIF=${SIF_DIR:-$PROJ/containers}/vllm.sif

echo "http://$(hostname -f):${PORT}/v1 ${MODEL}" > runs/vlm_endpoint.txt
trap 'rm -f runs/vlm_endpoint.txt' EXIT
echo "serving $MODEL on $(hostname -f):$PORT"

apptainer exec --nv --cleanenv --bind "$PROJ" --env HF_HOME="$PROJ/.hf_cache" "$VLLM_SIF" \
  vllm serve "$MODEL" --host 0.0.0.0 --port "$PORT" \
    --max-model-len 8192 --gpu-memory-utilization "$MEM_UTIL" \
    --limit-mm-per-prompt '{"image": 1}' ${VLLM_EXTRA:-}
