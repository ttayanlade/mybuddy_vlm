#!/usr/bin/env bash
# Pre-download open-source model weights into the project's HF cache (run on the LOGIN
# node: compute nodes may not have internet). ~17 GB for the default VLM + ~1.5 GB for
# Grounding DINO + SAM 2. Add more models as arguments.
#
#   scripts/slurm/download_models.sh
#   scripts/slurm/download_models.sh nvidia/Cosmos-Reason1-7B Qwen/Qwen3-VL-8B-Thinking
set -euo pipefail
cd "$(dirname "$0")/../.."
export HF_HOME=$PWD/.hf_cache
.venv/bin/pip install -q "huggingface_hub[cli]>=0.24"
MODELS=(Qwen/Qwen3-VL-8B-Instruct IDEA-Research/grounding-dino-base facebook/sam2.1-hiera-large "$@")
for m in "${MODELS[@]}"; do
  echo ">> $m"
  .venv/bin/huggingface-cli download "$m" --exclude "*.pth.tar" "original/*" >/dev/null
done
du -sh "$HF_HOME"
