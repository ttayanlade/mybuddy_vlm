#!/usr/bin/env bash
# Pre-download VLM weights into the project's HF cache (LOGIN node: compute nodes may lack
# internet). Default: Qwen/Qwen3-VL-8B-Instruct (~17 GB). Extra model ids as arguments.
set -euo pipefail
cd "$(dirname "$0")/../.."
export HF_HOME=$PWD/.hf_cache
.venv/bin/pip install -q "huggingface_hub[cli]>=0.24" 2>&1 | grep -vi "pip version\|upgrade" || true
for m in Qwen/Qwen3-VL-8B-Instruct "$@"; do
  echo ">> $m"   # python API: the CLI name changed across huggingface_hub versions (huggingface-cli -> hf)
  .venv/bin/python -c "import sys; from huggingface_hub import snapshot_download; print(snapshot_download(sys.argv[1]))" "$m"
done
du -sh "$HF_HOME"
