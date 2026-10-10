#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TASK="${1:?Usage: bash eval.sh TASK [random|clear] [GPU] [extra official arguments]}"
SETTING="${2:-random}"
GPU="${3:-0}"
if (( $# >= 3 )); then shift 3; else shift "$#"; fi
export TMPDIR="$ROOT/.runtime/tmp"
export HF_HOME="$ROOT/.runtime/huggingface"
export XDG_CACHE_HOME="$ROOT/.runtime/cache"
export MPLCONFIGDIR="$ROOT/.runtime/matplotlib"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export EMBODICHAIN_SIM_EXIT_PROCESS=0 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$ROOT:$ROOT/vendor/lerobot/src:${EMBODICHAIN_ROOT:-}${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$TMPDIR" "$HF_HOME" "$XDG_CACHE_HOME" "$MPLCONFIGDIR"
unset CUDA_VISIBLE_DEVICES
cd "$ROOT"
exec "${PYTHON_BIN:-python}" run_eval.py \
  --config policy/robosyn_team/deploy_policy.yml --overrides \
  --task_name "$TASK" --setting "$SETTING" --gpu_id "$GPU" "$@"
