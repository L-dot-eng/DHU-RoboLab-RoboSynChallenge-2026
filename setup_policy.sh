#!/usr/bin/env bash
# Run in an existing official EmbodiChain 0.2.4 simulator Python 3.11 environment.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON_BIN:-python}"
"$PY" -c 'import sys; assert sys.version_info[:2] == (3,11), "Use Python 3.11"; import embodichain, dexsim'
"$PY" -m pip install -r "$ROOT/requirements-policy.txt"
"$PY" -m pip install --no-deps -e "$ROOT/vendor/lerobot"
PYTHONPATH="$ROOT:$ROOT/vendor/lerobot/src:${PYTHONPATH:-}" "$PY" -c 'from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy; from lerobot.policies.factory import make_pre_post_processors; from lerobot.processor import PolicyProcessorPipeline; import policy.robosyn_team; print("policy imports OK")'
