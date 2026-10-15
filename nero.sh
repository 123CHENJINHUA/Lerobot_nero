#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export HF_HOME="$PROJECT_ROOT/.cache/huggingface"
export HF_LEROBOT_HOME="$PROJECT_ROOT/.cache/lerobot"
export XDG_CACHE_HOME="$PROJECT_ROOT/.cache/xdg"
export TORCH_HOME="$PROJECT_ROOT/.cache/torch"
export WANDB_DIR="$PROJECT_ROOT/runs"
export PYTHONDONTWRITEBYTECODE=1
cd "$PROJECT_ROOT"
if [[ -z "${NERO_PYTHON:-}" ]] && ! command -v python >/dev/null 2>&1; then
  echo "找不到 python；请先 conda activate lerobot，或设置 NERO_PYTHON。" >&2
  exit 127
fi
exec "${NERO_PYTHON:-python}" "$PROJECT_ROOT/nero.py" "$@"
