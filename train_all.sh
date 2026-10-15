#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"
RUN_ROOT="${RUN_ROOT:-runs/three_policies_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$RUN_ROOT"
RUN_ROOT="$(realpath "$RUN_ROOT")"
printf '%s\n' "$RUN_ROOT" > runs/latest_three_policies.txt
current_policy=setup
trap 'code=$?; if (( code != 0 )); then printf "%s FAILED %s exit=%s\n" "$(date -Is)" "$current_policy" "$code" >> "$RUN_ROOT/status.log"; fi' EXIT

for current_policy in act diffusion smolvla; do
  printf '%s START %s\n' "$(date -Is)" "$current_policy" >> "$RUN_ROOT/status.log"
  policy_args=()
  if [[ "$current_policy" == smolvla ]]; then
    # The full policy checkpoint already contains the VLM weights.
    policy_args=(--policy.path=lerobot/smolvla_base --policy.input_features='{}'
      --policy.load_vlm_weights=false)
  fi
  POLICY_TYPE="$current_policy" OUTPUT_DIR="$RUN_ROOT/$current_policy" \
    ./train.sh "${policy_args[@]}" > "$RUN_ROOT/$current_policy.log" 2>&1
  test -s "$RUN_ROOT/$current_policy/checkpoints/last/pretrained_model/model.safetensors"
  printf '%s DONE %s\n' "$(date -Is)" "$current_policy" >> "$RUN_ROOT/status.log"
done
printf '%s ALL_DONE\n' "$(date -Is)" >> "$RUN_ROOT/status.log"
