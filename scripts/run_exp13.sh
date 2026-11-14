#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4
mkdir -p runs
exec 9>runs/exp13.lock
flock -n 9 || exit 1
.venv/bin/python scripts/check_exp13.py
trap 'unset EXP08_RUN_DIR' EXIT

archive_attempt() {
  local archive
  archive="$dir/failed-attempt-$1-$(date +%s%N)"
  mkdir -p "$archive"
  find "$dir" -maxdepth 1 -type f -exec mv -t "$archive" {} +
  if [[ -d "$dir/checkpoints" ]]; then mv "$dir/checkpoints" "$archive/"; fi
}

run_attempt() {
  timeout 3650 .venv/bin/python -m src.train --config "configs/exp13_${model,,}.json" >> "$logfile" 2>&1 || return $?
  .venv/bin/python -m src.evaluate --checkpoint "$dir/checkpoints/step-600" --mode ponder --split validation >> "$logfile" 2>&1 || return $?
}

for model in T7; do
  dir="runs/EXP-13-$model"
  export EXP08_RUN_DIR="$dir"
  mkdir -p "$dir"
  if [[ -f "$dir/pipeline-complete.json" ]]; then continue; fi
  logfile="runs/exp13-$model-pipeline.log"
  if [[ -n "$(find "$dir" -maxdepth 1 -type f -print -quit)" || -d "$dir/checkpoints" ]]; then
    archive_attempt previous
  fi
  success=0
  for attempt in 1 2; do
    echo "START model=$model attempt=$attempt $(date -Is)" >> "$logfile"
    if run_attempt; then
      echo "{\"attempt\":$attempt,\"exit_code\":0}" >> "$dir/attempts.jsonl"
      echo '{"complete":true}' > "$dir/pipeline-complete.json"
      success=1
      break
    else
      rc=$?
      echo "{\"attempt\":$attempt,\"exit_code\":$rc}" >> "$dir/attempts.jsonl"
      echo "FAILED model=$model attempt=$attempt exit_code=$rc $(date -Is)" >> "$logfile"
      archive_attempt "$attempt"
    fi
  done
  if [[ $success -eq 0 ]]; then
    echo "FAILED $model after two attempts" >&2
    exit 1
  fi
done
.venv/bin/python scripts/summarize_exp13.py
unset EXP08_RUN_DIR
