#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4
mkdir -p runs
exec 9>runs/exp16.lock
flock -n 9 || exit 1
.venv/bin/python scripts/check_exp16.py
trap 'unset EXP08_RUN_DIR' EXIT

dir=runs/EXP-16-T6
logfile=runs/exp16-T6-pipeline.log
mkdir -p "$dir"
if [[ -f "$dir/pipeline-complete.json" ]]; then exit 0; fi

archive_attempt() {
  local archive
  archive="$dir/failed-attempt-$1-$(date +%s%N)"
  mkdir -p "$archive"
  find "$dir" -maxdepth 1 -type f -exec mv -t "$archive" {} +
  if [[ -d "$dir/checkpoints" ]]; then mv "$dir/checkpoints" "$archive/"; fi
}

run_attempt() {
  local stage rc
  export EXP08_RUN_DIR=runs/EXP-16-T6
  timeout 3650 .venv/bin/python -m src.train --config configs/exp16_t6.json >> "$logfile" 2>&1 || return $?
  .venv/bin/python -m src.evaluate --checkpoint "$dir/checkpoints/step-1200" --mode ponder --split validation >> "$logfile" 2>&1 || return $?
  stage=$(mktemp -d "$dir/.evaluation-step600.XXXXXX") || return $?
  if mkdir -p "$stage/checkpoints" &&
     ln -s "$PWD/$dir/checkpoints/step-600" "$stage/checkpoints/step-600" &&
     .venv/bin/python -m src.evaluate --checkpoint "$stage/checkpoints/step-600" --mode ponder --split validation >> "$logfile" 2>&1 &&
     mv "$stage/evaluation.json" "$dir/evaluation-step600.json"; then
    rm -rf -- "$stage" || return $?
  else
    rc=$?
    rm -rf -- "$stage"
    return "$rc"
  fi
  .venv/bin/python -m src.evaluate_rounds --checkpoint "$dir/checkpoints/step-1200" --output "$dir/rounds-T6.json" > "$dir/rounds-T6.log" 2>&1 || return $?
  .venv/bin/python -m src.evaluate_rounds --checkpoint runs/EXP-01-T1/checkpoints/step-600 --output "$dir/rounds-T1.json" > "$dir/rounds-T1.log" 2>&1 || return $?
  .venv/bin/python -m src.jacobi_check --checkpoint "$dir/checkpoints/step-1200" --output-dir "$dir" > "$dir/jacobi.log" 2>&1 || return $?
}

if [[ -n "$(find "$dir" -maxdepth 1 -type f -print -quit)" || -d "$dir/checkpoints" ]]; then
  archive_attempt previous
fi
export EXP08_RUN_DIR=runs/EXP-16-T6-pilot
mkdir -p runs/EXP-16-T6-pilot
timeout 3650 .venv/bin/python -m src.train --config configs/exp16_t6_pilot.json > runs/EXP-16-T6-pilot/train.log 2>&1
.venv/bin/python scripts/summarize_exp16.py --pilot
success=0
for attempt in 1 2; do
  echo "START model=T6 attempt=$attempt $(date -Is)" >> "$logfile"
  if run_attempt; then
    echo "{\"attempt\":$attempt,\"exit_code\":0}" >> "$dir/attempts.jsonl"
    echo '{"complete":true}' > "$dir/pipeline-complete.json"
    success=1
    break
  else
    rc=$?
    echo "{\"attempt\":$attempt,\"exit_code\":$rc}" >> "$dir/attempts.jsonl"
    echo "FAILED model=T6 attempt=$attempt exit_code=$rc $(date -Is)" >> "$logfile"
    archive_attempt "$attempt"
  fi
done
if [[ $success -eq 0 ]]; then
  echo "FAILED T6 after two attempts" >&2
  exit 1
fi
.venv/bin/python scripts/summarize_exp16.py
unset EXP08_RUN_DIR
