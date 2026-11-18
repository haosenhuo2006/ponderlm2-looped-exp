#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4
mkdir -p runs
exec 9>runs/step1.lock
flock -n 9 || exit 1
.venv/bin/python scripts/check_step1.py
trap 'unset EXP08_RUN_DIR PYTHIA_MODEL PYTHIA_REVISION' EXIT

archive_attempt() {
  local archive
  archive="$dir/failed-attempt-$1-$(date +%s%N)"
  mkdir -p "$archive"
  find "$dir" -maxdepth 1 -type f -exec mv -t "$archive" {} +
  if [[ -d "$dir/checkpoints" ]]; then mv "$dir/checkpoints" "$archive/"; fi
}

set_model() {
  PYTHIA_MODEL=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["model"])' "$config")
  PYTHIA_REVISION=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["revision"])' "$config")
  export PYTHIA_MODEL PYTHIA_REVISION
}

preflight() {
  .venv/bin/python - <<'PY_PREFLIGHT'
from unittest.mock import patch
import torch
from transformers import PreTrainedModel
from src.ponder import MODEL, REVISION, load_model

# Exercise load_model's config and official class without reading any weights.
def without_weights(cls, source, *args, config, **kwargs):
    with torch.device("meta"):
        return cls(config)

with patch.object(PreTrainedModel, "from_pretrained", classmethod(without_weights)), patch.object(torch.nn.Module, "cuda", lambda self: self):
    model = load_model("ponder")
parameters = sum(p.numel() for p in model.parameters())
print(f"{MODEL}@{REVISION}: {parameters} parameters", flush=True)
assert 160_000_000 <= parameters <= 165_000_000, parameters
PY_PREFLIGHT
}

run_attempt() {
  timeout 3650 .venv/bin/python -m src.train --config "$config" >> "$logfile" 2>&1 || return $?
  .venv/bin/python -m src.evaluate --checkpoint "$dir/checkpoints/step-1616" --mode vanilla --split validation >> "$logfile" 2>&1 || return $?
}

for seed in 42 43 44; do
    model="T0-s$seed"
    config="configs/step1_t0_s$seed.json"
    dir="runs/STEP1-$model"
    export EXP08_RUN_DIR="$dir"
    mkdir -p "$dir"
    if [[ -f "$dir/pipeline-complete.json" ]]; then continue; fi
    logfile="runs/step1-$model-pipeline.log"
    set_model
    preflight >> "$logfile" 2>&1
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
.venv/bin/python scripts/summarize_step1.py
unset EXP08_RUN_DIR PYTHIA_MODEL PYTHIA_REVISION
