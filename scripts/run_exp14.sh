#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4
mkdir -p runs
exec 9>runs/exp14.lock
flock -n 9 || exit 1
.venv/bin/python scripts/check_exp14.py
trap 'unset EXP08_RUN_DIR PYTHIA_MODEL PYTHIA_REVISION' EXIT

archive_attempt() {
  local archive
  archive="$dir/failed-attempt-$1-$(date +%s%N)"
  mkdir -p "$archive"
  find "$dir" -maxdepth 1 -type f -exec mv -t "$archive" {} +
  if [[ -d "$dir/checkpoints" ]]; then mv "$dir/checkpoints" "$archive/"; fi
}

run_attempt() {
  timeout 5500 .venv/bin/python -m src.train --config "configs/exp14_${model,,}.json" >> "$logfile" 2>&1 || return $?
  .venv/bin/python -m src.evaluate --checkpoint "$dir/checkpoints/step-600" --mode "$mode" --split validation >> "$logfile" 2>&1 || return $?
}

for model in T0 T1; do
  dir="runs/EXP-14-$model"
  export EXP08_RUN_DIR="$dir"
  mkdir -p "$dir"
  logfile="runs/exp14-$model-pipeline.log"
  PYTHIA_MODEL=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["model"])' "configs/exp14_${model,,}.json")
  PYTHIA_REVISION=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["revision"])' "configs/exp14_${model,,}.json")
  export PYTHIA_MODEL PYTHIA_REVISION
  .venv/bin/python - <<'PY_PREFLIGHT' >> "$logfile" 2>&1
import json
from pathlib import Path
from unittest.mock import patch
import torch
from transformers import AutoTokenizer, PreTrainedModel
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
revision_70m = "a39f36b100fe8a5377810d56c3f4789b9c53ac42"
text = "The quick brown fox jumps over the lazy dog. Language models predict the next token.\n"
small = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m", revision=revision_70m)
large = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
parity = small(text, add_special_tokens=False)["input_ids"] == large(text, add_special_tokens=False)["input_ids"]
Path("runs/exp14-tokenizer-parity.json").write_text(json.dumps(dict(revision_70m=revision_70m, revision_160m=REVISION, parity=parity, text=text), indent=2) + "\n")
assert parity, "70M/160M tokenizer IDs differ"
PY_PREFLIGHT
  if [[ -f "$dir/pipeline-complete.json" ]]; then continue; fi
  mode=ponder
  if [[ "$model" == T0 ]]; then mode=vanilla; fi
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
.venv/bin/python scripts/summarize_exp14.py
unset EXP08_RUN_DIR PYTHIA_MODEL PYTHIA_REVISION
