#!/usr/bin/env bash
set -uo pipefail
export EVALSETS26_RUN=runs/EXP-26
export EVALSETS26_CONFIG=configs/exp26.json
cd "$(dirname "$0")/.." || exit 1
mkdir -p runs/EXP-26/logs
exec 9>runs/exp26.lock
flock -n 9 || { echo "EXP-26 already running"; exit 1; }
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONUNBUFFERED=1
echo "EXP-26 pipeline start $(date -Is) pid=$$"
tasks=("$@")
if [ "${#tasks[@]}" -eq 0 ]; then
  tasks=(hellaswag piqa sciq pile)
fi
for task in "${tasks[@]}"; do
  case "$task" in
    hellaswag|piqa|sciq|pile) ;;
    *) echo "Invalid task: $task" >&2; exit 1 ;;
  esac
done
if ! .venv/bin/python -m src.evalsets26 --prepare > runs/EXP-26/logs/prepare.log 2>&1; then
  echo "EXP-26 prepare failed: see runs/EXP-26/logs/prepare.log" >&2
  exit 1
fi
.venv/bin/python - "$0" "$@" <<'META'
import hashlib
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
import torch
from src.evalsets26 import RUN, save, protocol_hash
path = RUN / 'execution.json'
now = datetime.now(timezone.utc).isoformat()
if path.exists():
    metadata = json.loads(path.read_text())
    metadata.setdefault('resumed_at', []).append(now)
else:
    files = ['src/evalsets26.py', 'scripts/summarize_exp26.py', 'scripts/run_exp26.sh',
             'configs/exp26.json', 'tests/test_evalsets26.py']
    metadata = dict(started_at=now, hardware=torch.cuda.get_device_name(), torch_version=torch.__version__,
                    git_hash=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                    git_status=subprocess.check_output(['git', 'status', '--short'], text=True),
                    source_sha256={f: hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in files},
                    protocol_hash=protocol_hash(), command=shlex.join(['bash'] + sys.argv[1:]),
                    environment={'CUBLAS_WORKSPACE_CONFIG': os.environ['CUBLAS_WORKSPACE_CONFIG']},
                    tests_log='runs/EXP-26/logs/tests.log')
save(path, metadata)
META
if [ "$?" -ne 0 ]; then exit 1; fi
# 全量 7 checkpoint × 4 任务，单卡串行；不要并行跑两个占 GPU 的任务。
checkpoint_specs=$(.venv/bin/python - <<'SPECS'
from src.evalsets26 import config
for name, spec in config()['checkpoints'].items():
    print(name, spec['model'], spec['revision'], sep='\t')
SPECS
) || exit 1
while IFS=$'\t' read -r checkpoint model revision; do
  for task in "${tasks[@]}"; do
    export PYTHIA_MODEL="$model"
    export PYTHIA_REVISION="$revision"
    echo "START $checkpoint $task $(date -Is)"
    if .venv/bin/python -m src.evalsets26 --checkpoint "$checkpoint" --task "$task" > "runs/EXP-26/logs/$task-$checkpoint.log" 2>&1; then
      echo "DONE $checkpoint $task $(date -Is)"
    else
      echo "FAILED $checkpoint $task $(date -Is): see per-item log"
    fi
  done
done <<< "$checkpoint_specs"
.venv/bin/python scripts/summarize_exp26.py || exit 1
echo "EXP-26 pipeline end $(date -Is)"
