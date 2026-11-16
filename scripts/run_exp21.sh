#!/usr/bin/env bash
set -uo pipefail
export DOWNSTREAM_RUN=runs/EXP-21
export DOWNSTREAM_CONFIG=configs/exp21_downstream.json
cd "$(dirname "$0")/.."
mkdir -p runs/EXP-21/logs
exec 9>runs/exp21.lock
flock -n 9 || { echo "EXP-21 already running"; exit 1; }
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONUNBUFFERED=1
echo "EXP-21 pipeline start $(date -Is) pid=$$"
mkdir -p runs/EXP-21/data
for task in arc lambada; do
  if [ ! -f "runs/EXP-21/data/$task.parquet" ]; then
    cp "runs/EXP-09/data/$task.parquet" "runs/EXP-21/data/$task.parquet" || exit 1
  fi
done
.venv/bin/python -m src.downstream --prepare > runs/EXP-21/logs/prepare.log 2>&1
 .venv/bin/python - <<'META'
import hashlib,json,os,subprocess,torch
from datetime import datetime,timezone
from pathlib import Path
from src.downstream import RUN,save,protocol_hash
files=["src/downstream.py","scripts/summarize_exp21.py","scripts/run_exp21.sh","configs/exp21_downstream.json","tests/test_downstream.py","tests/test_looped.py"]
save(RUN/"execution.json",dict(started_at=datetime.now(timezone.utc).isoformat(),
    hardware=torch.cuda.get_device_name(),torch_version=torch.__version__,
    git_hash=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
    git_status=subprocess.check_output(["git","status","--short"],text=True),
    source_sha256={f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in files},
    protocol_hash=protocol_hash(),command="nohup bash scripts/run_exp21.sh > runs/EXP-21/pipeline.log 2>&1 &",
    environment={"CUBLAS_WORKSPACE_CONFIG":os.environ["CUBLAS_WORKSPACE_CONFIG"]},
    tests_log="runs/EXP-21/logs/tests.log"))
META
for checkpoint in LOOPMID600 LOOPMID1200 T0_1200 T8_1200; do
  for task in arc lambada; do
    echo "START $checkpoint $task $(date -Is)"
    if .venv/bin/python -m src.downstream --checkpoint "$checkpoint" --task "$task" > "runs/EXP-21/logs/$checkpoint-$task.log" 2>&1; then
      echo "DONE $checkpoint $task $(date -Is)"
    else
      echo "FAILED $checkpoint $task: see per-item log"
    fi
  done
done
.venv/bin/python scripts/summarize_exp21.py
echo "EXP-21 pipeline end $(date -Is)"
