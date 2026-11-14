#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p runs/EXP-09/logs
exec 9>runs/EXP-09/pipeline.lock
flock -n 9 || { echo "EXP-09 already running"; exit 1; }
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONUNBUFFERED=1
echo "EXP-09 pipeline start $(date -Is) pid=$$"
.venv/bin/python -m src.downstream --prepare > runs/EXP-09/logs/prepare.log 2>&1
 .venv/bin/python - <<'META'
import hashlib,json,os,subprocess,torch
from datetime import datetime,timezone
from pathlib import Path
from src.downstream import RUN,save,protocol_hash
files=["src/downstream.py","scripts/summarize_exp09.py","scripts/run_exp09.sh","configs/exp09_downstream.json","tests/test_downstream.py"]
save(RUN/"execution.json",dict(started_at=datetime.now(timezone.utc).isoformat(),
    hardware=torch.cuda.get_device_name(),torch_version=torch.__version__,
    git_hash=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
    git_status=subprocess.check_output(["git","status","--short"],text=True),
    source_sha256={f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in files},
    protocol_hash=protocol_hash(),command="nohup bash scripts/run_exp09.sh > runs/EXP-09/pipeline.log 2>&1 &",
    environment={"CUBLAS_WORKSPACE_CONFIG":os.environ["CUBLAS_WORKSPACE_CONFIG"]},
    tests_log="runs/EXP-09/logs/tests.log"))
META
for checkpoint in T0 T1 T2 T4 T5; do
  for task in arc lambada; do
    echo "START $checkpoint $task $(date -Is)"
    if .venv/bin/python -m src.downstream --checkpoint "$checkpoint" --task "$task" > "runs/EXP-09/logs/$checkpoint-$task.log" 2>&1; then
      echo "DONE $checkpoint $task $(date -Is)"
    else
      echo "FAILED $checkpoint $task: see per-item log"
    fi
  done
done
.venv/bin/python scripts/summarize_exp09.py
echo "EXP-09 pipeline end $(date -Is)"
