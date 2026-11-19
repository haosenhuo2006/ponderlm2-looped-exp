#!/bin/bash
# Finish Yang linear looped (seed 42) after the driver was killed per user instruction.
# Waits for the orphaned looped training (pid 8882) to finish 50k steps,
# then runs eval-only on model_50000.pt and saves eval.json.
# Runs on WSL; standard model is never started.
REPO=/home/lenovo_jason/ponderlm2-looped-exp
CKPT=$REPO/runs/reproduction/yang-linear-looped-s42/model_50000.pt
EVAL_OUT=$REPO/runs/reproduction/yang-linear-looped-s42-eval.out
EVAL_JSON=$REPO/runs/reproduction/yang-linear-looped-s42/eval.json
cd $REPO
for i in $(seq 1 45); do
  if [ -f "$CKPT" ] && ! ps -p 8882 >/dev/null 2>&1; then
    echo "TRAIN_DONE iter=$i"
    break
  fi
  if ! ps -p 8882 >/dev/null 2>&1 && [ ! -f "$CKPT" ]; then
    echo "TRAIN_DIED_NO_CKPT"
    exit 1
  fi
  sleep 20
done
if [ ! -f "$CKPT" ]; then
  echo "TIMEOUT_NO_CKPT"
  exit 2
fi
ls -la "$CKPT"
tail -2 $REPO/runs/reproduction/yang-linear-looped-s42/train_log.jsonl
echo "RUN_EVAL"
.venv/bin/python -m src.train_yang --task linear_regression --model looped --seed 42 --eval-only --checkpoint "$CKPT" > "$EVAL_OUT" 2>&1
echo "eval rc=$?"
cat "$EVAL_OUT"
.venv/bin/python - "$EVAL_OUT" "$EVAL_JSON" <<'PYEOF'
import json, sys
out_path, json_path = sys.argv[1], sys.argv[2]
metrics = None
for line in open(out_path):
    line = line.strip()
    if line.startswith("{"):
        try:
            metrics = json.loads(line)
        except ValueError:
            pass
if metrics is None:
    print("NO_JSON_FOUND")
    sys.exit(1)
open(json_path, "w").write(json.dumps(metrics, indent=2) + "\n")
print("EVAL_JSON_SAVED", metrics)
PYEOF
exit 0
