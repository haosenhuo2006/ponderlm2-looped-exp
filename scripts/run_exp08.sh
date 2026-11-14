#!/usr/bin/env bash
set -euo pipefail
cd /home/lenovo_jason/ponderlm2-looped-exp
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4
exec 9>runs/exp08.lock
flock -n 9 || exit 1
.venv/bin/python -m scripts.check_exp08
for seed in 43 44; do
  for model in T0 T1 T2 T4; do
    dir="runs/EXP-08-seed$seed-$model"
    export EXP08_RUN_DIR="$dir"
    mkdir -p "$dir"
    if [[ -f "$dir/pipeline-complete.json" ]]; then continue; fi
    success=0
    for attempt in 1 2; do
      if [[ -e "$dir/train.jsonl" ]]; then
        archive="$dir/failed-attempt-$((attempt-1))-$(date +%s)"
        mkdir -p "$archive"
        find "$dir" -maxdepth 1 -type f -exec mv -t "$archive" {} +
        if [[ -d "$dir/checkpoints" ]]; then mv "$dir/checkpoints" "$archive/"; fi
      fi
      logfile="runs/exp08-$seed-$model-pipeline.log"
      echo "START seed=$seed model=$model attempt=$attempt $(date -Is)" >> "$logfile"
      set +e
      nohup timeout 3650 .venv/bin/python -m src.train --config "configs/exp08_seed${seed}_${model,,}.json" >> "$logfile" 2>&1 < /dev/null &
      child=$!
      wait "$child"
      rc=$?
      set -e
      echo "{\"attempt\":$attempt,\"train_exit_code\":$rc}" >> "$dir/attempts.jsonl"
      if [[ $rc -eq 0 ]]; then success=1; break; fi
    done
    if [[ $success -eq 0 ]]; then echo "FAILED $seed $model"; continue; fi
    if [[ "$model" == T4 ]]; then
      .venv/bin/python -m src.evaluate_looped --checkpoint "$dir/checkpoints/step-600" > "$dir/evaluate.log" 2>&1
    else
      mode=ponder
      args=()
      if [[ "$model" == T0 ]]; then mode=vanilla; fi
      if [[ "$model" == T2 ]]; then args=(--mask D1); fi
      .venv/bin/python -m src.evaluate --checkpoint "$dir/checkpoints/step-600" --mode "$mode" --split validation "${args[@]}" > "$dir/evaluate.log" 2>&1
    fi
    echo '{"complete":true}' > "$dir/pipeline-complete.json"
  done
done
unset EXP08_RUN_DIR
.venv/bin/python scripts/summarize_exp08.py
