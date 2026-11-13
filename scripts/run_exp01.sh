#!/usr/bin/env bash
set -euo pipefail
cd /home/lenovo_jason/ponderlm2-looped-exp
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=4
for pair in 't0 vanilla T0' 't1 ponder T1'; do
    read -r name mode label <<< "$pair"
    dir="runs/EXP-01-$label"
    mkdir -p "$dir"
    timeout 3650 .venv/bin/python -m src.train --config "configs/exp01_$name.json" > "$dir/train.log" 2>&1
    .venv/bin/python -m src.evaluate --checkpoint "$dir/checkpoints/step-600" --mode "$mode" --split validation > "$dir/evaluate.log" 2>&1
done
.venv/bin/python -m src.jacobi_check --checkpoint runs/EXP-01-T1/checkpoints/step-600 > runs/EXP-01-T1/jacobi.log 2>&1
