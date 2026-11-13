#!/usr/bin/env bash
set -euo pipefail
cd /home/lenovo_jason/ponderlm2-looped-exp
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4
for pair in '03 t2 T2 D1' '04 t3 T3 D2'; do
    read -r exp name label mask <<< "$pair"
    dir="runs/EXP-$exp-$label"
    mkdir -p "$dir"
    timeout 3650 .venv/bin/python -m src.train --config "configs/exp${exp}_${name}.json" > "$dir/train.log" 2>&1
    .venv/bin/python -m src.evaluate --checkpoint "$dir/checkpoints/step-600" --mode ponder --mask "$mask" > "$dir/evaluate.log" 2>&1
    .venv/bin/python -m src.evaluate --checkpoint "$dir/checkpoints/step-600" --mode ponder --mask D0 > "$dir/evaluate-D0.log" 2>&1
done
