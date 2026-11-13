#!/usr/bin/env bash
set -euo pipefail
cd /home/lenovo_jason/ponderlm2-looped-exp
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=4
dir=runs/EXP-06-T5
mkdir -p "$dir"
timeout 3650 .venv/bin/python -m src.train --config configs/exp06_t5.json > "$dir/train.log" 2>&1
.venv/bin/python -m src.evaluate --checkpoint "$dir/checkpoints/step-600" --mode ponder > "$dir/evaluate.log" 2>&1
.venv/bin/python -m src.evaluate_rounds --checkpoint "$dir/checkpoints/step-600" --output "$dir/rounds-T5.json" > "$dir/rounds-T5.log" 2>&1
.venv/bin/python -m src.evaluate_rounds --checkpoint runs/EXP-01-T1/checkpoints/step-600 --output "$dir/rounds-T1.json" > "$dir/rounds-T1.log" 2>&1
