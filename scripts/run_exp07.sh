#!/usr/bin/env bash
set -euo pipefail
cd /home/lenovo_jason/ponderlm2-looped-exp
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4
dir=runs/EXP-07-T6
mkdir -p "$dir" runs/EXP-07-T6-pilot
timeout 3650 .venv/bin/python -m src.train --config configs/exp07_t6_pilot.json > runs/EXP-07-T6-pilot/train.log 2>&1
.venv/bin/python scripts/summarize_exp07.py --pilot
timeout 3650 .venv/bin/python -m src.train --config configs/exp07_t6.json > "$dir/train.log" 2>&1
.venv/bin/python -m src.evaluate --checkpoint "$dir/checkpoints/step-600" --mode ponder > "$dir/evaluate.log" 2>&1
.venv/bin/python -m src.evaluate_rounds --checkpoint "$dir/checkpoints/step-600" --output "$dir/rounds-T6.json" > "$dir/rounds-T6.log" 2>&1
.venv/bin/python -m src.evaluate_rounds --checkpoint runs/EXP-01-T1/checkpoints/step-600 --output "$dir/rounds-T1.json" > "$dir/rounds-T1.log" 2>&1
.venv/bin/python -m src.jacobi_check --checkpoint "$dir/checkpoints/step-600" --output-dir "$dir" > "$dir/jacobi.log" 2>&1
.venv/bin/python scripts/summarize_exp07.py
