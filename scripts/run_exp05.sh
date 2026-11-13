#!/usr/bin/env bash
set -euo pipefail
cd /home/lenovo_jason/ponderlm2-looped-exp
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4
mkdir -p runs/EXP-05-T4
timeout 3650 .venv/bin/python -m src.train --config configs/exp05_t4.json > runs/EXP-05-T4/train.log 2>&1
.venv/bin/python -m src.evaluate_looped --checkpoint runs/EXP-05-T4/checkpoints/step-600 > runs/EXP-05-T4/evaluate.log 2>&1
