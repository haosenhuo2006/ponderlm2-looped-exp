"""Check the frozen Step 1 configurations without training or network access."""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import looped


def main():
    baseline = json.loads((ROOT / "configs/exp15_t0.json").read_text())
    for seed in (42, 43, 44):
        actual = json.loads((ROOT / f"configs/step1_t0_s{seed}.json").read_text())
        assert actual == dict(baseline, steps=1616, seed=seed,
                              run_dir=f"runs/STEP1-T0-s{seed}"), seed
    dimensions = dict(d=768, ff=3072, vocab=50304, layers=12, context=128)
    assert looped.flops(6, 2, **dimensions) == 339148800
    assert looped.flops(1, 1, **dimensions) == 251854848
    assert round(339148800 / 251854848, 4) == 1.3466
    assert round(1200 * 1.3466) == 1616
    pipeline = ROOT / "scripts/run_step1.sh"
    assert pipeline.is_file() and os.access(pipeline, os.X_OK)
    print("STEP1: three T0 configurations passed; FLOPs ratio 1.3466; 1616 steps; pipeline executable")


if __name__ == "__main__":
    main()
