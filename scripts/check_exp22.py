"""Verify middle-block configurations and compute budget without training."""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import looped


def main():
    for suffix, seed, steps in (("s42", 42, 600), ("s43", 43, 600),
                                ("s44", 44, 600), ("pilot", 42, 100)):
        template = "pilot" if suffix == "pilot" else "s42"
        baseline = json.loads((ROOT / f"configs/exp19_loopmid_{template}.json").read_text())
        c = json.loads((ROOT / f"configs/exp22_loopmid160_{suffix}.json").read_text())
        expected = dict(baseline, model="EleutherAI/pythia-160M",
                        revision="50f5173d932e8e61f858120bcb800b97af589f46",
                        seed=seed, run_dir=f"runs/EXP-22-loopmid160-{suffix}",
                        loop=dict(block_size=6, repetitions=2, start=3), run_budget_s=5400)
        assert c == expected, suffix
        assert c["steps"] == steps
        assert c["loop"] == {"block_size": 6, "repetitions": 2, "start": 3}
    baseline = json.loads((ROOT / "configs/exp14_t0.json").read_text())
    for seed in (43, 44):
        c = json.loads((ROOT / f"configs/exp22_t0_s{seed}.json").read_text())
        expected = dict(baseline, seed=seed, run_dir=f"runs/EXP-22-T0-s{seed}")
        assert c == expected, seed
    dimensions = dict(d=768, ff=3072, vocab=50304, layers=12, context=128)
    assert looped.flops(6, 2, **dimensions) == 339148800
    assert looped.flops(1, 1, **dimensions) == 251854848
    assert 12 + 6*(2-1) == 18
    # Registered R=2 versus R=3 comparison against the 1.43 target.
    ratio_r2 = looped.flops(6, 2, **dimensions) / looped.flops(1, 1, **dimensions)
    ratio_r3 = looped.flops(6, 3, **dimensions) / looped.flops(1, 1, **dimensions)
    assert 12 + 6*(3-1) == 24
    assert round(ratio_r2, 4) == 1.3466 and round(ratio_r3, 4) == 1.6932
    assert round(100*abs(ratio_r2/1.43-1), 1) == 5.8
    assert round(100*abs(ratio_r3/1.43-1), 1) == 18.4
    assert abs(ratio_r2/1.43-1) < abs(ratio_r3/1.43-1)
    pipeline = ROOT / "scripts/run_exp22.sh"
    assert pipeline.is_file() and os.access(pipeline, os.X_OK)
    print("EXP-22: full/pilot/T0 configurations passed; 18 calls, 339148800 FLOPs/token; pipeline executable")


if __name__ == "__main__":
    main()
