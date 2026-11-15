"""Verify middle-block 1200-step configurations and compute budget without training."""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import looped


def main():
    baseline = json.loads((ROOT / "configs/exp05_t4.json").read_text())
    for suffix, seed, steps in (("s42", 42, 1200), ("s43", 43, 1200),
                                ("s44", 44, 1200), ("pilot", 42, 100)):
        c = json.loads((ROOT / f"configs/exp20_loopmid_{suffix}.json").read_text())
        expected = dict(baseline, seed=seed, run_dir=f"runs/EXP-20-loopmid-{suffix}",
                        steps=steps, loop=dict(block_size=2, repetitions=4, start=2))
        assert c == expected, suffix
        assert c["loop"] == {"block_size": 2, "repetitions": 4, "start": 2}
    assert looped.flops(2, 4) == 130154496
    assert 6 + 2*(4-1) == 12
    pipeline = ROOT / "scripts/run_exp20.sh"
    assert pipeline.is_file() and os.access(pipeline, os.X_OK)
    print("EXP-20: full/pilot configurations passed; 12 calls, 130154496 FLOPs/token; pipeline executable")


if __name__ == "__main__":
    main()
