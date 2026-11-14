"""Check EXP-12 configuration identity without importing training dependencies."""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = {"T0": "exp01_t0", "T1": "exp01_t1", "T2": "exp03_t2"}


def main():
    for model, base in BASE.items():
        original = json.loads((ROOT / f"configs/{base}.json").read_text())
        config = json.loads((ROOT / f"configs/exp12_{model.lower()}.json").read_text())
        expected = dict(original, steps=1200, run_dir=f"runs/EXP-12-{model}")
        assert config == expected, f"{model}: configuration differs beyond steps/run_dir"
    pipeline = ROOT / "scripts/run_exp12.sh"
    assert pipeline.is_file() and os.access(pipeline, os.X_OK), "pipeline missing or not executable"
    print("EXP-12: all three configs match baselines except steps/run_dir; pipeline executable")


if __name__ == "__main__":
    main()
