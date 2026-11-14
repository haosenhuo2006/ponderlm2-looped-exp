"""Check T7 configuration identity without training dependencies."""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    baseline = json.loads((ROOT / "configs/exp01_t1.json").read_text())
    config = json.loads((ROOT / "configs/exp13_t7.json").read_text())
    expected = dict(baseline, steps=600, run_dir="runs/EXP-13-T7", split_optimizer_updates=True)
    assert config == expected, "T7 differs beyond steps/run_dir/split_optimizer_updates"
    pipeline = ROOT / "scripts/run_exp13.sh"
    assert pipeline.is_file() and os.access(pipeline, os.X_OK), "pipeline missing or not executable"
    print("EXP-13: T7 config matches T1 except allowed fields; pipeline executable")


if __name__ == "__main__":
    main()
