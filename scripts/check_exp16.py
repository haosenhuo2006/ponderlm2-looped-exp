"""Check T6 configuration identity without training dependencies."""
import json
import os
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    baseline = json.loads((ROOT / "configs/exp07_t6.json").read_text())
    for name, steps, run_dir in [
        ("exp16_t6.json", 1200, "runs/EXP-16-T6"),
        ("exp16_t6_pilot.json", 100, "runs/EXP-16-T6-pilot"),
    ]:
        config = json.loads((ROOT / "configs" / name).read_text())
        assert config == dict(baseline, steps=steps, run_dir=run_dir), name + " differs beyond steps/run_dir"
    rng = random.Random(42)
    expected = [rng.choice([2, 3]) for _ in range(1200)]
    assert expected.count(2) == 587 and expected.count(3) == 613
    assert sum(expected) == 3013
    pipeline = ROOT / "scripts/run_exp16.sh"
    assert pipeline.is_file() and os.access(pipeline, os.X_OK), "pipeline missing or not executable"
    print("EXP-16: configs match exp07_t6 except steps/run_dir; K sequence 587x2/613x3; pipeline executable")


if __name__ == "__main__":
    main()
