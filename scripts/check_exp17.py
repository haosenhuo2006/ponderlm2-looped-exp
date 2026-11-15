"""Check full and pilot T8 configurations without training dependencies."""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    baseline = json.loads((ROOT / "configs/exp01_t1.json").read_text())
    for suffix, steps, directory in (("", 600, "runs/EXP-17-T8"),
                                     ("_pilot", 100, "runs/EXP-17-T8-pilot")):
        config = json.loads((ROOT / f"configs/exp17_t8{suffix}.json").read_text())
        expected = dict(baseline, steps=steps, run_dir=directory,
                        split_optimizer_updates=True, intermediate_supervision=True)
        assert config == expected, "T8 differs beyond the four allowed fields"
    pipeline = ROOT / "scripts/run_exp17.sh"
    assert pipeline.is_file() and os.access(pipeline, os.X_OK), "pipeline missing or not executable"
    print("EXP-17: full/pilot configs match T1 except allowed fields; pipeline executable")


if __name__ == "__main__":
    main()
