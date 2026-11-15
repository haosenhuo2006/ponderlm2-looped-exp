"""Configuration identity and deterministic 1200-step sampling; no training."""
import hashlib
import json
import os
import random
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import order


def ks(seed, count):
    rng = random.Random(seed)
    return [rng.choice([2, 3]) for _ in range(count)]


def main():
    records = {}
    previous = os.environ.get("EXP08_RUN_DIR")
    try:
        for model, seeds, base in (("t8", (42, 43, 44), "exp17_t8"),
                                   ("t0", (43, 44), "exp12_t0")):
            baseline = json.loads((ROOT / f"configs/{base}.json").read_text())
            for seed in seeds:
                c = json.loads((ROOT / f"configs/exp18_{model}_1200_s{seed}.json").read_text())
                expected = dict(baseline, run_dir=f"runs/EXP-18-{model.upper()}-1200-s{seed}", seed=seed)
                if model == "t8":
                    expected["steps"] = 1200
                assert c == expected, (model, seed)
                sequence = ks(seed, 1200) if model == "t8" else [0] * 1200
                assert sequence == (ks(seed, 1200) if model == "t8" else [0] * 1200)
                if model == "t8" and seed == 42:
                    assert sequence[:600] == ks(42, 600)
                with tempfile.TemporaryDirectory() as directory:
                    os.environ["EXP08_RUN_DIR"] = directory
                    first = order(9411, 19200, seed)
                    Path(directory, "data-order.json").unlink()
                    assert first == order(9411, 19200, seed)
                records[f"{model}-s{seed}"] = dict(seed=seed, steps=1200,
                    K_counts={str(k): sequence.count(k) for k in sorted(set(sequence))},
                    K_sha256=hashlib.sha256(bytes(sequence)).hexdigest(),
                    order_count=len(first),
                    order_sha256=hashlib.sha256(json.dumps(first).encode()).hexdigest(),
                    exp17_K_prefix_matches=(sequence[:600] == ks(42, 600)) if model == "t8" and seed == 42 else None)
    finally:
        if previous is None:
            os.environ.pop("EXP08_RUN_DIR", None)
        else:
            os.environ["EXP08_RUN_DIR"] = previous
    pipeline = ROOT / "scripts/run_exp18.sh"
    assert pipeline.is_file() and os.access(pipeline, os.X_OK)
    (ROOT / "reports/exp18-check.json").write_text(json.dumps(records, indent=2) + "\n")
    print("EXP-18: all five configurations and deterministic K/data sequences passed")


if __name__ == "__main__":
    main()
