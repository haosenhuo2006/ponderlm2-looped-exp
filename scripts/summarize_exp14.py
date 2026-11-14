"""Validate and aggregate EXP-14 seed-42 training and paired evaluations."""
import hashlib
import json
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text())


def evaluation(path, model):
    result = read(path)
    assert math.isfinite(result["nll"]), f"{path}: nonfinite NLL"
    assert math.isfinite(result["ppl"]), f"{path}: nonfinite PPL"
    assert abs(math.exp(result["nll"]) - result["ppl"]) < 1e-8, path
    assert result["predicted_tokens"] == 249135 and result["seq"] == 256, path
    assert result["mode"] == ("vanilla" if model == "T0" else "ponder"), path
    assert result["split"] == "validation", path
    return {key: result[key] for key in ("nll", "ppl")}


def main():
    baseline = read(ROOT / "runs/exp01-order.json")
    assert baseline["seed"] == 42 and len(baseline["indices"]) == 9600
    source_hashes = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((ROOT / "src").glob("*.py"))
    }
    results = {}
    for model in ("T0", "T1"):
        directory = ROOT / f"runs/EXP-14-{model}"
        metrics = evaluation(directory / "evaluation.json", model)
        training = [json.loads(line) for line in (directory / "train.jsonl").read_text().splitlines()]
        assert [row["step"] for row in training] == list(range(1, 601)), model
        assert all(math.isfinite(row["loss"]) and math.isfinite(row["grad_norm"]) for row in training), model
        rng = random.Random(42)
        expected = [rng.choice([2, 3]) if model != "T0" else 0 for _ in range(600)]
        assert [row["K"] for row in training] == expected, f"{model}: K sequence"
        order = read(directory / "data-order.json")
        assert order == baseline, f"{model}: EXP-01 data order differs"
        metadata = read(directory / "metadata.json")
        assert metadata["source_sha256"] == source_hashes, f"{model}: source hashes differ"
        assert abs(metadata["parameters"] - 162_000_000) <= 1_000_000, f"{model}: parameter count"
        complete = read(directory / "complete.json")
        assert complete["steps"] == 600, model
        wall_s = complete["wall_s"]
        peak_mib = max(row["peak_memory_bytes"] for row in training) / 2**20
        assert math.isfinite(wall_s) and wall_s >= 0, model
        assert math.isfinite(peak_mib) and peak_mib >= 0, model
        results[model] = dict(
            seed=42, **metrics, wall_s=wall_s, peak_memory_mib=peak_mib,
            K_counts={str(k): expected.count(k) for k in (0, 2, 3)},
        )
    baseline_metrics = {
        model: evaluation(ROOT / f"runs/EXP-01-{model}/evaluation.json", model)
        for model in ("T0", "T1")
    }
    gaps = {
        "160M": {"T1-T0": results["T1"]["nll"] - results["T0"]["nll"]},
        "70M": {"T1-T0": baseline_metrics["T1"]["nll"] - baseline_metrics["T0"]["nll"]},
    }
    output = ROOT / "runs/exp14-summary.json"
    output.write_text(json.dumps(dict(results=results, gaps=gaps), indent=2, allow_nan=False) + "\n")
    print(f"EXP-14 validated: {output}")


if __name__ == "__main__":
    main()
