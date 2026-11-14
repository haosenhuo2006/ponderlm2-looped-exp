"""Validate and aggregate EXP-15 seed-42 training and paired evaluations."""
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
    shared_order = None
    for model in ("T0", "T1"):
        directory = ROOT / f"runs/EXP-15-{model}"
        evaluations = {
            "step-600": evaluation(directory / "evaluation-step600.json", model),
            "step-1200": evaluation(directory / "evaluation.json", model),
        }
        training = [json.loads(line) for line in (directory / "train.jsonl").read_text().splitlines()]
        assert [row["step"] for row in training] == list(range(1, 1201)), model
        assert all(math.isfinite(row["loss"]) and math.isfinite(row["grad_norm"]) for row in training), model
        rng = random.Random(42)
        expected = [rng.choice([2, 3]) if model != "T0" else 0 for _ in range(1200)]
        assert [row["K"] for row in training] == expected, f"{model}: K sequence"
        order = read(directory / "data-order.json")
        assert order["seed"] == 42 and len(order["indices"]) == 19200, model
        assert order["chunks"] == baseline["chunks"], f"{model}: chunk count"
        assert order["indices"][:9600] == baseline["indices"], f"{model}: EXP-01 order prefix differs"
        if shared_order is not None:
            assert order["indices"] == shared_order, f"{model}: model data orders differ"
        shared_order = order["indices"]
        metadata = read(directory / "metadata.json")
        assert metadata["source_sha256"] == source_hashes, f"{model}: source hashes differ"
        assert abs(metadata["parameters"] - 162_000_000) <= 1_000_000, f"{model}: parameter count"
        complete = read(directory / "complete.json")
        assert complete["steps"] == 1200, model
        wall_s = complete["wall_s"]
        peak_mib = max(row["peak_memory_bytes"] for row in training) / 2**20
        assert math.isfinite(wall_s) and wall_s >= 0, model
        assert math.isfinite(peak_mib) and peak_mib >= 0, model
        results[model] = dict(
            seed=42, **evaluations, wall_s=wall_s, peak_memory_mib=peak_mib,
            K_counts={str(k): expected.count(k) for k in (0, 2, 3)},
        )
    gaps = {
        "160M-1200": {
            "T1-T0": results["T1"]["step-1200"]["nll"] - results["T0"]["step-1200"]["nll"],
        },
    }
    for label, experiment in (("160M-600", "EXP-14"),
                              ("70M-1200", "EXP-12"),
                              ("70M-600", "EXP-01")):
        metrics = {
            model: evaluation(ROOT / f"runs/{experiment}-{model}/evaluation.json", model)
            for model in ("T0", "T1")
        }
        gaps[label] = {"T1-T0": metrics["T1"]["nll"] - metrics["T0"]["nll"]}
    output = ROOT / "runs/exp15-summary.json"
    output.write_text(json.dumps(dict(results=results, gaps=gaps), indent=2, allow_nan=False) + "\n")
    print(f"EXP-15 validated: {output}")


if __name__ == "__main__":
    main()
