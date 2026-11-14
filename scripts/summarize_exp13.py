"""Validate T7 artifacts and compare final validation NLL with T1/T5/T6."""
import hashlib
import json
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text())


def main():
    directory = ROOT / "runs/EXP-13-T7"
    training = [json.loads(line) for line in (directory / "train.jsonl").read_text().splitlines()]
    assert [row["step"] for row in training] == list(range(1, 601)), "T7: expected 600 ordered steps"
    assert all(math.isfinite(row["loss"]) and math.isfinite(row["grad_norm"]) for row in training)
    rng = random.Random(42)
    expected = [rng.choice([2, 3]) for _ in range(600)]
    assert [row["K"] for row in training] == expected, "T7: K sequence"
    assert [expected.count(2), expected.count(3)] == [306, 294]
    cumulative = 0
    for row, k in zip(training, expected):
        cumulative += k
        assert row["optimizer_updates"] == k and row["cumulative_optimizer_updates"] == cumulative
    assert cumulative == 1494
    baseline = read(ROOT / "runs/exp01-order.json")
    assert baseline["seed"] == 42 and len(baseline["indices"]) == 9600
    assert read(directory / "data-order.json") == baseline, "T7: data order differs from T1"
    source_hashes = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((ROOT / "src").glob("*.py"))
    }
    assert read(directory / "metadata.json")["source_sha256"] == source_hashes, "T7: source hashes differ"
    evaluation = read(directory / "evaluation.json")
    assert math.isfinite(evaluation["nll"]) and math.isfinite(evaluation["ppl"])
    assert abs(math.exp(evaluation["nll"]) - evaluation["ppl"]) < 1e-8
    assert evaluation["predicted_tokens"] == 249135 and evaluation["seq"] == 256
    assert evaluation["mode"] == "ponder" and evaluation["split"] == "validation"
    assert not evaluation.get("mask"), "T7: unexpected mask"
    complete = read(directory / "complete.json")
    assert complete["steps"] == 600
    wall_s = complete["wall_s"]
    peak_mib = max(row["peak_memory_bytes"] for row in training) / 2**20
    assert math.isfinite(wall_s) and wall_s >= 0
    assert math.isfinite(peak_mib) and peak_mib >= 0
    result = dict(nll=evaluation["nll"], ppl=evaluation["ppl"], wall_s=wall_s,
                  peak_memory_mib=peak_mib, seed=42, optimizer_updates=cumulative,
                  K_counts={str(k): expected.count(k) for k in (2, 3)})
    comparison = {}
    for model, run in (("T1", "EXP-01-T1"), ("T5", "EXP-06-T5"), ("T6", "EXP-07-T6")):
        nll = read(ROOT / f"runs/{run}/evaluation.json")["nll"]
        assert math.isfinite(nll), f"{model}: nonfinite NLL"
        comparison[model] = dict(nll=nll)
    comparison["T7"] = dict(nll=result["nll"])
    output = ROOT / "runs/exp13-summary.json"
    output.write_text(json.dumps(dict(
        results={"T7": result}, comparison=comparison,
        caveats=["T7 clips once per microbatch group; T1 clips once per data step. This residual difference is not controlled."],
    ), indent=2, allow_nan=False) + "\n")
    print(f"EXP-13 validated: {output}")


if __name__ == "__main__":
    main()
