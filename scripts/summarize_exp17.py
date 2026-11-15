"""Validate T8 artifacts and compare final validation NLL with T1/T5/T6/T7."""
import argparse
from statistics import mean, median
import hashlib
import json
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text())


def main():
    directory = ROOT / "runs/EXP-17-T8"
    training = [json.loads(line) for line in (directory / "train.jsonl").read_text().splitlines()]
    assert [row["step"] for row in training] == list(range(1, 601)), "T8: expected 600 ordered steps"
    assert all(len(row["round_ce"]) == row["K"] for row in training)
    assert all(math.isfinite(v) for row in training for v in
               [row["loss"], row["grad_norm"], row["final_prediction_ce"], *row["round_ce"]])
    rng = random.Random(42)
    expected = [rng.choice([2, 3]) for _ in range(600)]
    assert [row["K"] for row in training] == expected, "T8: K sequence"
    assert [expected.count(2), expected.count(3)] == [306, 294]
    cumulative = 0
    for row, k in zip(training, expected):
        cumulative += k
        assert row["optimizer_updates"] == k and row["cumulative_optimizer_updates"] == cumulative
    assert cumulative == 1494
    baseline = read(ROOT / "runs/exp01-order.json")
    assert baseline["seed"] == 42 and len(baseline["indices"]) == 9600
    assert read(directory / "data-order.json") == baseline, "T8: data order differs from T1"
    source_hashes = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((ROOT / "src").glob("*.py"))
    }
    assert read(directory / "metadata.json")["source_sha256"] == source_hashes, "T8: source hashes differ"
    evaluation = read(directory / "evaluation.json")
    assert math.isfinite(evaluation["nll"]) and math.isfinite(evaluation["ppl"])
    assert abs(math.exp(evaluation["nll"]) - evaluation["ppl"]) < 1e-8
    assert evaluation["predicted_tokens"] == 249135 and evaluation["seq"] == 256
    assert evaluation["mode"] == "ponder" and evaluation["split"] == "validation"
    assert not evaluation.get("mask"), "T8: unexpected mask"
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
    for model, run in (("T1", "EXP-01-T1"), ("T5", "EXP-06-T5"), ("T6", "EXP-07-T6"), ("T7", "EXP-13-T7")):
        nll = read(ROOT / f"runs/{run}/evaluation.json")["nll"]
        assert math.isfinite(nll), f"{model}: nonfinite NLL"
        comparison[model] = dict(nll=nll)
    comparison["T8"] = dict(nll=result["nll"])
    output = ROOT / "runs/exp17-summary.json"
    output.write_text(json.dumps(dict(
        results={"T8": result}, comparison=comparison,
        caveats=[
            "T8 clips once per microbatch group; T1 clips once per data step.",
            "Small-batch gradient noise is part of the more-frequent-update mechanism.",
            "T6 backpropagates each round CE separately; T8 backpropagates their mean once per microbatch. This supervision difference is part of the per-round-update mechanism definition.",
        ],
    ), indent=2, allow_nan=False) + "\n")
    print(f"EXP-17 validated: {output}")


def pilot():
    directory = ROOT / "runs/EXP-17-T8-pilot"
    result = dict(passed=False, decision="stop")
    try:
        training = [json.loads(line) for line in (directory / "train.jsonl").read_text().splitlines()]
        bases = [[json.loads(line) for line in (ROOT / f"runs/{run}/train.jsonl").read_text().splitlines()][:100]
                 for run in ("EXP-01-T1", "EXP-06-T5")]
        assert all(len(b) == 100 for b in bases), "baseline requires 100 steps"
        complete = (directory / "complete.json").exists()
        finite = all(math.isfinite(v) for row in training for v in
                     [row["loss"], row["grad_norm"], row["final_prediction_ce"], *row["round_ce"]])
        baseline = median(row["grad_norm"] for b in bases for row in b)
        measured = median(row["grad_norm"] for row in training)
        ratio = measured / baseline
        first = mean(row["loss"] for row in training[:20])
        last = mean(row["loss"] for row in training[-20:])
        stable = last <= 1.5 * first
        order = read(directory / "data-order.json")
        reference = read(ROOT / "runs/exp01-order.json")
        same_order = (len(order["indices"]) >= 1600 and
                      order["indices"][:1600] == reference["indices"][:1600])
        ordered = [row["step"] for row in training] == list(range(1, 101))
        round_shape = all(len(row["round_ce"]) == row["K"] for row in training)
        passed = (len(training) == 100 and ordered and complete and finite and
                  stable and .1 <= ratio <= 10 and same_order and round_shape)
        result.update(passed=passed, decision="continue" if passed else "stop",
                      rows=len(training), ordered_steps=ordered, complete=complete,
                      finite=finite, no_divergence=stable, data_order_prefix_matches=same_order,
                      round_shape=round_shape, gradient_median=measured,
                      baseline_gradient_median=baseline, gradient_ratio=ratio,
                      max_gradient=max(row["grad_norm"] for row in training),
                      first20_mean_ce=first, last20_mean_ce=last)
    except (OSError, ValueError, KeyError, TypeError, AssertionError, ZeroDivisionError) as error:
        result["error"] = str(error)
    # Keep failure evidence valid JSON even when the observed metrics are nonfinite.
    result = {key: (value if not isinstance(value, float) or math.isfinite(value) else None)
              for key, value in result.items()}
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "decision.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, allow_nan=False))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    if args.pilot:
        pilot()
    else:
        main()
