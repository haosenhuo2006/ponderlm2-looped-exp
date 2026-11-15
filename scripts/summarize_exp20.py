"""Validate middle-block loop x 1200 steps and apply the pre-registered decision rules.

Decision rules (fixed before running; see ~/workspace/phase5-design.md and the
phase-6 task brief -- do not change):
  gap = mean(loopmid1200 NLL - T0x1200 NLL) over seeds 42/43/44
  gap < -0.05 and all three seeds better  -> looped_wins_1200 (main question YES on Looped side)
  |gap| <= 0.05                          -> tie_persists
  gap > +0.05                            -> looped_loses_1200
Controls are read from the already-validated phase-5 summaries (exp18, exp19);
their recomputed means are asserted against the reported values.
"""
import argparse
import hashlib
import json
import math
import random
from pathlib import Path
from statistics import mean, stdev, median

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text())


def evaluation(path, mode):
    e = read(path)
    assert math.isfinite(e["nll"]) and math.isfinite(e["ppl"]), path
    assert abs(math.exp(e["nll"]) - e["ppl"]) < 1e-8, path
    assert e["predicted_tokens"] == 249135 and e["seq"] == 256, path
    assert e["mode"] == mode and e["split"] == "validation", path
    return e


def validate(directory, seed, steps, thought=False):
    rows = [json.loads(line) for line in (directory / "train.jsonl").read_text().splitlines()]
    assert [r["step"] for r in rows] == list(range(1, steps+1)), directory
    assert all(math.isfinite(r[k]) for r in rows for k in ("loss", "grad_norm")), directory
    rng = random.Random(seed)
    assert [r["K"] for r in rows] == [rng.choice([2, 3]) if thought else 0 for _ in rows], directory
    data = read(directory / "data-order.json")
    assert data["seed"] == seed and data["chunks"] == 9411 and len(data["indices"]) == steps*16, directory
    import numpy as np
    rng_order = np.random.default_rng(seed)
    expected = []
    while len(expected) < steps*16:
        expected.extend(rng_order.permutation(9411).tolist())
    assert data["indices"] == expected[:steps*16], directory
    if seed == 42:
        baseline = read(ROOT / "runs/exp01-order.json")
        assert data["indices"][:9600] == baseline["indices"], directory
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted((ROOT / "src").glob("*.py"))}
    assert read(directory / "metadata.json")["source_sha256"] == hashes, directory
    complete = read(directory / "complete.json")
    assert complete["steps"] == steps, directory
    wall = complete["wall_s"]
    peak = max(r["peak_memory_bytes"] for r in rows)
    assert math.isfinite(wall) and wall >= 0 and math.isfinite(peak) and peak >= 0
    return dict(wall_s=wall, peak_memory_bytes=peak)


def aggregate(results):
    assert len(results) == 3
    return {key: dict(mean=mean(r[key] for r in results.values()),
                      sd=stdev(r[key] for r in results.values()), n=3)
            for key in ("nll", "ppl")}


def paired(left, right):
    values = {str(s): left[s]["nll"] - right[s]["nll"] for s in (42, 43, 44)}
    signs = [1 if v > 0 else -1 if v < 0 else 0 for v in values.values()]
    return dict(by_seed=values, mean=mean(values.values()),
                direction_consistent=len(set(signs)) == 1,
                all_better=all(v < 0 for v in values.values()))


def write(name, result):
    path = ROOT / f"runs/{name}-summary.json"
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"Validated: {path}")


def load_control(summary_path, *keys):
    node = read(ROOT / summary_path)
    for key in keys:
        node = node[key]
    out = {}
    for seed in (42, 43, 44):
        entry = node[str(seed)]
        out[seed] = dict(seed=seed, nll=entry["nll"], ppl=entry["ppl"])
    return out


def main():
    results = {}
    for seed in (42, 43, 44):
        directory = ROOT / f"runs/EXP-20-loopmid-s{seed}"
        e = evaluation(directory / "evaluation.json", "looped")
        for key, value in dict(block_size=2, repetitions=4, loop_block_start=2,
                               layers=[2, 3], inference_flops_per_token_estimate=130154496).items():
            assert e[key] == value, (directory, key)
        results[seed] = dict(seed=seed, nll=e["nll"], ppl=e["ppl"],
                             **validate(directory, seed, 1200))
    controls = {
        "T0x1200": load_control("runs/exp18-summary.json", "results", "T0"),
        "T8x1200": load_control("runs/exp18-summary.json", "results", "T8"),
        "loopmid600": load_control("runs/exp19-summary.json", "results"),
    }
    # Sanity: recomputed control means must match the reported phase-5 values.
    assert abs(mean(r["nll"] for r in controls["T0x1200"].values()) - 4.7147) < 1e-4
    assert abs(mean(r["nll"] for r in controls["T8x1200"].values()) - 4.8319) < 1e-4
    assert abs(mean(r["nll"] for r in controls["loopmid600"].values()) - 4.8710) < 1e-4
    gaps = {m: paired(results, r) for m, r in controls.items()}
    gap_t0 = gaps["T0x1200"]["mean"]
    if gap_t0 < -0.05 and gaps["T0x1200"]["all_better"]:
        decision = "looped_wins_1200"
    elif abs(gap_t0) <= 0.05:
        decision = "tie_persists"
    else:
        decision = "looped_loses_1200"
    write("exp20", dict(results=results, aggregate=aggregate(results), controls=controls,
        control_aggregates={m: aggregate(r) for m, r in controls.items()},
        paired_differences=gaps, decision=decision, caveats=[
            "WikiText-2 only; 70M tiny-scale; three seeds.",
            "Training FLOPs approximately 2x vanilla; 12 layer calls; inference estimate 130154496 FLOPs/token.",
            "Same analytical inference cost as T4 and EXP-19; prefix evaluation recomputes context without KV cache.",
            "Decision rules fixed before running: win needs mean gap < -0.05 with all three seeds better; |gap| <= 0.05 is a tie.",
        ]))


def pilot():
    directory = ROOT / "runs/EXP-20-loopmid-pilot"
    result = dict(passed=False, decision="stop")
    try:
        rows = [json.loads(line) for line in (directory / "train.jsonl").read_text().splitlines()]
        base = [json.loads(line) for line in (ROOT / "runs/EXP-05-T4/train.jsonl").read_text().splitlines()][:100]
        assert len(base) == 100
        finite = all(math.isfinite(r[k]) for r in rows for k in ("loss", "grad_norm"))
        baseline = median(r["grad_norm"] for r in base)
        measured = median(r["grad_norm"] for r in rows)
        ratio = measured / baseline
        first, last = mean(r["loss"] for r in rows[:20]), mean(r["loss"] for r in rows[-20:])
        data = read(directory / "data-order.json")
        reference = read(ROOT / "runs/exp01-order.json")
        same_order = len(data["indices"]) >= 1600 and data["indices"][:1600] == reference["indices"][:1600]
        ordered = [r["step"] for r in rows] == list(range(1, 101))
        complete = (directory / "complete.json").exists()
        passed = len(rows) == 100 and ordered and complete and finite and last <= 1.5*first and .1 <= ratio <= 10 and same_order
        result.update(passed=passed, decision="continue" if passed else "stop",
            rows=len(rows), ordered_steps=ordered, complete=complete, finite=finite,
            no_divergence=last <= 1.5*first, data_order_prefix_matches=same_order,
            gradient_median=measured, baseline_gradient_median=baseline,
            gradient_ratio=ratio, first20_mean_ce=first, last20_mean_ce=last,
            max_gradient=max(r["grad_norm"] for r in rows))
    except (OSError, ValueError, KeyError, TypeError, AssertionError, ZeroDivisionError) as error:
        result["error"] = str(error)
    result = {k: v if not isinstance(v, float) or math.isfinite(v) else None for k, v in result.items()}
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "decision.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, allow_nan=False))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    pilot() if args.pilot else main()
