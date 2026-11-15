"""Validate EXP-18 artifacts and paired three-seed comparisons."""
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


def main():
    results = {"T8": {}, "T0": {}}
    for model, seeds in (("T8", (42, 43, 44)), ("T0", (43, 44))):
        for seed in seeds:
            directory = ROOT / f"runs/EXP-18-{model}-1200-s{seed}"
            e = evaluation(directory / "evaluation.json", "ponder" if model == "T8" else "vanilla")
            record = dict(seed=seed, nll=e["nll"], ppl=e["ppl"],
                          **validate(directory, seed, 1200, model == "T8"))
            if model == "T8":
                middle = evaluation(directory / "evaluation-step600.json", "ponder")
                record["step-600"] = {k: middle[k] for k in ("nll", "ppl")}
            results[model][seed] = record
    directory = ROOT / "runs/EXP-12-T0"
    e = evaluation(directory / "evaluation.json", "vanilla")
    rows = [json.loads(line) for line in (directory / "train.jsonl").read_text().splitlines()]
    results["T0"][42] = dict(seed=42, nll=e["nll"], ppl=e["ppl"],
                             wall_s=read(directory / "complete.json")["wall_s"],
                             peak_memory_bytes=max(r["peak_memory_bytes"] for r in rows),
                             reused_from="EXP-12-T0 (previously validated)")
    gap = paired(results["T8"], results["T0"])
    if gap["mean"] > .05:
        decision = "thought loses; close thought line"
    elif gap["mean"] <= 0:
        decision = "thought ties or wins; investigate mechanism"
    else:
        decision = "within 0.05 but mean direction still worse; inconclusive"
    write("exp18", dict(results=results,
        aggregates={m: aggregate(r) for m, r in results.items()}, paired_t8_minus_t0=gap,
        decision=decision, caveats=[
            "T8 clips separately for each microbatch group.",
            "Cosine LR is stretched over 1200 steps; step-600 is not directly comparable to EXP-17's 600-step endpoint.",
            "WikiText-2 only; 70M tiny-scale; three seeds.",
            "Training FLOPs are the primary budget; T8 costs approximately 8x vanilla per step; inference approximately 1.47x vanilla.",
        ]))


if __name__ == "__main__":
    main()
