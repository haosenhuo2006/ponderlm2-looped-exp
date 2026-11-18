"""Validate Step 1 evidence and judge frozen branches; never execute follow-ups."""
import hashlib
import json
import math
import random
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (42, 43, 44)


def read(path):
    return json.loads(path.read_text())


def validate_prefix(data, seed, expected):
    reference = ROOT / f"runs/EXP-25-T0-s{seed}/data-order.json"
    if reference.is_file():
        old = read(reference)
        assert old["seed"] == seed and old["chunks"] == 9411, reference
        assert len(old["indices"]) == 19200, reference
        assert data["indices"][:19200] == old["indices"], reference
        return str(reference.relative_to(ROOT))
    if seed != 42:
        raise FileNotFoundError(f"Required EXP-25 prefix evidence missing: {reference}")
    for name in ("EXP-15-T0-s42/data-order.json", "EXP-15-T0/data-order.json"):
        alternate = ROOT / "runs" / name
        if alternate.is_file():
            old = read(alternate)
            assert old["seed"] == 42 and old["chunks"] == 9411, alternate
            assert len(old["indices"]) == 19200, alternate
            assert data["indices"][:19200] == old["indices"], alternate
            return f"s42 fallback: {alternate.relative_to(ROOT)}"
    alternate = ROOT / "runs/exp01-order.json"
    if not alternate.is_file():
        raise FileNotFoundError(f"Missing {reference}; s42 fallback also missing: {alternate}")
    old = read(alternate)
    assert len(old["indices"]) == 9600, alternate
    assert data["indices"][:9600] == old["indices"] == expected[:9600], alternate
    # order() is a step-count-independent seeded permutation stream.
    assert data["indices"][:19200] == expected[:19200]
    return "s42 fallback: exp01-order 9600 indices + seeded order() extension to 19200"


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
    prefix_evidence = validate_prefix(data, seed, expected)
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted((ROOT / "src").glob("*.py"))}
    assert read(directory / "metadata.json")["source_sha256"] == hashes, directory
    complete = read(directory / "complete.json")
    assert complete["steps"] == steps, directory
    wall = complete["wall_s"]
    peak = max(r["peak_memory_bytes"] for r in rows)
    assert math.isfinite(wall) and wall >= 0 and math.isfinite(peak) and peak >= 0
    return dict(wall_s=wall, peak_memory_bytes=peak, prefix_evidence=prefix_evidence)


def aggregate(results):
    assert len(results) == 3
    return {key: dict(mean=mean(r[key] for r in results.values()),
                      sd=stdev(r[key] for r in results.values()), n=3)
            for key in ("nll", "ppl")}



def paired(left, right):
    values = {str(s): left[s]["nll"] - right[s]["nll"] for s in SEEDS}
    average, sd = mean(values.values()), stdev(values.values())
    se = sd / math.sqrt(3)
    signs = [1 if v > 0 else -1 if v < 0 else 0 for v in values.values()]
    return dict(by_seed=values, mean=average, sd=sd, se=se,
                ci95=[average - 4.303 * se, average + 4.303 * se],
                t_critical=4.303, n=3,
                direction_consistent=len(set(signs)) == 1,
                all_better=all(v < 0 for v in values.values()))


def judge(gap, rollback):
    passed = gap["mean"] < -0.05 and gap["all_better"] and gap["direction_consistent"]
    # Rollback is the specific exception to branch (a). No numeric definition
    # of "very wide" is frozen; a crossing interval does not veto section 3.
    if rollback > 0.05 and passed:
        branch = "c"
    elif gap["mean"] >= -0.05:
        branch = "b"
    elif passed:
        branch = "a"
    else:
        branch = "d"
    decision = {
        "a": "revival holds at equal FLOPs (within this recipe)",
        "b": "revival is a compute artifact",
        "c": "undecided — direction A contaminated by T0 rollback; Step 1B required",
        "d": "undecided",
    }[branch]
    annotation = None
    if gap["direction_consistent"] and gap["ci95"][0] < 0 < gap["ci95"][1]:
        annotation = "operational threshold passed, statistically uncertain" if passed else "statistically uncertain"
    return dict(branch=branch, decision=decision, operational_threshold_passed=passed,
                branch_status="undecided" if branch == "d" else branch,
                annotation=annotation)


def main():
    previous = read(ROOT / "runs/exp25-summary.json")
    loop_results = {s: previous["results"][str(s)] for s in SEEDS}
    controls = {}
    for seed in SEEDS:
        reused = loop_results[seed]
        assert math.isfinite(reused["nll"])
        path = ROOT / f"runs/EXP-25-loopmid160-s{seed}/evaluation.json"
        if path.is_file():
            nll = read(path)["nll"]
            assert math.isfinite(nll) and abs(nll - reused["nll"]) <= 1e-9, path
        directory = ROOT / f"runs/STEP1-T0-s{seed}"
        e = evaluation(directory / "evaluation.json", "vanilla")
        controls[seed] = dict(seed=seed, nll=e["nll"], ppl=e["ppl"],
                              **validate(directory, seed, 1616))
    old = {s: previous["controls"]["T0"][str(s)] for s in SEEDS}
    gap = paired(loop_results, controls)
    rollback = mean(r["nll"] for r in controls.values()) - mean(r["nll"] for r in old.values())
    verdict = judge(gap, rollback)
    result = dict(results=loop_results, controls={"T0_1616": controls, "T0_1200": old},
                  aggregate=previous["aggregate"],
                  control_aggregates={"T0_1616": aggregate(controls), "T0_1200": aggregate(old)},
                  paired_differences=gap, t0_1616_minus_1200=rollback,
                  **verdict, caveats=[
                      "Three pairs; WikiText-2; within this recipe only; no follow-up authorized.",
                      "Cosine schedule horizon rescheduled to 1616 steps; approximately 2.75 epochs.",
                      "Section 3 strict >0.05 takes precedence over section 4(a) >=0.05.",
                      "Branch c takes precedence over a; very wide interval has no frozen numeric cutoff.",
                      "Profiler is a separate post-hoc annotation; analytical budget ratio 1.3466."])
    path = ROOT / "runs/step1-summary.json"
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print("STEP1 VERDICT")
    print(f"branch ({verdict['branch']}): {verdict['decision']}")
    print(f"paired differences: {gap['by_seed']}")
    print(f"mean={gap['mean']:.6f}, SD={gap['sd']:.6f}, SE={gap['se']:.6f}, 95% t CI={gap['ci95']}")
    print(f"direction consistent={gap['direction_consistent']}; T0 rollback={rollback:.6f}")
    if verdict["annotation"]:
        print(verdict["annotation"])
    print(f"Validated: {path}")


if __name__ == "__main__":
    main()
