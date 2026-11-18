"""Lightweight Step 1 checks: no training, checkpoints, GPU or network."""
import importlib.util
import json
import math
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"scripts/{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


summary = load("summarize_step1")
profile = load("profile_flops_step1")


class Step1Tests(unittest.TestCase):
    def test_config_parity(self):
        baseline = json.loads((ROOT / "configs/exp15_t0.json").read_text())
        for seed in (42, 43, 44):
            actual = json.loads((ROOT / f"configs/step1_t0_s{seed}.json").read_text())
            self.assertEqual(actual, dict(baseline, steps=1616, seed=seed,
                                          run_dir=f"runs/STEP1-T0-s{seed}"))
        self.assertEqual(1616, round(1200 * 1.3466))

    def test_paired_statistics(self):
        left = {s: {"nll": x} for s, x in zip((42, 43, 44), (1., 2., 3.))}
        right = {s: {"nll": 0.} for s in (42, 43, 44)}
        gap = summary.paired(left, right)
        self.assertEqual(gap["by_seed"], {"42": 1., "43": 2., "44": 3.})
        self.assertEqual(gap["mean"], 2.)
        self.assertEqual(gap["sd"], 1.)
        self.assertAlmostEqual(gap["se"], 1 / math.sqrt(3))
        self.assertEqual(gap["ci95"], [2 - 4.303 / math.sqrt(3), 2 + 4.303 / math.sqrt(3)])

    def test_branches(self):
        right = {s: {"nll": 0.} for s in (42, 43, 44)}
        decisions = {
            "a": "revival holds at equal FLOPs (within this recipe)",
            "b": "revival is a compute artifact",
            "c": "undecided — direction A contaminated by T0 rollback; Step 1B required",
            "d": "undecided",
        }
        for values, rollback, branch in [((-.1, -.2, -.3), 0., "a"),
                                         ((0., .1, .2), 0., "b"),
                                         ((-.1, -.2, -.3), .06, "c"),
                                         ((-.4, -.3, .1), 0., "d"),
                                         ((-.02, -.03, -.01), .06, "b"),
                                         ((-.4, -.3, .1), .06, "d")]:
            left = {s: {"nll": x} for s, x in zip((42, 43, 44), values)}
            verdict = summary.judge(summary.paired(left, right), rollback)
            self.assertEqual(verdict["branch"], branch)
            self.assertEqual(verdict["decision"], decisions[branch])
        gap = dict(mean=-.05, all_better=True, direction_consistent=True, ci95=[-.1, .1])
        self.assertEqual(summary.judge(gap, 0)["branch"], "b")
        gap["mean"] = -.1
        self.assertEqual(summary.judge(gap, 0)["annotation"],
                         "operational threshold passed, statistically uncertain")

    def test_profile_schema(self):
        payload = json.dumps(dict(device="cpu", microbatch=1, seq=256,
            vanilla_flops_per_token=100., looped_flops_per_token=134.66,
            measured_ratio=1.3466, analytical_ratio=1.3466,
            deviation_pct=0., within_5pct=True))
        data = profile.validate_profile(json.loads(payload))
        for key in profile.FIELDS:
            incomplete = dict(data)
            del incomplete[key]
            with self.assertRaises(AssertionError):
                profile.validate_profile(incomplete)


if __name__ == "__main__":
    unittest.main()
