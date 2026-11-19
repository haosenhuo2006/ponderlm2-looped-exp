"""Yang linear reproduction driver (seed 42): looped then standard, back-to-back.

Frozen spec (prereg v1.0, Yang Q4):
  task linear_regression, d=20, k=40, b(n_loops)=20, T(n_loop_window)=15,
  50k steps, batch 64, Adam lr 1e-4, no wd, no clip, from scratch.
  Models: looped 1-layer GPT (D=256, 8 heads) vs standard 12-layer GPT (D=256, 8 heads).
  Eval: 1280 prompts, eval_seed 1234, final-round prediction, normalized MSE
  (denominator 20) + 90% bootstrap CI (1000 trials).

Launched detached (setsid); writes only to log files; never prints to stdout.
Idempotent: skips a model whose final checkpoint + eval.json already exist;
if only the final checkpoint exists, runs eval only (no retraining).
train_yang.py has NO resume: an interrupted model must restart from step 1.
"""

import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path("/home/lenovo_jason/ponderlm2-looped-exp")
PY = REPO / ".venv" / "bin" / "python"
BASE = REPO / "runs" / "reproduction"
DRIVER_LOG = BASE / "yang-linear-s42-driver.log"
SUMMARY = BASE / "yang-linear-s42.json"

TASK = "linear_regression"
SEED = 42
STEPS = 50000
MODELS = ["looped", "standard"]


def log(msg):
    line = time.strftime("%Y-%m-%d %H:%M:%S") + " " + msg + "\n"
    with DRIVER_LOG.open("a", encoding="utf-8") as f:
        f.write(line)


def run_dir(model):
    return BASE / ("yang-linear-" + model + "-s" + str(SEED))


def final_ckpt(model):
    return run_dir(model) / "model_50000.pt"


def eval_json_path(model):
    return run_dir(model) / "eval.json"


def train(model):
    out_dir = run_dir(model)
    cmd = [str(PY), "-m", "src.train_yang", "--task", TASK, "--model", model,
           "--seed", str(SEED), "--out-dir", str(out_dir)]
    log("train start model=" + model + " cmd=" + " ".join(cmd))
    with (BASE / ("yang-linear-" + model + "-s" + str(SEED) + "-train.out")).open(
            "w", encoding="utf-8") as lf:
        r = subprocess.run(cmd, cwd=REPO, stdout=lf, stderr=subprocess.STDOUT)
    log("train exit rc=" + str(r.returncode) + " model=" + model)
    return r.returncode


def evaluate(model):
    ckpt = final_ckpt(model)
    cmd = [str(PY), "-m", "src.train_yang", "--task", TASK, "--model", model,
           "--seed", str(SEED), "--eval-only", "--checkpoint", str(ckpt)]
    log("eval start model=" + model + " cmd=" + " ".join(cmd))
    with (BASE / ("yang-linear-" + model + "-s" + str(SEED) + "-eval.out")).open(
            "w", encoding="utf-8") as lf:
        r = subprocess.run(cmd, cwd=REPO, stdout=lf, stderr=subprocess.STDOUT)
    log("eval exit rc=" + str(r.returncode) + " model=" + model)
    if r.returncode != 0:
        return None
    metrics = None
    text = (BASE / ("yang-linear-" + model + "-s" + str(SEED) + "-eval.out")).read_text(
        encoding="utf-8")
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                metrics = json.loads(line)
            except ValueError:
                pass
    if metrics is None:
        return None
    eval_json_path(model).write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return metrics


def trim(model):
    d = run_dir(model)
    removed = []
    for p in sorted(d.glob("model_*.pt")):
        if p.name != "model_50000.pt":
            p.unlink()
            removed.append(p.name)
    log("trim model=" + model + " removed=" + str(removed))


def main():
    BASE.mkdir(parents=True, exist_ok=True)
    log("driver start: yang linear s42, looped then standard, 50k steps each")
    results = {}
    for model in MODELS:
        if final_ckpt(model).exists() and eval_json_path(model).exists():
            log("skip model=" + model + ": final checkpoint + eval.json already present")
            results[model] = json.loads(eval_json_path(model).read_text(encoding="utf-8"))
            continue
        if final_ckpt(model).exists():
            log("retrain skipped model=" + model + ": final checkpoint exists, eval only")
        else:
            rc = train(model)
            if rc != 0 or not final_ckpt(model).exists():
                log("ABORT model=" + model + ": training failed rc=" + str(rc)
                    + "; driver stops here")
                return 1
        metrics = evaluate(model)
        if metrics is None:
            log("WARN model=" + model + ": eval failed; keeping checkpoints, continuing")
            results[model] = None
            continue
        log("eval done model=" + model + " normalized_mse=" + str(metrics["normalized_mse"]))
        results[model] = metrics
        trim(model)
    looped = results.get("looped")
    standard = results.get("standard")
    diff = None
    ratio = None
    if looped and standard:
        diff = looped["normalized_mse"] - standard["normalized_mse"]
        ratio = looped["normalized_mse"] / standard["normalized_mse"]
    summary = {
        "task": TASK,
        "seed": SEED,
        "steps": STEPS,
        "batch_size": 64,
        "lr": 1e-4,
        "d": 20,
        "k": 40,
        "b_n_loops": 20,
        "T_n_loop_window": 15,
        "normalization_denominator": 20,
        "n_eval": 1280,
        "eval_seed": 1234,
        "models": {
            "looped": "1-layer GPT, D=256, 8 heads, 847873 params",
            "standard": "12-layer GPT, D=256, 8 heads, 9535233 params",
        },
        "looped": looped,
        "standard": standard,
        "diff_looped_minus_standard": diff,
        "ratio_looped_over_standard": ratio,
        "note": "Single seed (42). Prereg Yang verdict needs 2 seeds for direction "
                "consistency; this run alone cannot apply the full verdict rule.",
    }
    SUMMARY.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    log("summary written to " + str(SUMMARY) + " diff=" + str(diff) + " ratio=" + str(ratio))
    log("driver done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
