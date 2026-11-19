import json
import os
import subprocess
from pathlib import Path

ROOT = Path("/home/lenovo_jason/ponderlm2-looped-exp")
PY = str(ROOT / ".venv/bin/python")
LOG = ROOT / "runs/reproduction/run_l2_resume.log"
RUNS = ["l2-vanilla-s44", "l2-looped-s44"]

def log(msg):
    with LOG.open("a") as f:
        f.write(msg + "\n")

def run_env(name):
    env = dict(os.environ)
    env["HF_HUB_OFFLINE"] = "1"
    env["OMP_NUM_THREADS"] = "4"
    env["PYTHIA_MODEL"] = "EleutherAI/pythia-160M"
    env["PYTHIA_REVISION"] = "50f5173d932e8e61f858120bcb800b97af589f46"
    env["EXP08_RUN_DIR"] = str(ROOT / "runs/reproduction" / name)
    return env

def run_to_file(args, env, outfile):
    with open(outfile, "w") as f:
        r = subprocess.run(args, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
    return r.returncode

def tail(path, n=2):
    lines = Path(path).read_text().strip().split("\n")
    return "\n".join(lines[-n:])

def eval_lm(name):
    cfg = ROOT / "runs/reproduction" / (name + ".json")
    rundir = ROOT / "runs/reproduction" / name
    eval_out = ROOT / "runs/reproduction" / (name + "-eval.out")
    log("=== eval " + name + " ===")
    rc = run_to_file([PY, "-m", "src.evaluate_looped2", "lm",
                      "--checkpoint", str(rundir / "checkpoints" / "step-1200"),
                      "--eval-r", "4", "--s0-seed", "0",
                      "--output", str(rundir / "evaluation.json")], run_env(name), eval_out)
    log(tail(eval_out, 6))
    if rc != 0:
        log("FAIL eval " + name + " rc=" + str(rc))
        return 1
    res = json.loads((rundir / "evaluation.json").read_text())
    row = res["per_r"][0]
    log(name + " NLL=" + str(row.get("nll", row)))
    subprocess.run(["git", "add", str(rundir), str(cfg), str(ROOT / "runs/reproduction" / (name + "-train.out")), str(eval_out)], cwd=ROOT, env=run_env(name))
    subprocess.run(["git", "commit", "-m", "L2 " + name + ": train+eval done"], cwd=ROOT, env=run_env(name))
    log("=== " + name + " DONE ===")
    return 0

def main():
    LOG.write_text("")
    for name in RUNS:
        cfg = ROOT / "runs/reproduction" / (name + ".json")
        rundir = ROOT / "runs/reproduction" / name
        train_out = ROOT / "runs/reproduction" / (name + "-train.out")
        log("=== train " + name + " ===")
        rc = run_to_file([PY, "-m", "src.train", "--config", str(cfg)], run_env(name), train_out)
        log(tail(train_out, 2))
        if rc != 0:
            log("FAIL train " + name + " rc=" + str(rc))
            return 1
        if not (rundir / "complete.json").exists():
            log("FAIL " + name + " no complete.json")
            return 1
        if eval_lm(name) != 0:
            return 1
    rc = subprocess.run(["git", "push"], cwd=ROOT, env=run_env("l2-vanilla-s44")).returncode
    log("git push rc=" + str(rc))
    log("L2 ALL DONE")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
