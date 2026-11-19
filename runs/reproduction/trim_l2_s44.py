import os
import shutil
import time
from pathlib import Path

ROOT = Path("/home/lenovo_jason/ponderlm2-looped-exp")
LOG = ROOT / "runs/reproduction/trim_l2_s44.log"
DRIVER_LOG = ROOT / "runs/reproduction/run_l2_resume.log"
GROUPS = ["l2-vanilla-s44", "l2-looped-s44"]
INTERMEDIATE_STEPS = [200, 400, 600, 800, 1000]
POLL_S = 30
DEADLINE_S = 3 * 3600

def log(msg):
    with LOG.open("a") as f:
        f.write(msg + "\n")

def dir_size(path):
    total = 0
    for dp, dn, fn in os.walk(path):
        for f in fn:
            fp = os.path.join(dp, f)
            try:
                total += os.path.getsize(fp)
            except OSError:
                pass
    return total

def trim_group(name):
    rundir = ROOT / "runs/reproduction" / name
    ckpt_dir = rundir / "checkpoints"
    final = ckpt_dir / "step-1200"
    if not final.exists():
        log("SKIP " + name + ": step-1200 missing, not trimming")
        return False
    freed = 0
    deleted = 0
    for step in INTERMEDIATE_STEPS:
        d = ckpt_dir / ("step-" + str(step))
        if d.exists():
            freed += dir_size(d)
            shutil.rmtree(d)
            deleted += 1
    log(name + ": deleted " + str(deleted) + " intermediate checkpoints, freed " + str(round(freed / 1e9, 2)) + " GB, kept step-1200")
    return True

def main():
    LOG.write_text("")
    done = set()
    waited = 0
    while len(done) < len(GROUPS) and waited < DEADLINE_S:
        try:
            text = DRIVER_LOG.read_text()
        except OSError:
            text = ""
        if "FAIL" in text:
            log("driver reported FAIL, watcher exiting without trimming")
            return 1
        for name in GROUPS:
            if name not in done and ("=== " + name + " DONE ===") in text:
                if trim_group(name):
                    done.add(name)
        if len(done) < len(GROUPS):
            time.sleep(POLL_S)
            waited += POLL_S
    if len(done) == len(GROUPS):
        log("TRIM ALL DONE")
        return 0
    log("watcher deadline reached, trimmed: " + str(sorted(done)))
    return 1

if __name__ == "__main__":
    raise SystemExit(main())
