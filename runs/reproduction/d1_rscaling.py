"""D1: R-scaling eval of the EXP-25 looped checkpoint (seed 42, step-1200).

Replicates the parallel-NLL computation of src/evaluate_looped.py exactly
(bf16 autocast, FP32 CE, sum over logits[:, :-1] vs x[:, 1:], all 977
validation chunks, batch 1), but sweeps r in {1, 2, 4, 8} and writes results
to runs/reproduction/d1_rscaling.json instead of overwriting the run's
evaluation.json.

r=8 bypass: looped.configure() validates 1 <= R <= 4, so configure with the
checkpoint's stored R=2 first, then set model.config.loop_repetitions = 8
directly. layer_schedule() reads the config with no validation, and
src/looped.py is untouched. Run: .venv/bin/python runs/reproduction/d1_rscaling.py
"""
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src import looped
from src.data import chunks
from src.ponder import ROOT, load_model

CKPT = "runs/EXP-25-loopmid160-s42/checkpoints/step-1200"
OUT = Path("runs/reproduction/d1_rscaling.json")
RS = [1, 2, 4, 8]


@torch.no_grad()
def nll_at_r(model, data, r):
    looped.configure(model, 6, 2, start=3)  # valid R; sets block/start
    if r != 2:
        model.config.loop_repetitions = r  # bypass the R<=4 check for r=8
    assert model.config.loop_repetitions == r
    assert model.config.loop_block_size == 6
    assert model.config.loop_block_start == 3
    total, count = 0.0, 0
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for x in data:
            x = x[None].cuda()
            out = looped.forward(model, x)
            total += F.cross_entropy(
                out.logits[:, :-1].float().reshape(-1, model.config.vocab_size),
                x[:, 1:].reshape(-1), reduction="sum").item()
            count += x.shape[1] - 1
    return total / count, count


@torch.no_grad()
def main():
    torch.manual_seed(42)
    model = load_model("vanilla", CKPT).eval()
    data = chunks("validation")
    assert tuple(data.shape) == (977, 256), tuple(data.shape)
    results = []
    for r in RS:
        start = time.perf_counter()
        nll, count = nll_at_r(model, data, r)
        elapsed = time.perf_counter() - start
        row = dict(r=r, nll=nll, ppl=math.exp(nll), predicted_tokens=count,
                   chunks=len(data), seconds=elapsed)
        results.append(row)
        print(f"r={r}  nll={nll:.6f}  ppl={math.exp(nll):.4f}  "
              f"tokens={count}  seconds={elapsed:.1f}", flush=True)
    out = dict(
        checkpoint=CKPT, split="validation", seq=256, batch=1, seed=42,
        method="parallel NLL, same computation as src/evaluate_looped.py "
               "(bf16 autocast, FP32 CE); r=8 via direct config override "
               "after configure(R=2); src/looped.py untouched",
        block_size=6, loop_block_start=3,
        git_hash=subprocess.check_output(["git", "rev-parse", "HEAD"],
                                         cwd=ROOT, text=True).strip(),
        torch=torch.__version__, gpu=torch.cuda.get_device_name(),
        results=results)
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
