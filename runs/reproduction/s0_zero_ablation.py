"""s0=0 ablation for D3: split the +5.65 NLL zero-step damage into the
random-initial-state part and the sandwich-norms part.

Method -- NO src/ changes:
  evaluate_looped2._lm_validation calls looped2.forward without passing s0,
  so forward samples s0 via the module-global looped2.sample_s0. This script
  monkeypatches looped2.sample_s0 to return an all-zero tensor of the
  requested shape/dtype/device, then calls the official evaluator's
  _load_checkpoint / _lm_validation unchanged. Everything else (checkpoint,
  data, autocast, CE reduction) is identical to the D3 run, so the only
  difference between this and D3 is s0 = 0 instead of s0 ~ trunc-normal.

  r=1 is the key group: with s0=0 and r=1, the core input is the clean
  prelude output e, so the remaining damage vs the 4.3733 base is the
  sandwich-norms structural change alone. r=4 is a control (same as D3's
  main depth, but with s0=0).

Reference numbers (D3, random s0; validation, Pythia-160M):
  base = 4.3733
  random s0, r=1 = 12.0266 ; r=2 = 10.0770 ; r=4 = 10.0225 (seed 0)
"""

import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
os.environ.update(
    HF_HUB_OFFLINE="1",
    TRANSFORMERS_OFFLINE="1",
    PYTHIA_MODEL="EleutherAI/pythia-160M",
    PYTHIA_REVISION="50f5173d932e8e61f858120bcb800b97af589f46",
)
sys.path.insert(0, str(ROOT))

import torch

from src import looped2
from src.data import chunks
from src.evaluate_looped2 import _git_hash, _lm_validation, _load_checkpoint


def _zero_s0(shape, device, dtype, generator=None):
    # Ablation: zero initial state. Signature mirrors looped2.sample_s0.
    return torch.zeros(shape, device=device, dtype=dtype)


def main():
    # Sanity: the patch must actually produce zeros before we trust any NLL.
    probe = _zero_s0((2, 3), "cpu", torch.float32)
    assert probe.shape == (2, 3) and bool((probe == 0).all()), "zero-s0 probe failed"
    looped2.sample_s0 = _zero_s0
    assert looped2.sample_s0 is _zero_s0, "monkeypatch did not stick"

    checkpoint = ROOT / "runs/reproduction/d3-zero-step"
    model, cfg = _load_checkpoint(str(checkpoint))
    data = chunks("validation", seq=256)
    per_r = []
    for r in (1, 4):
        row = _lm_validation(model, cfg, data, r, 0)
        row["s0_mode"] = "zero"
        per_r.append(row)
        print(f"r={r}: NLL={row['nll']:.4f} PPL={row['ppl']:.2f}", flush=True)

    result = dict(
        mode="looped2-s0-zero",
        split="validation",
        checkpoint=str(checkpoint),
        config=asdict(cfg),
        per_r=per_r,
        git_hash=_git_hash(),
    )
    out = ROOT / "runs/reproduction/s0-zero-ablation.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
