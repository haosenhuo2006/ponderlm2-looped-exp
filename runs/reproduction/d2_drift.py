"""D2: hidden-state drift diagnosis on the EXP-25 looped checkpoint.

Existing src/diagnose.py / src/probe_h.py / src/jacobi_check.py are all
thought-line (ponder/Jacobi) tools: they load_model("ponder") and run the
interleaved thought schedule. They cannot run the looped forward, so this
script steps through the looped layers manually, replicating src/looped.py
forward() exactly (same embed_in, causal mask, rotary reuse, per-layer call
signature), and captures h after the prelude (round 0) and after each of 8
repetitions of the loop block layers 3..8 (rounds 1..8), before the coda.

Per round, averaged over the first N_CHUNKS validation chunks (batch 1,
bf16 autocast, FP32 metrics):
  - pairwise_cos: mean off-diagonal cosine similarity between token positions
    (256 tokens per chunk). -> 1.0 means all token states identical: the
    Huginn Bad Run 1 collapse signal.
  - prev_cos: mean per-token cosine similarity vs the previous round.
  - norm: mean per-token L2 norm.
  - rel_change: mean per-token ||h_i - h_{i-1}|| / ||h_{i-1}||.

Run: .venv/bin/python runs/reproduction/d2_drift.py
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data import chunks
from src.ponder import ROOT, load_model

CKPT = "runs/EXP-25-loopmid160-s42/checkpoints/step-1200"
OUT = Path("runs/reproduction/d2_drift.json")
N_CHUNKS = 32
MAX_ROUND = 8
START, BLOCK = 3, 6


@torch.no_grad()
def round_states(model, x):
    """Return [h0, h1, ..., h8]: h after prelude, then after each loop rep."""
    backbone = model.gpt_neox
    e = backbone.embed_in(x)
    cache_position = torch.arange(x.shape[1], device=x.device)
    positions = cache_position[None].expand(x.shape[0], -1)
    mask = backbone._update_causal_mask(torch.ones_like(x), e, cache_position,
                                        None, False)
    h = backbone.emb_dropout(e)
    rotary = backbone.rotary_emb(h, positions)
    layers = backbone.layers

    def run(layer, h):
        return layer(h, attention_mask=mask, position_ids=positions,
                     head_mask=None, layer_past=None, use_cache=False,
                     output_attentions=False, cache_position=cache_position,
                     position_embeddings=rotary)[0]

    states = []
    for layer in layers[:START]:
        h = run(layer, h)
    states.append(h.float())
    for _ in range(MAX_ROUND):
        for layer in layers[START:START + BLOCK]:
            h = run(layer, h)
        states.append(h.float())
    return states


def pairwise_cos(h):
    """Mean off-diagonal cosine similarity across token positions (1, T, d)."""
    u = F.normalize(h[0], dim=-1)
    gram = u @ u.T
    t = gram.shape[0]
    off = gram[~torch.eye(t, dtype=torch.bool, device=gram.device)]
    return off.mean().item()


@torch.no_grad()
def main():
    torch.manual_seed(42)
    model = load_model("vanilla", CKPT).eval()
    data = chunks("validation")[:N_CHUNKS]
    assert tuple(data.shape) == (N_CHUNKS, 256), tuple(data.shape)
    acc = None
    start = time.perf_counter()
    for x in data:
        x = x[None].cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            states = round_states(model, x)
        per_round = []
        prev = None
        for h in states:
            row = dict(pairwise_cos=pairwise_cos(h),
                       norm=h.norm(dim=-1).mean().item())
            if prev is not None:
                row["prev_cos"] = F.cosine_similarity(
                    h[0], prev[0], dim=-1).mean().item()
                row["rel_change"] = ((h - prev).norm(dim=-1)
                                     / prev.norm(dim=-1)).mean().item()
            else:
                row["prev_cos"] = None
                row["rel_change"] = None
            per_round.append(row)
            prev = h
        if acc is None:
            acc = per_round
        else:
            for a, b in zip(acc, per_round):
                for k, v in b.items():
                    if v is not None:
                        a[k] = a.get(k, 0.0) + v
        del states
    elapsed = time.perf_counter() - start
    rounds = []
    for i, a in enumerate(acc):
        row = {"round": i}
        for k, v in a.items():
            row[k] = None if v is None else v / N_CHUNKS
        rounds.append(row)
        pc = "  --  " if row["prev_cos"] is None else f"{row['prev_cos']:.6f}"
        rc = "  --  " if row["rel_change"] is None else f"{row['rel_change']:.6f}"
        print(f"round {i}: pairwise_cos={row['pairwise_cos']:.6f}  "
              f"prev_cos={pc}  norm={row['norm']:.4f}  rel_change={rc}",
              flush=True)
    out = dict(
        checkpoint=CKPT, split="validation", seq=256, batch=1, seed=42,
        n_chunks=N_CHUNKS, max_round=MAX_ROUND,
        loop_block="layers 3..8 (block_size 6, start 3), round 0 = after prelude",
        method="manual looped forward replicating src/looped.py forward(); "
               "bf16 autocast, FP32 metrics; src/diagnose.py, src/probe_h.py "
               "and src/jacobi_check.py are ponder/Jacobi tools and cannot "
               "run the looped forward, so not reused",
        git_hash=subprocess.check_output(["git", "rev-parse", "HEAD"],
                                         cwd=ROOT, text=True).strip(),
        torch=torch.__version__, gpu=torch.cuda.get_device_name(),
        seconds=elapsed, rounds=rounds)
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
