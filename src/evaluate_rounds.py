"""Full-split, fixed-K T1/T5 refinement diagnostics; never overwrite primary eval."""
import argparse
import json
import time
from pathlib import Path
import torch
import torch.nn.functional as F
from .data import chunks
from .ponder import load_model, forward as official
from .masked import forward


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    torch.manual_seed(42)
    model = load_model("ponder", a.checkpoint).eval()
    data = chunks("validation")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = forward(model, data[:1].cuda(), 3, "D0", return_round_logits=True)
        ref = official(model, data[:1].cuda(), 3)
        diff = (out.logits.float()-ref.logits.float()).abs().max().item()
        assert diff < 1e-6, diff
    del out, ref
    torch.cuda.synchronize()
    start = time.perf_counter()
    totals, final, count = [0., 0., 0.], 0., 0
    for i, row in enumerate(data):
        x = row[None].cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = forward(model, x, 3, "D0", return_round_logits=True)
        for k, z in enumerate(out.round_logits):
            totals[k] += F.cross_entropy(z[:, :-1].float().reshape(-1, z.shape[-1]),
                                        x[:, 1:].reshape(-1), reduction="sum").item()
        final += F.cross_entropy(out.logits[:, :-1].float().reshape(-1, out.logits.shape[-1]),
                                 x[:, 1:].reshape(-1), reduction="sum").item()
        count += x.shape[1]-1
        if i % 100 == 0:
            print(i, "chunks", flush=True)
    torch.cuda.synchronize()
    result = dict(checkpoint=a.checkpoint, split="validation", chunks=len(data), seq=256,
                  batch=1, predicted_tokens=count, K=3, round_nll=[v/count for v in totals],
                  final_prediction_nll=final/count, seconds=time.perf_counter()-start,
                  official_final_max_abs_logit_difference=diff,
                  round_definition="kth interleaved refinement thought logits; excludes token-only initialization and separate final prediction",
                  precision="bf16 autocast; FP32 CE")
    Path(a.output).write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
