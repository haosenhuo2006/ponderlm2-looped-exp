"""EXP-01 validation convention; exact causal prefix recomputation for T4."""
import argparse
import json
import math
import time
from pathlib import Path
import torch
import torch.nn.functional as F
from .ponder import load_model
from .data import chunks
from . import looped


@torch.no_grad()
def sequential(model, x):
    total = 0.
    for i in range(x.shape[1]-1):
        out = looped.forward(model, x[:, :i+1], last_only=True)
        total += F.cross_entropy(out.logits[:, -1].float(), x[:, i+1]).item()
    return total, x.shape[1]-1


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    a = p.parse_args()
    torch.manual_seed(42)
    model = load_model("vanilla", a.checkpoint).eval()
    looped.configure(model, model.config.loop_block_size, model.config.loop_repetitions)
    data = chunks("validation")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        looped.forward(model, data[:1].cuda())
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        total, count = 0., 0
        for x in data:
            x = x[None].cuda()
            out = looped.forward(model, x)
            total += F.cross_entropy(out.logits[:, :-1].float().reshape(-1, model.config.vocab_size),
                                     x[:, 1:].reshape(-1), reduction="sum").item()
            count += x.shape[1]-1
        torch.cuda.synchronize()
        elapsed = time.perf_counter()-start
        parallel_peak = torch.cuda.max_memory_allocated()
        del out
        sequential(model, data[:1, :16].cuda())
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        seq_total, seq_count = 0., 0
        for x in data[:8, :128]:
            loss, n = sequential(model, x[None].cuda())
            seq_total += loss
            seq_count += n
        torch.cuda.synchronize()
        seq_elapsed = time.perf_counter()-start
    c = model.config
    result = dict(mode="looped", checkpoint=a.checkpoint, split="validation", seq=256, batch=1,
                  block_size=c.loop_block_size, repetitions=c.loop_repetitions,
                  layers=list(range(c.num_hidden_layers-c.loop_block_size, c.num_hidden_layers)),
                  nll=total/count, ppl=math.exp(total/count), predicted_tokens=count, chunks=len(data),
                  seconds=elapsed, tokens_s=count/elapsed, latency_ms_per_token=1000*elapsed/count,
                  peak_memory_bytes=parallel_peak, sequential_nll_8x128=seq_total/seq_count,
                  sequential_predicted_tokens=seq_count, sequential_seconds=seq_elapsed,
                  sequential_tokens_s=seq_count/seq_elapsed,
                  sequential_latency_ms_per_token=1000*seq_elapsed/seq_count,
                  sequential_peak_memory_bytes=torch.cuda.max_memory_allocated(),
                  sequential_method="full-context forward per position; last-position vocab head; no KV cache",
                  inference_flops_per_token_estimate=looped.flops(c.loop_block_size,c.loop_repetitions),
                  flops_convention=looped.CONVENTION,
                  flops_scope="hypothetical cached sequential arithmetic; measured prefix recomputation costs more",
                  forward_path="thin wrapper calling original official backbone layer modules directly",
                  precision="bf16 autocast; FP32 CE")
    # Actual prefix-recomputation analytical arithmetic, averaged over prefixes 1..127.
    calls = c.num_hidden_layers+c.loop_block_size*(c.loop_repetitions-1)
    d, ff, vocab = c.hidden_size, c.intermediate_size, c.vocab_size
    result["measured_prefix_flops_per_output_estimate"] = sum(
        calls*(n*(8*d*d+4*d*ff)+2*d*n*(n+1))+2*d*vocab for n in range(1,128))/127
    path = Path(a.checkpoint).parents[1]/"evaluation.json"
    path.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
