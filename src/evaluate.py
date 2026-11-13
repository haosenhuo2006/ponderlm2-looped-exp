"""Parallel Jacobi validation plus separately timed cached sequential inference."""
import argparse
import json
import math
import time
from pathlib import Path
import torch
import torch.nn.functional as F
from .ponder import load_model, forward, embeddings, hidden
from .data import chunks

@torch.no_grad()
def sequential(model, x):
    """Teacher-forced causal decoding: x_i -> h_i -> next-token logits."""
    cache = None
    losses = []
    e = embeddings(model, x)
    for i in range(x.shape[1]-1):
        pos = torch.tensor([[i]], device=x.device)
        out = hidden(model, e[:,i:i+1], pos, cache, True)
        cache = out.past_key_values
        if model.config.num_latent_thoughts:
            out = hidden(model, out.last_hidden_state, pos, cache, True)
            cache = out.past_key_values
        logits = model.embed_out(out.last_hidden_state[:, -1]).float()
        losses.append(F.cross_entropy(logits, x[:,i+1]).item())
    # Legacy tuple cache; count actual tensor bytes.
    cache_bytes = sum(t.numel()*t.element_size() for layer in cache for t in layer)
    return sum(losses), len(losses), cache_bytes

@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--mode", choices=["vanilla","ponder"], required=True)
    p.add_argument("--split", default="validation", choices=["validation"])
    a = p.parse_args()
    torch.manual_seed(42)
    model = load_model(a.mode, a.checkpoint).eval()
    data = chunks(a.split)
    # Untimed warm-up, then synchronize around the whole validation pass.
    forward(model, data[0:1].cuda(), 3)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    total, count = 0., 0
    for x in data:
        x = x.unsqueeze(0).cuda()
        out = forward(model, x, 3)
        # Float32 CE from the official thought-position/vanilla logits.
        total += F.cross_entropy(out.logits[:,:-1].float().reshape(-1,model.config.vocab_size),
                                 x[:,1:].reshape(-1), reduction="sum").item()
        count += x.shape[1]-1
    torch.cuda.synchronize()
    elapsed = time.perf_counter()-start
    result = dict(mode=a.mode,checkpoint=a.checkpoint,split=a.split,nll=total/count,ppl=math.exp(total/count),
                  predicted_tokens=count,chunks=len(data),seq=256,jacobi_iterations=3,
                  seconds=elapsed,tokens_s=count/elapsed,peak_memory_bytes=torch.cuda.max_memory_allocated())
    # 8 fixed validation sequences: true sequential cached inference, same code path.
    with torch.autocast("cuda", dtype=torch.bfloat16):
        sequential(model, data[:1,:16].cuda())
        torch.cuda.synchronize()
        t = time.perf_counter()
        nll, n, cache_bytes = 0., 0, 0
        for x in data[:8,:128]:
            loss, num, cache_bytes = sequential(model,x.unsqueeze(0).cuda())
            nll += loss
            n += num
        torch.cuda.synchronize()
    result.update(sequential_nll_8x128=nll/n,sequential_tokens_s=n/(time.perf_counter()-t),kv_cache_bytes_127_tokens=cache_bytes)
    c = model.config
    # Approximate multiply-add=2 dense transformer FLOPs, excluding embedding lookup.
    d, layers, ff, vocab = c.hidden_size,c.num_hidden_layers,c.intermediate_size,c.vocab_size
    factor = 2 if a.mode == "ponder" else 1
    transformer = layers*(8*d*d+4*d*ff)
    result["inference_flops_per_token_estimate"] = factor*transformer + 2*d*vocab + 4*layers*d*factor*factor*128
    result["flops_convention"] = "2 FLOPs/MAC; cached context mean 128 real tokens; transformer dense + causal QK/AV + one vocab head; excludes norms/softmax"
    path = Path(a.checkpoint).parents[1] / "evaluation.json"
    path.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
if __name__ == "__main__":
    main()
