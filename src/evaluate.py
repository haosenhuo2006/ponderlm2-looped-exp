"""Parallel validation plus separately timed sequential inference."""
import argparse
from contextlib import nullcontext
import json
import math
import time
from pathlib import Path
import torch
import torch.nn.functional as F
from . import looped
from .ponder import load_model, forward, embeddings, hidden
from .data import chunks
from .masked import forward as masked_forward
from .diagnose import sequential as masked_sequential
from .evaluate_looped import sequential as looped_sequential


def detect_loop_config(config, *, mode=None, checkpoint=None):
    block_size = getattr(config, "loop_block_size", None)
    repetitions = getattr(config, "loop_repetitions", None)
    start = getattr(config, "loop_block_start", None)
    if block_size is None and repetitions is None and start is None:
        if mode == "looped":
            raise ValueError(
                f"Checkpoint {checkpoint!r} has no loop fields for --mode looped. "
                "Check that the checkpoint was saved with its loop configuration."
            )
        return None
    missing = [field for field, value in (("loop_block_size", block_size),
                                          ("loop_repetitions", repetitions))
               if value is None]
    if missing:
        raise ValueError(f"Checkpoint {checkpoint!r} is missing loop fields: {', '.join(missing)}")
    return dict(block_size=block_size, repetitions=repetitions, start=start)


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
    p.add_argument("--mode", choices=["vanilla","ponder","looped"], required=True)
    p.add_argument("--split", default="validation", choices=["validation"])
    p.add_argument("--mask", choices=["D0", "D1", "D2"])
    a = p.parse_args()
    torch.manual_seed(42)
    model = load_model("vanilla" if a.mode == "looped" else a.mode, a.checkpoint).eval()
    loop_config = detect_loop_config(model.config, mode=a.mode, checkpoint=a.checkpoint)
    if loop_config is not None:
        looped.configure(model, **loop_config)
        if a.mask is not None:
            p.error("--mask is unsupported for looped checkpoints")

    def run_forward(model, x, k):
        if loop_config is not None:
            return looped.forward(model, x, labels=x)
        if a.mask is None:
            return forward(model, x, k)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return masked_forward(model, x, k, a.mask)

    def run_sequential(model, x):
        if loop_config is not None:
            loss, n = looped_sequential(model, x)
            return loss, n, 0
        if a.mask is None:
            return sequential(model, x)
        cache_holder = []
        def capture_cache(module, args, out):
            cache_holder[:] = [out.past_key_values]
        hook = model.gpt_neox.register_forward_hook(capture_cache)
        try:
            loss, n = masked_sequential(model, x, a.mask)
        finally:
            hook.remove()
        # Count actual tensors: rotary keys and values can have different dtypes.
        cache_bytes = sum(t.numel()*t.element_size()
                          for layer in cache_holder[0] for t in layer)
        return loss, n, cache_bytes
    data = chunks(a.split)
    # Untimed warm-up, then synchronize around the whole validation pass.
    parallel_context = (torch.autocast("cuda", dtype=torch.bfloat16)
                        if loop_config is not None else nullcontext())
    with parallel_context:
        run_forward(model, data[0:1].cuda(), 3)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        total, count = 0., 0
        for x in data:
            x = x.unsqueeze(0).cuda()
            out = run_forward(model, x, 3)
            # Float32 CE over next-token targets.
            total += F.cross_entropy(out.logits[:,:-1].float().reshape(-1,model.config.vocab_size),
                                     x[:,1:].reshape(-1), reduction="sum").item()
            count += x.shape[1]-1
        torch.cuda.synchronize()
        elapsed = time.perf_counter()-start
    result = dict(mode="looped" if loop_config is not None else a.mode,
                  checkpoint=a.checkpoint,split=a.split,nll=total/count,ppl=math.exp(total/count),
                  predicted_tokens=count,chunks=len(data),seq=256,jacobi_iterations=3,
                  seconds=elapsed,tokens_s=count/elapsed,peak_memory_bytes=torch.cuda.max_memory_allocated())
    # Sequential inference on 8 fixed validation sequences.
    with torch.autocast("cuda", dtype=torch.bfloat16):
        run_sequential(model, data[:1,:16].cuda())
        torch.cuda.synchronize()
        t = time.perf_counter()
        nll, n, cache_bytes = 0., 0, 0
        for x in data[:8,:128]:
            loss, num, cache_bytes = run_sequential(model,x.unsqueeze(0).cuda())
            nll += loss
            n += num
        torch.cuda.synchronize()
    result.update(sequential_nll_8x128=nll/n,sequential_tokens_s=n/(time.perf_counter()-t),kv_cache_bytes_127_tokens=cache_bytes)
    c = model.config
    # Approximate multiply-add=2 dense transformer FLOPs, excluding embedding lookup.
    d, layers, ff, vocab = c.hidden_size,c.num_hidden_layers,c.intermediate_size,c.vocab_size
    if loop_config is not None:
        result.pop("jacobi_iterations")
        result.update(
            loop_block_start=c.loop_block_start,
            loop_block_size=c.loop_block_size,
            loop_repetitions=c.loop_repetitions,
            forward_path="looped.forward (restored from checkpoint config)",
            inference_flops_per_token_estimate=looped.flops(
                c.loop_block_size, c.loop_repetitions, d=d, ff=ff, layers=layers),
            flops_convention=looped.CONVENTION,
        )
    else:
        factor = 2 if a.mode == "ponder" else 1
        transformer = layers*(8*d*d+4*d*ff)
        result["inference_flops_per_token_estimate"] = factor*transformer + 2*d*vocab + 4*layers*d*factor*factor*128
        result["flops_convention"] = "2 FLOPs/MAC; cached context mean 128 real tokens; transformer dense + causal QK/AV + one vocab head; excludes norms/softmax"
    if a.mask is not None:
        result.update(mask=a.mask, forward_path="manual official backbone/interleave/extraction",
                      diagnostic=a.mask == "D0", batch=1, precision="bf16 autocast; FP32 CE")
    filename = "evaluation-D0.json" if a.mask == "D0" else "evaluation.json"
    path = Path(a.checkpoint).parents[1] / filename
    path.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
if __name__ == "__main__":
    main()
