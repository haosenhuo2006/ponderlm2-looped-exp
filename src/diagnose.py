"""Inference-only manual Jacobi diagnostics using the unmodified official backbone."""
import argparse
import json
import math
import subprocess
import time
from pathlib import Path
import torch
import torch.nn.functional as F
from .ponder import load_model, embeddings, forward
from .masks import MASKS
from .data import chunks


def backbone(model, e, pos, mask, cache=None, use_cache=False):
    return model.gpt_neox(inputs_embeds=e, position_ids=pos,
                          attention_mask=None if mask is None else mask.to(device=e.device, dtype=e.dtype),
                          past_key_values=cache, use_cache=use_cache, return_dict=True)


def interleaved(model, e, h, condition):
    b, n, d = e.shape
    pos = torch.arange(n, device=e.device)[None].expand(b, -1)
    inp, pp, official_mask = model.prepare_gpt_inputs_for_stages(
        [e, h], b, n, d, e.device, pos, torch.ones_like(pos), None, model.config)
    return backbone(model, inp, pp, official_mask if condition == "D0" else MASKS[condition](n, 1))


def parallel_logits(model, x, condition):
    e = embeddings(model, x)
    pos = torch.arange(x.shape[1], device=x.device)[None]
    out = backbone(model, e, pos, torch.ones_like(pos) if condition == "D0" else MASKS[condition](x.shape[1], 0))
    h = model.extract_hidden_for_computation(out, slice(0, None, 1), model.config)
    # Same official schedule: initialization + 3 refinements + final logit pass.
    for _ in range(3):
        out = interleaved(model, e, h, condition)
        h = model.extract_hidden_for_computation(out, slice(0, None, 2), model.config)
    out = interleaved(model, e, h, condition)
    return model.embed_out(out.last_hidden_state[:, 1::2])


def sequential(model, x, condition):
    """Exact teacher-forced cached sequential path; select each global query row."""
    e = embeddings(model, x)
    mask = MASKS[condition](x.shape[1], 1)
    cache, total = None, 0.
    for i in range(x.shape[1] - 1):
        pos = torch.tensor([[i]], device=x.device)
        r = 2 * i
        out = backbone(model, e[:, i:i+1], pos, None if condition == "D0" else mask[:, :, r:r+1, :r+1], cache, True)
        out = backbone(model, out.last_hidden_state, pos,
                       None if condition == "D0" else mask[:, :, r+1:r+2, :r+2], out.past_key_values, True)
        cache = out.past_key_values
        total += F.cross_entropy(model.embed_out(out.last_hidden_state[:, -1]).float(),
                                 x[:, i+1], reduction="sum").item()
    return total, x.shape[1]-1


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--condition", choices=list(MASKS), required=True)
    p.add_argument("--split", choices=["validation"], default="validation")
    a = p.parse_args()
    torch.manual_seed(42)
    dest = Path("runs") / ("EXP-02-" + a.condition)
    dest.mkdir(parents=True, exist_ok=True)
    model = load_model("ponder", a.checkpoint).eval()
    data = chunks(a.split)
    assert tuple(data.shape) == (977, 256)
    baseline = Path("runs/EXP-02-D0/predictions.json")
    reference = json.loads(baseline.read_text()) if a.condition != "D0" else None
    with torch.autocast("cuda", dtype=torch.bfloat16):
        warm = parallel_logits(model, data[:1].cuda(), a.condition)
        parity = None
        if a.condition == "D0":
            official = forward(model, data[:1].cuda(), 3).logits
            parity = (warm.float()-official.float()).abs().max().item()
            assert parity < 1e-6, parity
        del warm
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        total, count, equal, predictions = 0., 0, 0, []
        for index, row in enumerate(data):
            x = row[None].cuda()
            logits = parallel_logits(model, x, a.condition)[:, :-1].float()
            total += F.cross_entropy(logits.reshape(-1, model.config.vocab_size),
                                     x[:, 1:].reshape(-1), reduction="sum").item()
            pred = logits.argmax(-1).flatten().tolist()
            predictions.append(pred)
            equal += len(pred) if reference is None else sum(
                u == v for u, v in zip(pred, reference[index]))
            count += len(pred)
            if index % 100 == 0:
                print(a.condition, index, "chunks", flush=True)
        torch.cuda.synchronize()
        elapsed = time.perf_counter()-start
        peak = torch.cuda.max_memory_allocated()
        sequential(model, data[:1, :16].cuda(), a.condition)
        torch.cuda.synchronize()
        t = time.perf_counter()
        sloss, sn = 0., 0
        for row in data[:8, :128]:
            loss, n = sequential(model, row[None].cuda(), a.condition)
            sloss += loss
            sn += n
        torch.cuda.synchronize()
        st = time.perf_counter()-t
    result = dict(condition=a.condition, checkpoint=a.checkpoint, split=a.split,
                  nll=total/count, ppl=math.exp(total/count), predicted_tokens=count,
                  chunks=len(data), seq=256, batch=1, seed=42, jacobi_iterations=3,
                  schedule="token-only initialization + 3 refinements + final thought-logit pass",
                  forward_path="manual official backbone/interleave/extraction",
                  precision="bf16 autocast; FP32 CE", seconds=elapsed, tokens_s=count/elapsed,
                  peak_memory_bytes=peak, top1_agreement=equal/count,
                  top1_change_rate=1-equal/count, sequential_nll_8x128=sloss/sn,
                  sequential_seconds=st, sequential_tokens_s=sn/st,
                  official_d0_max_abs_logit_difference=parity,
                  git_base=subprocess.check_output(["git","rev-parse","HEAD"], text=True).strip(),
                  torch=torch.__version__, gpu=torch.cuda.get_device_name(),
                  gpu_total_bytes=torch.cuda.get_device_properties(0).total_memory)
    (dest/"evaluation.json").write_text(json.dumps(result, indent=2)+"\n")
    if a.condition == "D0":
        baseline.write_text(json.dumps(predictions))
    (dest/"run.txt").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
