"""Inference-only logit-lens probe of latent thoughts (D0/D1)."""
import argparse
import json
from pathlib import Path
import subprocess
import time

import torch

from .data import chunks
from .diagnose import backbone, interleaved
from .ponder import embeddings, load_model


@torch.no_grad()
def probe_chunk(model, x, condition):
    """Probe one chunk; ratios cover only the shifted next-token targets.

    The model must be in eval mode. A fresh CPU generator makes the sequence
    shuffle reproducible, with the same permutation for each baseline state.
    """
    if condition not in ("D0", "D1"):
        raise ValueError("condition must be D0 or D1")
    if x.ndim != 2 or x.shape[0] != 1 or x.shape[1] < 2:
        raise ValueError("x must have shape (1, seq), with seq >= 2")
    with torch.autocast(x.device.type, dtype=torch.bfloat16):
        e = embeddings(model, x)
        pos = torch.arange(x.shape[1], device=x.device)[None]
        out = backbone(model, e, pos, torch.ones_like(pos))
        token_hidden = out.last_hidden_state
        h = model.extract_hidden_for_computation(out, slice(0, None, 1), model.config)
        states = [h]
        thoughts = []
        for _ in range(3):
            out = interleaved(model, e, h, condition)
            thoughts.append(out.last_hidden_state[:, 1::2])
            h = model.extract_hidden_for_computation(out, slice(0, None, 2), model.config)
            states.append(h)
        out = interleaved(model, e, h, condition)

        def top1(hidden):
            return model.embed_out(hidden)[:, :-1].float().argmax(dim=-1)

        final_pred = top1(out.last_hidden_state[:, 1::2])
        targets = x[:, 1:]

        def accuracy(pred, reference):
            return (pred == reference).sum().item() / targets.numel()

        def metrics(hidden):
            pred = top1(hidden)
            return dict(next_token_acc=accuracy(pred, targets),
                        final_agreement=accuracy(pred, final_pred))

        permutation = torch.randperm(x.shape[1], generator=torch.Generator().manual_seed(123))
        permutation = permutation.to(x.device)
        token_acc = accuracy(top1(token_hidden), targets)
        baselines = {"shuffled": {}, "token_hidden": {}, "zero": {}}
        for name, hidden in (("H0", states[0]), ("H3", states[3]),
                             ("round_3_thought", thoughts[2])):
            baselines["shuffled"][name] = dict(
                next_token_acc=accuracy(top1(hidden[:, permutation]), targets))
            baselines["token_hidden"][name] = dict(next_token_acc=token_acc)
            baselines["zero"][name] = dict(
                next_token_acc=accuracy(top1(torch.zeros_like(hidden)), targets))
        return dict(
            per_k={str(k): metrics(hidden) for k, hidden in enumerate(states)},
            round_thought={str(k): metrics(hidden) for k, hidden in enumerate(thoughts, 1)},
            baselines=baselines, final_logit_acc=accuracy(final_pred, targets))


def _accumulate(total, result, weight):
    """Accumulate nested metrics with an explicit shifted-token denominator."""
    for key, value in result.items():
        if isinstance(value, dict):
            _accumulate(total.setdefault(key, {}), value, weight)
        else:
            total[key] = total.get(key, 0.0) + value * weight


def _normalize(total, count):
    return {key: _normalize(value, count) if isinstance(value, dict) else value / count
            for key, value in total.items()}


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--condition", choices=("D0", "D1"), required=True)
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be positive")
    torch.manual_seed(42)
    model = load_model("ponder", args.checkpoint).eval()
    data = chunks("validation")
    assert tuple(data.shape) == (977, 256), tuple(data.shape)
    data = data[:args.limit]
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    total, count = {}, 0
    for row in data:
        x = row[None].cuda()
        result = probe_chunk(model, x, args.condition)
        weight = x[:, 1:].numel()
        _accumulate(total, result, weight)
        count += weight
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    result = dict(
        checkpoint=args.checkpoint, condition=args.condition, split="validation",
        n_chunks=len(data), seq=256, jacobi_iterations=3, seed=42, shuffle_seed=123,
        schedule="token-only initialization + 3 interleaved refinements + final thought-logit pass; H extracted from even/token positions",
        precision="bf16 autocast; FP32 logits for argmax comparisons",
        seconds=elapsed, peak_memory_bytes=torch.cuda.max_memory_allocated(),
        git_hash=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1], text=True).strip(),
        torch=torch.__version__, gpu=torch.cuda.get_device_name(),
        interpretation="logit-lens 只测可读性，一致率高不代表 h 被因果地使用了；低则说明 h 几乎没编码可读的预测信息。",
        **_normalize(total, count))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
