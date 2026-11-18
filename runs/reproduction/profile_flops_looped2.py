"""Forward-only profiler audit for the NEW looped2 graph.

Method mirrors scripts/profile_flops_step1.py (torch.profiler, with_flops=True,
sdpa MATH backend so attention matmuls are visible, bf16 autocast).
Primary check (design section 10): for fixed r in {1, 4, 12}, the measured
forward-FLOPs ratio vs vanilla must match the analytic forward ratio from
src.looped2.layer_flops_per_token within +/-1%. The profiler counts full
seq^2 causal attention while the analytic formula uses mean context 128, so
absolute per-token numbers are expected to run ~+2% high; that systematic
offset mostly cancels in the ratio and is reported, not asserted.
The backward (k-truncation) part of train_flops_per_token cannot be seen by a
forward profiler; it was verified by code review against the backward graph.
"""
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    os.chdir(ROOT)
    os.environ.update(
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        PYTHIA_MODEL="EleutherAI/pythia-160M",
        PYTHIA_REVISION="50f5173d932e8e61f858120bcb800b97af589f46",
    )
    sys.path.insert(0, str(ROOT))
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel

    from src.ponder import load_model
    from src import looped2

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"LOOPED2 profiler device={device}", flush=True)
    model = load_model("vanilla")
    model.eval()
    assert 160_000_000 <= sum(p.numel() for p in model.parameters()) <= 165_000_000
    h, ff, vocab = (model.config.hidden_size, model.config.intermediate_size,
                    model.config.vocab_size)
    assert (h, ff, vocab) == (768, 3072, 50304), (h, ff, vocab)
    x = torch.zeros((1, 256), dtype=torch.long, device=device)
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    layer, head = looped2.layer_flops_per_token(h, ff, vocab, context=128)

    def measure(forward, calls):
        with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
            with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
                with torch.profiler.profile(activities=activities, with_flops=True,
                                            record_shapes=True) as prof:
                    forward()
                    if device == "cuda":
                        torch.cuda.synchronize()
        events = prof.key_averages()
        total = sum(e.flops or 0 for e in events)
        dense = sum(e.flops or 0 for e in events if e.key in ("aten::mm", "aten::addmm"))
        attention = sum(e.flops or 0 for e in events if e.key == "aten::bmm")
        expected_dense = 256 * (calls * (8 * h * h + 4 * h * ff) + 2 * h * vocab)
        expected_attention = calls * 4 * 256 * 256 * h
        if (not math.isfinite(total) or total <= 0
                or abs(dense / expected_dense - 1) > .01
                or abs(attention / expected_attention - 1) > .01):
            raise RuntimeError(
                f"Untrustworthy profiler FLOPs coverage: total={total}, "
                f"dense={dense}, attention={attention}, calls={calls}")
        return total / 256

    vanilla = measure(lambda: model(input_ids=x, use_cache=False), 12)
    vanilla_analytic = 12 * layer + head
    cfg = looped2.resolve_config(model.config.num_hidden_layers,
                                 {"widths": [3, 6, 3]}, 1.0)
    looped2.apply_config(model, cfg)
    results = {
        "device": device,
        "microbatch": 1,
        "seq": 256,
        "vanilla_flops_per_token": vanilla,
        "vanilla_analytic_per_token": vanilla_analytic,
        "vanilla_absolute_deviation_pct": 100 * (vanilla / vanilla_analytic - 1),
        "expected_train_flops_ratio": looped2.expected_flops_ratio(),
        "r": {},
    }
    for r in (1, 4, 12):
        rng = torch.Generator().manual_seed(1234)
        measured = measure(
            lambda: looped2.forward(model, x, cfg=cfg, mode="eval",
                                    rng=rng, r_override=r),
            6 + 6 * r)
        analytic = (6 + 6 * r) * layer + head
        measured_ratio = measured / vanilla
        analytic_ratio = analytic / vanilla_analytic
        results["r"][str(r)] = {
            "measured_flops_per_token": measured,
            "analytic_flops_per_token": analytic,
            "absolute_deviation_pct": 100 * (measured / analytic - 1),
            "measured_ratio_vs_vanilla": measured_ratio,
            "analytic_ratio_vs_vanilla": analytic_ratio,
            "ratio_deviation_pct": 100 * (measured_ratio / analytic_ratio - 1),
        }
        print(f"r={r}: measured={measured:.1f} analytic={analytic:.1f} "
              f"abs_dev={100 * (measured / analytic - 1):+.3f}% "
              f"ratio_dev={100 * (measured_ratio / analytic_ratio - 1):+.3f}%", flush=True)
    ok = all(abs(v["ratio_deviation_pct"]) < 1 for v in results["r"].values())
    results["pass_ratio_lt_1pct"] = ok
    (ROOT / "runs/reproduction").mkdir(exist_ok=True)
    (ROOT / "runs/reproduction/profile-flops-looped2.json").write_text(
        json.dumps(results, indent=2) + "\n")
    print("PASS" if ok else "FAIL", flush=True)
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
