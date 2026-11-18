"""Independent, offline forward-only profiler audit; no analytical fallback."""
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIELDS = {"device", "microbatch", "seq", "vanilla_flops_per_token",
          "looped_flops_per_token", "measured_ratio", "analytical_ratio",
          "deviation_pct", "within_5pct"}


def validate_profile(data):
    assert FIELDS <= data.keys()
    assert data["device"] in ("cpu", "cuda")
    assert data["microbatch"] == 1 and data["seq"] == 256
    for key in ("vanilla_flops_per_token", "looped_flops_per_token", "measured_ratio"):
        assert math.isfinite(data[key]) and data[key] > 0, key
    assert data["analytical_ratio"] == 1.3466
    ratio = data["looped_flops_per_token"] / data["vanilla_flops_per_token"]
    assert math.isclose(data["measured_ratio"], ratio, rel_tol=1e-12)
    deviation = 100 * (ratio / 1.3466 - 1)
    assert math.isfinite(data["deviation_pct"])
    assert math.isclose(data["deviation_pct"], deviation, abs_tol=1e-10)
    assert type(data["within_5pct"]) is bool
    assert data["within_5pct"] == (abs(deviation) <= 5)
    return data


def main():
    os.chdir(ROOT)
    config = json.loads((ROOT / "configs/step1_t0_s42.json").read_text())
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      PYTHIA_MODEL=config["model"], PYTHIA_REVISION=config["revision"])
    sys.path.insert(0, str(ROOT))
    import torch
    from unittest.mock import patch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from src.ponder import load_model
    from src import looped

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"STEP1 profiler device={device}", flush=True)
    # load_model unconditionally calls cuda(); preserve its official constructor
    # and weights path while adapting only that transfer for CPU-only hosts.
    if device == "cpu":
        with patch.object(torch.nn.Module, "cuda", lambda self: self.to("cpu")):
            model = load_model("vanilla")
    else:
        model = load_model("vanilla")
    model.eval()
    assert 160_000_000 <= sum(p.numel() for p in model.parameters()) <= 165_000_000
    x = torch.zeros((1, 256), dtype=torch.long, device=device)
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    def measure(forward, calls):
        # Expose attention matmuls to the profiler rather than accepting opaque
        # fused SDPA kernels whose FLOPs would silently be omitted.
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
        # Coverage checks only: these analytical counts never become outputs.
        expected_dense = 256 * (calls * (8*768*768 + 4*768*3072) + 2*768*50304)
        expected_attention = calls * 4 * 256 * 256 * 768
        if (not math.isfinite(total) or total <= 0
                or abs(dense / expected_dense - 1) > .01
                or abs(attention / expected_attention - 1) > .01):
            raise RuntimeError(f"Untrustworthy profiler FLOPs coverage: total={total}, dense={dense}, attention={attention}")
        return total / 256

    vanilla = measure(lambda: model(input_ids=x, use_cache=False), 12)
    looped.configure(model, block_size=6, repetitions=2, start=3)
    repeated = measure(lambda: looped.forward(model, x), 18)
    ratio = repeated / vanilla
    deviation = 100 * (ratio / 1.3466 - 1)
    result = validate_profile(dict(device=device, microbatch=1, seq=256,
        vanilla_flops_per_token=vanilla, looped_flops_per_token=repeated,
        measured_ratio=ratio, analytical_ratio=1.3466,
        deviation_pct=deviation, within_5pct=abs(deviation) <= 5))
    (ROOT / "runs").mkdir(exist_ok=True)
    (ROOT / "runs/step1-flops-profile.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    annotation = "" if result["within_5pct"] else " ANNOTATION REQUIRED: budget ratio not matched within ±5%"
    print(f"device={device}; measured ratio={ratio:.6f}; deviation={deviation:+.3f}%{annotation}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"STEP1 profiler ERROR: measured FLOPs unavailable or untrustworthy: {error}", file=sys.stderr)
        raise SystemExit(1)
