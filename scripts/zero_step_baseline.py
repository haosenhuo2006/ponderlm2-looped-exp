"""Zero-training parallel validation of three architectures on cached weights."""
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("PYTHIA_MODEL", "EleutherAI/pythia-160M")
os.environ.setdefault("PYTHIA_REVISION", "50f5173d932e8e61f858120bcb800b97af589f46")
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch
import torch.nn.functional as F
from src.ponder import load_model, forward, MODEL, REVISION
from src.data import chunks
from src import looped


@torch.no_grad()
def main():
    destination = ROOT / "runs/zero-step-baseline"
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / "zero-step-baseline.json"
    if path.exists():
        raise FileExistsError(path)
    # chunks writes provenance here rather than into any existing run.
    os.environ["EXP08_RUN_DIR"] = str(destination)
    # Refuse the download paths in chunks(); the dataset must already exist.
    data_dir = ROOT / "data"
    if not ((data_dir / "wikitext-2-raw-v1.zip").exists() or
            (data_dir / "validation.parquet").exists()):
        raise FileNotFoundError("cached validation source required")
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    data = chunks("validation")
    if tuple(data.shape) != (977, 256):
        raise ValueError(f"unexpected validation shape: {tuple(data.shape)}")
    variants = []
    for mode in ("vanilla", "looped", "thought"):
        torch.manual_seed(42)
        model = load_model("ponder" if mode == "thought" else "vanilla").eval()
        if mode == "looped":
            looped.configure(model, 6, 2, start=3)
        def run(x):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                return looped.forward(model, x) if mode == "looped" else forward(model, x, 3)
        run(data[:1].cuda())
        torch.cuda.synchronize()
        begin = time.perf_counter()
        total, count = 0., 0
        for row in data:
            x = row.unsqueeze(0).cuda()
            out = run(x)
            total += F.cross_entropy(
                out.logits[:, :-1].float().reshape(-1, model.config.vocab_size),
                x[:, 1:].reshape(-1), reduction="sum").item()
            count += x.shape[1] - 1
        torch.cuda.synchronize()
        variants.append(dict(mode=mode, nll=total/count, ppl=math.exp(total/count),
                             predicted_tokens=count, chunks=len(data),
                             seconds=time.perf_counter()-begin))
        del out, model
        torch.cuda.empty_cache()
    result = dict(model=MODEL, revision=REVISION,
                  loop=dict(block_size=6, repetitions=2, start=3),
                  git_hash=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                  hardware=torch.cuda.get_device_name(), torch_version=torch.__version__,
                  seed=42, batch=1, seq=256, jacobi_iterations=3,
                  precision="bf16 autocast; FP32 CE", variants=variants)
    encoded = json.dumps(result, indent=2) + "\n"
    path.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
