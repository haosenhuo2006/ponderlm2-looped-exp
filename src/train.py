import hashlib
import platform
import importlib.metadata
import argparse
import json
import math
import random
import subprocess
import time
from pathlib import Path
import torch
from .data import chunks, order
from .ponder import load_model, forward
from .masked import forward as masked_forward
from . import looped

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--mask", choices=["D1", "D2"])
    a = p.parse_args()
    c = json.loads(Path(a.config).read_text())
    if a.mask is not None:
        c["mask"] = a.mask
    torch.manual_seed(c["seed"])
    rng = random.Random(c["seed"])
    dest = Path(c["run_dir"])
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "config.json").write_text(json.dumps(c, indent=2))
    model = load_model(c["mode"])
    if c.get("loop"):
        looped.configure(model, **c["loop"])
    data = chunks(seq=c["seq"])
    indices = order(len(data), c["steps"] * c["grad_accum"], c["seed"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=c["lr"], betas=tuple(c["betas"]), weight_decay=c["weight_decay"])
    model.train()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    metadata = dict(git_hash=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
                    torch=torch.__version__, gpu=torch.cuda.get_device_name(),
                    parameters=sum(p.numel() for p in model.parameters()), start_unix=time.time(),
                    python=platform.python_version(),
                    versions={n:importlib.metadata.version(n) for n in ["transformers","numpy","pyarrow","tokenizers"]},
                    gpu_total_bytes=torch.cuda.get_device_properties(0).total_memory,
                    source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in sorted(Path("src").glob("*.py"))})
    (dest / "metadata.json").write_text(json.dumps(metadata, indent=2))
    with (dest / "train.jsonl").open("w", buffering=1) as log:
        for step in range(1, c["steps"]+1):
            if time.perf_counter()-start > 3600:
                raise RuntimeError("60-minute run budget exceeded; refusing to shorten training silently")
            t = time.perf_counter()
            lr = c["lr"] * (step/c["warmup"] if step <= c["warmup"] else
                 0.1+0.9*0.5*(1+math.cos(math.pi*(step-c["warmup"])/(c["steps"]-c["warmup"]))))
            for group in optimizer.param_groups:
                group["lr"] = lr
            k = rng.choice([2,3]) if c["mode"] == "ponder" else 0
            optimizer.zero_grad(set_to_none=True)
            loss_sum = 0.
            for micro in range(c["grad_accum"]):
                x = data[indices[(step-1)*c["grad_accum"]+micro]].unsqueeze(0).cuda()
                if c.get("loop"):
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        loss = looped.forward(model, x, labels=x).loss
                elif c.get("mask"):
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        loss = masked_forward(model, x, k, c["mask"]).loss
                else:
                    loss = forward(model, x, k).loss
                if not torch.isfinite(loss):
                    raise RuntimeError("nonfinite loss")
                (loss / c["grad_accum"]).backward()
                loss_sum += loss.item()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), c["clip"], error_if_nonfinite=True)
            optimizer.step()
            torch.cuda.synchronize()
            row = dict(step=step, loss=loss_sum/c["grad_accum"], K=k, lr=lr,
                       tokens=step*c["seq"]*c["grad_accum"], tokens_s=c["seq"]*c["grad_accum"]/(time.perf_counter()-t),
                       grad_norm=float(norm), peak_memory_bytes=torch.cuda.max_memory_allocated(),
                       elapsed_s=time.perf_counter()-start)
            log.write(json.dumps(row)+"\n")
            print(json.dumps(row), flush=True)
            if step % 200 == 0 or step == c["steps"]:
                checkpoint = dest / "checkpoints" / ("step-"+str(step))
                model.save_pretrained(checkpoint)
                torch.save(dict(optimizer=optimizer.state_dict(), step=step, torch_rng=torch.get_rng_state(),
                                cuda_rng=torch.cuda.get_rng_state(), python_rng=rng.getstate()), checkpoint / "trainer.pt")
                with (dest / "checkpoint-index.jsonl").open("a") as index:
                    index.write(json.dumps(dict(step=step,path=str(checkpoint.resolve())))+"\n")
    (dest / "complete.json").write_text(json.dumps(dict(steps=c["steps"],wall_s=time.perf_counter()-start)))
if __name__ == "__main__":
    main()
