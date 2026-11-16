"""Train randomly initialized vanilla or looped Pythia on online addition."""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import platform
import random
import subprocess
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, GPTNeoXConfig

from . import looped
from .addition import encode_example, sample_operands
from .ponder import ROOT

# Pin the 70M structure/tokenizer independently of other experiments' env overrides.
MODEL = "EleutherAI/pythia-70m"
REVISION = "a39f36b100fe8a5377810d56c3f4789b9c53ac42"
VENDORED = ROOT / "vendor/PonderLM-2/src/llamafactory/model/modeling/modeling_gpt_neox.py"


def random_model(config=None):
    """Construct the vendored class directly; never fetch pretrained weights."""
    spec = importlib.util.spec_from_file_location("addition_gpt_neox", VENDORED)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if config is None:
        config = GPTNeoXConfig.from_pretrained(MODEL, revision=REVISION)
    for key, value in dict(num_latent_thoughts=0, num_jacobi_iterations=3,
                           random_jacobi_iterations=False, scale_embeds=False,
                           interpolation=False, recurrent_model=False,
                           use_cache=False).items():
        setattr(config, key, value)
    config._attn_implementation = "sdpa"
    return module.GPTNeoXForCausalLM(config).float()


def make_batch(tok, rng, batch, max_digits, device):
    examples = [encode_example(tok, *sample_operands(rng, max_digits))
                for _ in range(batch)]
    width = max(len(ids) for ids, _ in examples)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    x = torch.full((batch, width), pad, dtype=torch.long, device=device)
    labels = torch.full_like(x, -100)
    # Right pads are later than all real tokens: causal attention cannot let them
    # affect answer representations, and -100 excludes their targets from loss.
    for row, (ids, targets) in enumerate(examples):
        x[row, :len(ids)] = torch.tensor(ids, device=device)
        labels[row, :len(ids)] = torch.tensor(targets, device=device)
    return x, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    c = json.loads(Path(args.config).read_text())
    if c["steps"] < 1 or c["batch"] < 1 or not 0 <= c["warmup"] <= c["steps"]:
        raise ValueError("require positive steps/batch and 0 <= warmup <= steps")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("training requires a CUDA GPU with bf16 support")
    start = time.perf_counter()
    deadline = start + 3600

    def check_deadline():
        if time.perf_counter() > deadline:
            raise RuntimeError("60-minute run budget exceeded; refusing to shorten training silently")

    torch.manual_seed(c["seed"])
    random.seed(c["seed"])
    rng = random.Random(c["seed"])
    dest = Path(c["run_dir"])
    dest.mkdir(parents=True, exist_ok=True)
    # Avoid accidentally mixing two runs or overwriting an existing checkpoint.
    if any((dest / name).exists() for name in ("config.json", "train.jsonl", "checkpoints")):
        raise FileExistsError(f"run artifacts already exist in {dest}")
    (dest / "config.json").write_text(json.dumps(c, indent=2) + "\n")
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    model = random_model().cuda()
    looped.configure(model, **(c["loop"] or dict(block_size=2, repetitions=1, start=2)))
    optimizer = torch.optim.AdamW(model.parameters(), lr=c["lr"], betas=tuple(c["betas"]),
                                 weight_decay=c["weight_decay"])
    try:
        git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                           text=True, stderr=subprocess.DEVNULL).strip()
    except subprocess.CalledProcessError:
        git_hash = None  # Source-only copies may lack .git; hashes still identify code.
    sources = sorted((ROOT / "src").glob("*.py")) + [VENDORED]
    metadata = dict(git_hash=git_hash, torch=torch.__version__,
                    gpu=torch.cuda.get_device_name(), parameters=sum(p.numel() for p in model.parameters()),
                    python=platform.python_version(), start_unix=time.time(),
                    versions={n: importlib.metadata.version(n) for n in ("transformers", "tokenizers")},
                    gpu_total_bytes=torch.cuda.get_device_properties(0).total_memory,
                    model=MODEL, revision=REVISION, initialization="random", run_budget_s=3600,
                    source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in sources})
    (dest / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    model.train()
    with (dest / "train.jsonl").open("w", buffering=1) as log:
        for step in range(1, c["steps"] + 1):
            check_deadline()
            factor = (step / c["warmup"] if step <= c["warmup"] else
                      .1 + .9 * .5 * (1 + math.cos(math.pi * (step-c["warmup"]) /
                                                   (c["steps"]-c["warmup"]))))
            lr = c["lr"] * factor
            for group in optimizer.param_groups:
                group["lr"] = lr
            x, labels = make_batch(tok, rng, c["batch"], c["max_train_digits"], "cuda")
            optimizer.zero_grad(set_to_none=True)
            # Parameters and AdamW state stay FP32; only computation uses bf16.
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = looped.forward(model, x, labels=labels).loss
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            torch.cuda.synchronize()
            check_deadline()
            if step % 100 == 0 or step == c["steps"]:
                row = dict(step=step, loss=loss.item(), grad_norm=float(norm), lr=lr,
                           elapsed=time.perf_counter()-start)
                log.write(json.dumps(row) + "\n")
                print(json.dumps(row), flush=True)
    check_deadline()
    checkpoint = dest / "checkpoints" / "final"
    model.save_pretrained(checkpoint)
    torch.save(dict(optimizer=optimizer.state_dict(), step=c["steps"],
                    torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(),
                    cuda_rng_all=torch.cuda.get_rng_state_all(), python_rng=rng.getstate(),
                    global_python_rng=random.getstate()), checkpoint / "trainer.pt")
    check_deadline()
    with (dest / "checkpoint-index.jsonl").open("a") as index:
        index.write(json.dumps(dict(step=c["steps"], path=str(checkpoint.resolve()))) + "\n")
    (dest / "complete.json").write_text(json.dumps(dict(steps=c["steps"], wall_s=time.perf_counter()-start)))


if __name__ == "__main__":
    main()
