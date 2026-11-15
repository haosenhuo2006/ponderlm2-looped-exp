"""Evaluation-only refinement scan; never overwrite the primary evaluation."""
import argparse
import json
import math
import sys
import time
import zipfile
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
from src.data import chunks
from src.ponder import ROOT, load_model
from src.masked import forward_eval


def parse_rounds(value):
    """Argparse type for comma-separated refinement counts in [0, 5]."""
    try:
        rounds = [int(part) for part in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("rounds must be comma-separated integers from 0 to 5") from exc
    if any(r not in range(6) for r in rounds):
        raise argparse.ArgumentTypeError("rounds must be integers from 0 to 5")
    if len(set(rounds)) != len(rounds):
        raise argparse.ArgumentTypeError("rounds must not contain duplicates")
    return rounds


def validation_chunks():
    """Use the standard chunks reader without downloads or provenance writes."""
    data = ROOT / "data"
    if not (data / "validation-tokens.npy").is_file():
        raise FileNotFoundError("K scan requires the existing validation token cache")
    if not zipfile.is_zipfile(data / "wikitext-2-raw-v1.zip") and not (data / "validation.parquet").is_file():
        raise FileNotFoundError("K scan requires the existing validation source")
    # chunks() otherwise creates runs/ and rewrites its provenance JSON even
    # when the token cache exists. No other writer is used on this cached path.
    with patch("src.data.Path.mkdir"), patch("src.data.Path.write_text"):
        data = chunks("validation")
    assert tuple(data.shape) == (977, 256), tuple(data.shape)
    return data


def final_ce_sum(logits, x):
    """The standard evaluation's FP32 shifted sum-CE and target count."""
    loss = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                           x[:, 1:].reshape(-1), reduction="sum")
    return loss.item(), x[:, 1:].numel()


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--rounds", type=parse_rounds, default="0,1,2,3,4,5")
    p.add_argument("--mask", choices=["D0", "D1", "D2"], default="D0")
    a = p.parse_args()
    output = Path(a.output)
    # resolve() also protects aliases/symlinks to the primary evaluation file.
    recorded_path = Path(a.checkpoint).absolute().parents[1] / "evaluation.json"
    if output.name == "evaluation.json" or output.resolve().name == "evaluation.json" or output.resolve() == recorded_path.resolve():
        p.error("--output must not overwrite evaluation.json")
    if output.exists() and recorded_path.exists() and output.samefile(recorded_path):
        p.error("--output must not overwrite evaluation.json")
    torch.manual_seed(42)
    model = load_model("ponder", a.checkpoint).eval()
    data = validation_chunks()
    nlls, ppls, seconds = {}, {}, {}
    torch.cuda.synchronize()
    start = time.perf_counter()
    for r in a.rounds:
        # Match the standard evaluation's untimed warm-up for each schedule.
        with torch.autocast("cuda", dtype=torch.bfloat16):
            warm = forward_eval(model, data[:1].cuda(), r, a.mask)
        del warm
        torch.cuda.synchronize()
        round_start = time.perf_counter()
        total, count = 0., 0
        for row in data:
            x = row[None].cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = forward_eval(model, x, r, a.mask)
            loss, num = final_ce_sum(out.logits, x)
            total += loss
            count += num
            del out
        torch.cuda.synchronize()
        assert count == 249135, count
        key = str(r)
        seconds[key] = time.perf_counter()-round_start
        nlls[key] = total/count
        ppls[key] = math.exp(nlls[key])
    elapsed = time.perf_counter()-start
    result = dict(checkpoint=a.checkpoint, split="validation", chunks=len(data), seq=256,
                  predicted_tokens=count, mask=a.mask, rounds=nlls, ppl_per_round=ppls,
                  seconds_per_round=seconds, total_seconds=elapsed,
                  round_definition="R interleaved refinements + one separate final prediction pass; R=0: token-only initialization followed directly by final pass",
                  precision="bf16 autocast; FP32 CE", torch=torch.__version__,
                  gpu=torch.cuda.get_device_name())
    if 3 in a.rounds and recorded_path.exists():
        recorded = json.loads(recorded_path.read_text())["nll"]
        difference = abs(nlls["3"]-recorded)
        result.update(recorded_nll=recorded, nll_R3_abs_difference=difference)
        if not difference < 1e-6:
            raise ValueError(f"R=3 NLL mismatch: measured={nlls['3']}, recorded={recorded}, abs_difference={difference}")
    text = json.dumps(result, indent=2)+"\n"
    output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
