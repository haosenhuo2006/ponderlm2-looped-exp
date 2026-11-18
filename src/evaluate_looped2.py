"""Evaluate the looped2 reproduction on WikiText-2 or decimal addition."""

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import statistics
import subprocess

import torch
from torch.nn import functional as F
from transformers import AutoTokenizer, GPTNeoXConfig

from . import addition, looped2
from .data import chunks
from .ponder import ROOT, load_model
from .train_addition import MODEL as ADDITION_MODEL, REVISION as ADDITION_REVISION


def _git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _load_checkpoint(checkpoint, *, allow_unscaled_vanilla=False):
    config = GPTNeoXConfig.from_pretrained(checkpoint)
    cfg = looped2.config_from_checkpoint(config)
    if cfg is not None:
        return looped2.load_looped2_checkpoint(checkpoint), cfg
    if getattr(config, "loop_repetitions", 1) != 1:
        raise ValueError("legacy looped checkpoints require the legacy evaluator")
    if not allow_unscaled_vanilla and not hasattr(config, "looped2_emb_scale"):
        raise ValueError("expected a looped2 or scaled vanilla checkpoint")
    return load_model("vanilla", checkpoint).eval(), None


@torch.no_grad()
def _lm_validation(model, cfg, data, r, s0_seed):
    # D5: restart the stream for every depth so each chunk draws the same s0.
    generator = torch.Generator().manual_seed(s0_seed)
    total, count = 0.0, 0
    for row in data:
        x = row.unsqueeze(0).cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if cfg is None:
                out = looped2.plain_forward(model, x, emb_scale=model.config.looped2_emb_scale)
            else:
                out = looped2.forward(model, x, cfg=cfg, mode="eval", rng=generator, r_override=r)
        logits = out.logits[:, :-1].float().reshape(-1, model.config.vocab_size)
        total += F.cross_entropy(logits, x[:, 1:].reshape(-1), reduction="sum").item()
        count += x.shape[1] - 1
    nll = total / count
    return dict(r=r, nll=nll, ppl=math.exp(nll), predicted_tokens=count,
                chunks=len(data), s0_seed=s0_seed)


def evaluate_lm(args, checkpoint):
    model, cfg = _load_checkpoint(checkpoint)
    data = chunks(args.split, seq=256)
    depths = args.eval_r if cfg is not None else [1]
    config = asdict(cfg) if cfg is not None else dict(emb_scale=model.config.looped2_emb_scale)
    result = dict(mode="looped2" if cfg is not None else "vanilla", split=args.split, config=config,
                  per_r=[_lm_validation(model, cfg, data, r, args.s0_seed) for r in depths])
    if args.s0_seeds:
        main_r = cfg.eval_r if cfg is not None and cfg.eval_r in depths else depths[0]
        losses = [_lm_validation(model, cfg, data, main_r, seed)["nll"] for seed in args.s0_seeds]
        result["s0_variance"] = dict(r=main_r, seeds=args.s0_seeds, nll=losses,
                                     mean=statistics.mean(losses), std=statistics.pstdev(losses))
    return result


@torch.no_grad()
def _greedy_vanilla(model, tok, a, b, max_new_tokens):
    device = next(model.parameters()).device
    x = torch.tensor([tok.encode(f"{a}+{b}=", add_special_tokens=False)],
                     dtype=torch.long, device=device)
    generated = []
    with torch.autocast(device.type, dtype=torch.bfloat16):
        for _ in range(max_new_tokens):
            token = model(input_ids=x).logits[0, -1].argmax().item()
            if token == tok.eos_token_id:
                break
            generated.append(token)
            x = torch.cat((x, x.new_tensor([[token]])), dim=1)
    return tok.decode(generated, skip_special_tokens=True).strip()


@torch.no_grad()
def _greedy_looped2(model, cfg, tok, a, b, r, seed, max_new_tokens):
    x = torch.tensor([tok.encode(f"{a}+{b}=", add_special_tokens=False)],
                     dtype=torch.long, device="cuda")
    # D5: each example has its own stream, independent of other examples' EOS.
    generator = torch.Generator().manual_seed(seed)
    generated = []
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for _ in range(max_new_tokens):
            logits = looped2.forward(model, x, last_only=True, cfg=cfg, mode="eval",
                                     rng=generator, r_override=r).logits
            token = logits[0, -1].argmax().item()
            if token == tok.eos_token_id:
                break
            generated.append(token)
            x = torch.cat((x, x.new_tensor([[token]])), dim=1)
    return tok.decode(generated, skip_special_tokens=True).strip()


def _addition_accuracy(model, cfg, tok, test_set, r, s0_seed, max_new_tokens):
    per_length = {str(length): dict(n=0, correct=0, acc=0.0) for length in range(1, 7)}
    for i, example in enumerate(test_set):
        a, b = example["a"], example["b"]
        row = per_length[str(max(len(str(a)), len(str(b))))]
        if cfg is None:
            answer = _greedy_vanilla(model, tok, a, b, max_new_tokens)
        else:
            answer = _greedy_looped2(model, cfg, tok, a, b, r, s0_seed + i, max_new_tokens)
        row["n"] += 1
        row["correct"] += int(answer == str(example["answer"]))

    def accuracy(lengths):
        rows = [per_length[str(length)] for length in lengths]
        count = sum(row["n"] for row in rows)
        return sum(row["correct"] for row in rows) / count if count else 0.0

    for length, row in per_length.items():
        row["acc"] = accuracy((length,))
    # D4: the ID set consists solely of examples with three-digit operands.
    return dict(r=r, s0_seed=s0_seed, per_length=per_length, id_acc=per_length["3"]["acc"],
                in_distribution_acc_pooled_1_3=accuracy(range(1, 4)),
                ood_4_acc=per_length["4"]["acc"], ood_56_acc=accuracy((5, 6)),
                overall_acc=accuracy(range(1, 7)))


def evaluate_addition(args, checkpoint):
    run_config = json.loads(Path(args.config).read_text())
    model, cfg = _load_checkpoint(checkpoint, allow_unscaled_vanilla=True)
    tok = AutoTokenizer.from_pretrained(ADDITION_MODEL, revision=ADDITION_REVISION)
    test_set = addition.build_test_set(seed=args.seed, per_length=args.per_length)
    if cfg is None:
        # Vanilla greedy decoding uses the native model forward. The hook scales
        # embed_in output by looped2_emb_scale before emb_dropout, as in
        # looped2.plain_forward during training.
        emb_scale = getattr(model.config, "looped2_emb_scale", 1.0)
        with model.gpt_neox.embed_in.register_forward_hook(lambda m, x, e: e * emb_scale):
            per_r = [_addition_accuracy(model, cfg, tok, test_set, 1, args.s0_seed,
                                        args.max_new_tokens)]
        effective_config = dict(emb_scale=emb_scale)
    else:
        per_r = [_addition_accuracy(model, cfg, tok, test_set, r, args.s0_seed,
                                    args.max_new_tokens) for r in args.eval_r]
        effective_config = asdict(cfg)
    return dict(config=run_config, effective_config=effective_config,
                test_seed=args.seed, per_length_count=args.per_length,
                loop_source="checkpoint", per_r=per_r)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    lm = commands.add_parser("lm")
    add = commands.add_parser("addition")
    for command in (lm, add):
        command.add_argument("--checkpoint", required=True)
        command.add_argument("--eval-r", type=int, nargs="+", action="extend")
        command.add_argument("--s0-seed", type=int, default=0)
        command.add_argument("--output", required=True)
    lm.add_argument("--s0-seeds", type=int, nargs="+",
                    help="evaluate the checkpoint's primary depth if requested, otherwise the first depth")
    lm.add_argument("--split", default="validation", choices=["validation"])
    add.add_argument("--config", required=True)
    add.add_argument("--seed", type=int, default=1234)
    add.add_argument("--per-length", type=int, default=500)
    add.add_argument("--max-new-tokens", type=int, default=8)
    args = parser.parse_args()
    args.eval_r = args.eval_r or [4]
    if any(r < 1 for r in args.eval_r):
        parser.error("--eval-r values must be positive")
    if args.command == "addition" and (args.per_length < 1 or args.max_new_tokens < 1):
        parser.error("--per-length and --max-new-tokens must be positive")
    checkpoint = Path(args.checkpoint).resolve()
    evaluate = evaluate_lm if args.command == "lm" else evaluate_addition
    result = evaluate(args, checkpoint)
    result.update(checkpoint=str(checkpoint), git_hash=_git_hash())
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
