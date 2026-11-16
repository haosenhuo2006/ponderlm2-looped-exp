"""Evaluate a saved vanilla or looped addition model by operand length."""
import argparse
import importlib.util
import json
import subprocess
from pathlib import Path

from transformers import AutoTokenizer

from . import addition, looped
from .ponder import ROOT
from .train_addition import MODEL, REVISION, VENDORED


def evaluate(model, tok, test_set, max_new_tokens=8):
    per_length = {str(length): dict(n=0, correct=0, acc=0.0)
                  for length in range(1, 7)}
    for example in test_set:
        length = str(max(len(str(example["a"])), len(str(example["b"]))))
        row = per_length[length]
        answer = addition.greedy_answer(model, tok, example["a"], example["b"],
                                        max_new_tokens=max_new_tokens)
        row["n"] += 1
        row["correct"] += int(answer == str(example["answer"]))
    for row in per_length.values():
        row["acc"] = row["correct"] / row["n"] if row["n"] else 0.0

    def accuracy(lengths):
        rows = [per_length[str(length)] for length in lengths]
        n = sum(row["n"] for row in rows)
        return sum(row["correct"] for row in rows) / n if n else 0.0

    return dict(per_length=per_length, in_distribution_acc=accuracy(range(1, 4)),
                ood_acc=accuracy(range(4, 7)), overall_acc=accuracy(range(1, 7)),
                n_total=sum(row["n"] for row in per_length.values()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--per-length", type=int, default=500)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    args = parser.parse_args()
    if args.per_length < 1 or args.max_new_tokens < 1:
        parser.error("--per-length and --max-new-tokens must be positive")
    config = json.loads(Path(args.config).read_text())
    checkpoint = Path(args.checkpoint).resolve()
    spec = importlib.util.spec_from_file_location("addition_gpt_neox", VENDORED)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.GPTNeoXForCausalLM.from_pretrained(checkpoint)
    model.config.use_cache = False
    looped.configure(model, **(config["loop"] or dict(block_size=2, repetitions=1, start=2)))
    model = model.cuda().eval()
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    results = evaluate(model, tok, addition.build_test_set(seed=args.seed,
                                                         per_length=args.per_length),
                       max_new_tokens=args.max_new_tokens)
    try:
        git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                           text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        git_hash = None
    # Preserve the per_length metrics; record the CLI count under a distinct key.
    results.update(checkpoint=str(checkpoint), config=config, test_seed=args.seed,
                   per_length_count=args.per_length, model=MODEL, revision=REVISION,
                   git_hash=git_hash)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + "\n")
    for length, row in results["per_length"].items():
        print(f"{length} digits: {row['correct']}/{row['n']} acc={row['acc']:.6f}", flush=True)


if __name__ == "__main__":
    main()
