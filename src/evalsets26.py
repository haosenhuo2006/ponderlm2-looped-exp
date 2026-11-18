"""EXP-26: 0-shot likelihood MC (HellaSwag/PIQA/SciQ) and Pile PPL.
SciQ support is excluded. Pile uses validation backup shard 0 because the
canonical The Eye URL has an expired certificate and returns 404 even with -k.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time
import zipfile

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import torch
from transformers import AutoTokenizer
from .ponder import load_model, forward, MODEL, REVISION
from . import looped

ROOT = Path(__file__).resolve().parents[1]
RUN = Path(os.environ.get("EVALSETS26_RUN", ROOT / "runs/EXP-26"))
CONFIG = Path(os.environ.get("EVALSETS26_CONFIG", ROOT / "configs/exp26.json"))
# EXP-26 uses no optional attention masking. An absent or null mask means no mask.
MASK_SEMANTICS = "EXP-26 uses no attention masking; checkpoint spec mask must be absent or None."
CANONICAL_URL_DEAD = (
    "https://the-eye.eu/public/AI/pile/val.jsonl.zst: curl exit 60; certificate "
    "issued 2025-12-21, expired 2026-03-21; -k range probe returned 404; HF "
    "discussion reports removal. Use mit-han-lab/pile-val-backup validation."
)


def validate_spec_mask(spec):
    if spec.get("mask") is not None:
        raise ValueError(f"{MASK_SEMANTICS} Got mask={spec['mask']!r}.")


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def protocol_hash():
    h = hashlib.sha256()
    for path in (CONFIG.resolve(), ROOT / "src/evalsets26.py"):
        h.update(path.read_bytes())
    return h.hexdigest()


def config():
    return json.loads(CONFIG.read_text())


def prompt(question, answer):
    return config()["prompt"].format(question=question, answer=answer)


def answer_span_nll(logits, ids, answer_start):
    """FP32 CE sum for answer targets, including the boundary token."""
    if not 1 <= answer_start < len(ids):
        raise ValueError("answer_start must identify an answer token with context")
    targets = torch.as_tensor(ids[answer_start:], device=logits.device, dtype=torch.long)
    return torch.nn.functional.cross_entropy(
        logits[0, answer_start-1:len(ids)-1].float(), targets, reduction="sum").item()


def score_options(answer_nlls, answer_bytes, totals=None, answer_tokens=None):
    """R1 primary; R2 UTF-8 byte normalization; R3/raw diagnostics only."""
    if not answer_nlls or len(answer_nlls) != len(answer_bytes) or any(b <= 0 for b in answer_bytes):
        raise ValueError("nonempty options and positive UTF-8 byte counts required")
    def pick(scores):
        return min(range(len(scores)), key=scores.__getitem__)
    result = dict(r1=pick(answer_nlls),
                  r2=pick([a/b for a, b in zip(answer_nlls, answer_bytes)]))
    if totals is not None and answer_tokens is not None:
        if len(totals) != len(answer_nlls) or len(answer_tokens) != len(totals) or any(n <= 0 for n in answer_tokens):
            raise ValueError("diagnostics require matching totals and positive token counts")
        result.update(r3=pick([v/n for v, n in zip(totals, answer_tokens)]), raw=pick(totals))
    return result


def answer_token_count(enc, prefix_len):
    return sum(end > prefix_len for _, end in enc["offset_mapping"])


def build_sciq_options(row):
    # Each option is scored independently: a fixed gold index introduces no
    # position bias. Support is retained for audit, never placed in the prompt.
    return dict(question=row["question"], options=[row[k] for k in
                ("correct_answer", "distractor1", "distractor2", "distractor3")],
                gold=0, support=row["support"])


def map_hellaswag(row):
    assert len(row["endings"]) == 4
    assert row["ctx"]
    gold = int(row["label"])
    assert gold in range(4)
    return dict(question=row["ctx"], options=row["endings"], gold=gold)


def map_piqa(row):
    gold = int(row["label"])
    assert gold in (0, 1)
    return dict(question=row["goal"], options=[row["sol1"], row["sol2"]], gold=gold)


def result_path(name, task, limit=None):
    return RUN / ("smoke" if limit is not None else "results") / (task + "-" + name + ".json")


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for data in iter(lambda: f.read(1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


def pile_rows(path):
    import pyarrow.parquet as pq
    for batch in pq.ParquetFile(path).iter_batches(batch_size=1000, columns=["text", "meta"]):
        yield from batch.to_pylist()


def chunk_stream(rows, tok, blocks=500, block_tokens=256, stats=None):
    """Stop within the final document; empty tokenizations do not count as docs."""
    if stats is None:
        stats = {}
    stats.update(docs_consumed=0, tokens_consumed=0, leftover_tokens=0, pile_set_counts={})
    pending = []
    completed = 0
    counts = Counter()
    for row in rows:
        ids = tok(row["text"], add_special_tokens=False)["input_ids"]
        if not ids:
            continue
        stats["docs_consumed"] += 1
        counts[row["meta"]["pile_set_name"]] += 1
        stats["pile_set_counts"] = dict(counts)
        pending.extend(ids)
        while len(pending) >= block_tokens:
            chunk, pending = pending[:block_tokens], pending[block_tokens:]
            completed += 1
            stats["tokens_consumed"] = completed * block_tokens
            stats["leftover_tokens"] = len(pending)
            yield chunk
            if completed == blocks:
                return
        stats["leftover_tokens"] = len(pending)
    raise ValueError(f"shard 0 insufficient: {completed}/{blocks} full blocks")


def pile_metrics(records, block_tokens=256):
    mean_nll = sum(r["chunk_nll"] for r in records) / len(records)
    predicted_tokens = len(records) * (block_tokens - 1)
    return dict(mean_nll=mean_nll, ppl=math.exp(mean_nll),
                predicted_tokens=predicted_tokens,
                total_nll_sum=sum(r["chunk_nll"] * (block_tokens - 1) for r in records))


def prepare():
    import pyarrow.parquet as pq
    RUN.mkdir(parents=True, exist_ok=True)
    provenance = {"tasks": {}}
    for task, spec in config()["tasks"].items():
        timeout = 1800 if task == "pile" else 180
        try:
            filename = {"piqa": "piqa.zip", "pile": "pile-0.parquet"}.get(task, task + ".parquet")
            raw = RUN / "data" / filename
            raw.parent.mkdir(parents=True, exist_ok=True)
            if not raw.exists():
                tmp = raw.with_suffix(".download")
                subprocess.run(["curl", "-fL", "--retry", "2", "--max-time", str(timeout),
                                spec["url"], "-o", str(tmp)], check=True)
                # Validate before publishing the downloaded source.
                if sha256(tmp) != spec["sha256"]:
                    raise ValueError("source SHA256 mismatch")
                tmp.replace(raw)
            sha = sha256(raw)
            if sha != spec["sha256"]:
                raise ValueError("source SHA256 mismatch")
            extra = {}
            if task == "pile":
                pf = pq.ParquetFile(raw)
                assert {"text", "meta"} <= set(pf.schema_arrow.names)
                counts = Counter()
                first = 0
                for row in pile_rows(raw):
                    counts[row["meta"]["pile_set_name"]] += 1
                    first += 1
                    if first == 1000:
                        break
                assert first == 1000 and len(counts) >= 2
                # Row count is not the sufficiency criterion: 500 blocks need 128000
                # tokens, and chunk_stream raises if shard 0 cannot supply them.
                extra = dict(row_count=pf.metadata.num_rows,
                             pile_set_counts_first_1000=dict(counts),
                             canonical_url_dead=CANONICAL_URL_DEAD)
                entry = dict(spec, sha256=sha, status="ready", count=spec["blocks"],
                             selected_sha256=None, **extra)
            else:
                if task == "piqa":
                    with zipfile.ZipFile(raw) as z:
                        rows = [json.loads(line) for line in z.read("physicaliqa-train-dev/dev.jsonl").decode().splitlines()]
                        labels = z.read("physicaliqa-train-dev/dev-labels.lst").decode().splitlines()
                    assert len(rows) == len(labels) == 1838
                    rows = [dict(goal=r["goal"], sol1=r["sol1"], sol2=r["sol2"], label=int(label))
                            for r, label in zip(rows, labels)]
                    for row in rows:
                        map_piqa(row)
                else:
                    rows = pq.read_table(raw).to_pylist()
                    if task == "hellaswag":
                        assert len(rows) == 10042
                        for row in rows:
                            map_hellaswag(row)
                        rows = rows[:2000]
                        assert {int(r["label"]) for r in rows} == set(range(4))
                    else:
                        assert len(rows) == 1000
                        rows = [build_sciq_options(r) for r in rows]
                        extra["support_excluded"] = True
                out = RUN / "data" / (task + ".json")
                save(out, rows)
                entry = dict(spec, sha256=sha, status="ready", count=len(rows),
                             selected_sha256=sha256(out), **extra)
            provenance["tasks"][task] = entry
        except Exception as e:
            provenance["tasks"][task] = dict(spec, status="failed", error=repr(e))
            if task == "pile":
                provenance["tasks"][task]["canonical_url_dead"] = CANONICAL_URL_DEAD
    save(RUN / "data-provenance.json", provenance)
    print(json.dumps(provenance, indent=2), flush=True)


def model_forward(model, x, spec):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        if spec["mode"] == "looped":
            return looped.forward(model, x, labels=x)
        return forward(model, x, 3)


@torch.no_grad()
def evaluate(name, task, limit=None):
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    cfg = config()
    spec = cfg["checkpoints"][name]
    validate_spec_mask(spec)
    provenance = json.loads((RUN / "data-provenance.json").read_text())["tasks"][task]
    if provenance["status"] != "ready":
        print("SKIP unavailable", task, flush=True)
        return
    if task == "pile":
        count = limit if limit is not None else cfg["tasks"][task]["blocks"]
        rows = None
    else:
        rows = json.loads((RUN / "data" / (task + ".json")).read_text())
        if limit is not None:
            rows = rows[:limit]
        count = len(rows)
    path = result_path(name, task, limit)
    if path.exists() and limit is None:
        previous = json.loads(path.read_text())
        if previous.get("protocol_hash") == protocol_hash() and previous["count"] == count:
            print("REUSE verified full result", path, flush=True)
            return
        raise ValueError("existing EXP-26 result protocol mismatch; refusing stale reuse")
    torch.manual_seed(cfg["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    model = load_model("vanilla" if spec["mode"] == "looped" else spec["mode"], spec["path"]).eval()
    if spec["mode"] == "looped":
        looped.configure(model, model.config.loop_block_size, model.config.loop_repetitions,
                         start=getattr(model.config, "loop_block_start", None))
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    begin = time.perf_counter()
    records = []

    def run(ids):
        # Never truncate or select samples by length.
        effective = len(ids) * (2 if spec["mode"] == "ponder" else 1)
        if effective > model.config.max_position_embeddings:
            raise ValueError(f"context overflow: {effective}")
        return model_forward(model, torch.tensor([ids], device="cuda"), spec)

    stats = {}
    if task == "pile":
        iterator = chunk_stream(pile_rows(RUN / "data/pile-0.parquet"), tok,
                                count, cfg["tasks"][task]["block_tokens"], stats)
    else:
        iterator = iter(rows)
    for index, row in enumerate(iterator):
        if task == "pile":
            out = run(row)
            nll = out.loss.float().item()
            if not torch.isfinite(out.loss):
                raise ValueError("nonfinite loss")
            record = dict(index=index, chunk_nll=nll)
        else:
            row = map_hellaswag(row) if task == "hellaswag" else map_piqa(row) if task == "piqa" else row
            q = row["question"]
            totals, counts, answer_nlls, byte_counts = [], [], [], []
            for answer in row["options"]:
                prefix = q + " Answer: "
                enc = tok(prompt(q, answer), add_special_tokens=False, return_offsets_mapping=True)
                n = answer_token_count(enc, len(prefix))
                assert n > 0
                k0 = next(i for i, (_, end) in enumerate(enc["offset_mapping"]) if end > len(prefix))
                assert n == len(enc["input_ids"]) - k0
                out = run(enc["input_ids"])
                total = out.loss.float().item() * (len(enc["input_ids"]) - 1)
                if not torch.isfinite(out.loss):
                    raise ValueError("nonfinite loss")
                totals.append(total)
                counts.append(n)
                answer_nlls.append(answer_span_nll(out.logits, enc["input_ids"], k0))
                byte_counts.append(len(answer.encode("utf-8")))
            scores = score_options(answer_nlls, byte_counts, totals, counts)
            pred, raw_pred = scores["r1"], scores["raw"]
            gold = row["gold"]
            record = dict(index=index, correct=pred == gold, prediction=pred, gold=gold,
                          raw_correct=raw_pred == gold, raw_prediction=raw_pred, total_nll=totals,
                          answer_tokens=counts, per_answer_token_nll=[v/n for v, n in zip(totals, counts)])
            record.update(answer_nll=answer_nlls, answer_bytes=byte_counts,
                          per_answer_byte_nll=[a/b for a, b in zip(answer_nlls, byte_counts)])
            for key in ("r1", "r2", "r3"):
                record[key + "_prediction"] = scores[key]
                record[key + "_correct"] = scores[key] == gold
        records.append(record)
        if (index + 1) % (50 if task == "pile" else 10) == 0 or index == 0:
            progress = dict(checkpoint=name, task=task, completed=index+1, elapsed=time.perf_counter()-begin)
            if task == "pile":
                progress["mean_nll"] = sum(r["chunk_nll"] for r in records)/len(records)
            else:
                progress["correct"] = sum(r["correct"] for r in records)
            print(json.dumps(progress), flush=True)
    torch.cuda.synchronize()
    correct = None if task == "pile" else sum(r["correct"] for r in records)
    accuracy = None if task == "pile" else correct/len(records)
    result = dict(checkpoint=name, task=task, config=spec, protocol_hash=protocol_hash(), count=len(records),
                  mask=None, mask_semantics=MASK_SEMANTICS,
                  correct=correct, accuracy=accuracy, primary_count=len(records), primary_correct=correct,
                  primary_accuracy=accuracy, seconds=time.perf_counter()-begin,
                  peak_memory_bytes=torch.cuda.max_memory_allocated(),
                  git_hash=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                  hardware=torch.cuda.get_device_name(), torch_version=torch.__version__, records=records)
    if task == "pile":
        result.update(blocks=len(records), block_tokens=cfg["tasks"][task]["block_tokens"], **stats, **pile_metrics(records))
    else:
        result["raw_accuracy"] = sum(r["raw_correct"] for r in records)/len(records)
        result["accuracy_r1"] = result["accuracy"]
        for key in ("r2", "r3"):
            result["accuracy_" + key] = sum(r[key + "_correct"] for r in records)/len(records)
        for key in ("r1", "r3"):
            result["longest_pick_rate_" + key] = sum(
                r["answer_tokens"][r[key + "_prediction"]] == max(r["answer_tokens"])
                for r in records)/len(records)
        result["scoring"] = {
            "primary": "R1 answer-span CE sum (argmin A)",
            "secondary": "R2 byte-normalized (argmin A/utf8-bytes)",
            "r3_deprecated": True, "raw_diagnostic": True,
            "answer_span": "first token with offset end > len(prefix), inclusive"}
    save(path, result)
    print(json.dumps({k: v for k, v in result.items() if k != "records"}, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prepare", action="store_true")
    p.add_argument("--checkpoint", choices=list(config()["checkpoints"]))
    p.add_argument("--task", choices=["hellaswag", "piqa", "sciq", "pile"])
    p.add_argument("--limit", type=int)
    a = p.parse_args()
    if a.prepare:
        prepare()
    else:
        if a.checkpoint is None or a.task is None:
            p.error("--checkpoint and --task are required without --prepare")
        evaluate(a.checkpoint, a.task, a.limit)


if __name__ == "__main__":
    main()
