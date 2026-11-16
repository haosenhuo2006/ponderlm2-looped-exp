"""EXP-09 fixed downstream protocol; no training or prior-run writes."""
import argparse
import hashlib
import json
import os
import re
import subprocess
import time
import urllib.request
from pathlib import Path
import torch
from transformers import AutoTokenizer
from .ponder import load_model, forward, MODEL, REVISION
from . import looped, masked

ROOT = Path(__file__).resolve().parents[1]
RUN = Path(os.environ.get("DOWNSTREAM_RUN", ROOT / "runs/EXP-09"))
CONFIG = Path(os.environ.get("DOWNSTREAM_CONFIG", ROOT / "configs/exp09_downstream.json"))

def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)

def protocol_hash():
    h = hashlib.sha256()
    for path in (CONFIG.resolve(), ROOT / "src/downstream.py"):
        h.update(path.read_bytes())
    return h.hexdigest()

def config():
    return json.loads(CONFIG.read_text())

def prompt(question, answer):
    return config()["prompt"].format(question=question, answer=answer)

def select_answer(totals, answer_tokens):
    scores = [total / n for total, n in zip(totals, answer_tokens)]
    return max(range(len(scores)), key=lambda i: -scores[i])

def last_word_tokens(tok, text):
    text = text.rstrip()
    match = re.search(r"\S+$", text)
    if not match:
        raise ValueError("empty LAMBADA text")
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    start = next(i for i, (_, end) in enumerate(enc["offset_mapping"]) if end > match.start())
    if start < 1:
        raise ValueError("no context before final word")
    return enc["input_ids"], start, text[match.start():]

def word_correct(logits, ids, start):
    # logits at j-1 predicts token j, including every subtoken under teacher forcing.
    gold = torch.as_tensor(ids[start:], device=logits.device)
    return bool(torch.equal(logits[0, start-1:len(ids)-1].argmax(-1), gold))

def prepare():
    import pyarrow.parquet as pq
    RUN.mkdir(parents=True, exist_ok=True)
    cfg = config()
    provenance = {"official_arc_probe": {"url": "https://ai2-public-datasets.s3.amazonaws.com/arc/ARC-V1-Feb2018-2.zip", "failure": "HTTP 404 during feasibility probe; authorized HF fallback"}, "tasks": {}}
    for task, spec in cfg["tasks"].items():
        try:
            raw = RUN / "data" / (task + ".parquet")
            raw.parent.mkdir(exist_ok=True)
            if not raw.exists():
                tmp = raw.with_suffix(".download")
                subprocess.run(["curl", "-fL", "--retry", "2", "--max-time", "180", spec["url"], "-o", str(tmp)], check=True)
                tmp.replace(raw)
            sha = hashlib.sha256(raw.read_bytes()).hexdigest()
            if spec.get("sha256") and sha != spec["sha256"]:
                raise ValueError("source SHA256 mismatch")
            rows = pq.read_table(raw).to_pylist()
            if task == "arc":
                assert len(rows) == 570, len(rows)
            else:
                assert len(rows) >= 500
                rows = rows[:500]
            out = RUN / "data" / (task + ".json")
            save(out, rows)
            provenance["tasks"][task] = dict(spec, sha256=sha, status="ready", count=len(rows),
                selected_sha256=hashlib.sha256(out.read_bytes()).hexdigest(), selection="all validation in source order" if task == "arc" else "first 500 standard test rows in source order")
        except Exception as e:
            provenance["tasks"][task] = dict(spec, status="failed", error=repr(e))
    save(RUN / "data-provenance.json", provenance)
    print(json.dumps(provenance, indent=2), flush=True)

def model_forward(model, x, spec):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        if spec["mode"] == "looped":
            return looped.forward(model, x, labels=x)
        if spec.get("mask"):
            return masked.forward(model, x, 3, spec["mask"], labels=x)
        return forward(model, x, 3)

@torch.no_grad()
def evaluate(name, task, limit=None):
    cfg = config()
    spec = cfg["checkpoints"][name]
    provenance = json.loads((RUN / "data-provenance.json").read_text())["tasks"][task]
    if provenance["status"] != "ready":
        print("SKIP unavailable", task, flush=True)
        return
    rows = json.loads((RUN / "data" / (task + ".json")).read_text())
    if limit:
        rows = rows[:limit]
    result_path = RUN / ("smoke" if limit else "results") / (name+"-"+task+".json")
    if result_path.exists() and not limit:
        previous = json.loads(result_path.read_text())
        if previous.get("protocol_hash") == protocol_hash() and previous["count"] == len(rows):
            print("REUSE verified full result", result_path, flush=True)
            return
        raise ValueError("existing EXP-09 result protocol mismatch; refusing stale reuse")
    torch.manual_seed(cfg["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    model = load_model("vanilla" if spec["mode"] == "looped" else spec["mode"], spec["path"]).eval()
    if spec["mode"] == "looped":
        looped.configure(model, model.config.loop_block_size, model.config.loop_repetitions, start=getattr(model.config, "loop_block_start", None))
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
    for index, row in enumerate(rows):
        if task == "arc":
            q = row["question"]
            totals, counts = [], []
            for answer in row["choices"]["text"]:
                prefix = q + " Answer: "
                enc = tok(prompt(q, answer), add_special_tokens=False, return_offsets_mapping=True)
                n = sum(end > len(prefix) for _, end in enc["offset_mapping"])
                assert n > 0
                out = run(enc["input_ids"])
                total = out.loss.float().item() * (len(enc["input_ids"]) - 1)
                if not torch.isfinite(out.loss):
                    raise ValueError("nonfinite loss")
                totals.append(total)
                counts.append(n)
            pred = select_answer(totals, counts)
            raw_pred = min(range(len(totals)), key=totals.__getitem__)
            gold = list(row["choices"]["label"]).index(row["answerKey"])
            record = dict(index=index, id=row["id"], correct=pred == gold, prediction=pred, gold=gold,
                raw_correct=raw_pred == gold, raw_prediction=raw_pred, total_nll=totals,
                answer_tokens=counts, per_answer_token_nll=[v/n for v,n in zip(totals,counts)])
        else:
            ids, start, word = last_word_tokens(tok, row["text"])
            out = run(ids)
            record = dict(index=index, correct=word_correct(out.logits, ids, start),
                word=word, answer_tokens=len(ids)-start, target_start=start,
                gold=ids[start:], prediction=out.logits[0,start-1:len(ids)-1].argmax(-1).tolist())
        records.append(record)
        if (index + 1) % 10 == 0 or index == 0:
            print(json.dumps(dict(checkpoint=name, task=task, completed=index+1,
                correct=sum(r["correct"] for r in records), elapsed=time.perf_counter()-begin)), flush=True)
    torch.cuda.synchronize()
    singles = [r for r in records if r.get("answer_tokens") == 1] if task == "lambada" else records
    result = dict(checkpoint=name, task=task, config=spec, protocol_hash=protocol_hash(), count=len(records),
        correct=sum(r["correct"] for r in records), accuracy=sum(r["correct"] for r in records)/len(records),
        primary_count=len(singles), primary_correct=sum(r["correct"] for r in singles),
        primary_accuracy=sum(r["correct"] for r in singles)/len(singles) if singles else None,
        seconds=time.perf_counter()-begin, peak_memory_bytes=torch.cuda.max_memory_allocated(),
        git_hash=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        hardware=torch.cuda.get_device_name(), torch_version=torch.__version__, records=records)
    if task == "arc":
        result["raw_accuracy"] = sum(r["raw_correct"] for r in records)/len(records)
    else:
        result["multi_token_count"] = len(records)-len(singles)
        result["multi_token_correct"] = sum(r["correct"] for r in records if r["answer_tokens"] > 1)
    save(RUN / ("smoke" if limit else "results") / (name+"-"+task+".json"), result)
    print(json.dumps({k:v for k,v in result.items() if k != "records"},indent=2),flush=True)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prepare", action="store_true")
    p.add_argument("--checkpoint", choices=list(config()["checkpoints"]))
    p.add_argument("--task", choices=["arc","lambada"])
    p.add_argument("--limit", type=int)
    a = p.parse_args()
    if a.prepare:
        prepare()
    else:
        evaluate(a.checkpoint,a.task,a.limit)

if __name__ == "__main__":
    main()
