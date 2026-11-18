"""EXP-26: replay Pile block chunking on CPU and record each block's set.

Reuses pile_rows from src/evalsets26.py and the exact tokenization call
from chunk_stream; tracks, for every pending token, which document
(pile_set_name) it came from. A block belongs to the set of its first
token -- the only unambiguous attribution.

Tokenizer loading is identical to evaluate() in src/evalsets26.py:
AutoTokenizer.from_pretrained(MODEL, revision=REVISION) with MODEL /
REVISION imported from src.ponder. CPU only; no model is loaded.
"""
import argparse
from collections import Counter
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from transformers import AutoTokenizer

from src.evalsets26 import chunk_stream, pile_rows
from src.ponder import MODEL, REVISION

PARQUET = ROOT / "runs/EXP-26/data/pile-0.parquet"
BLOCK_TOKENS = 256


def replay(rows, tok, blocks):
    """Mirror chunk_stream exactly, plus per-token set origin.

    Returns (block_sets, stats, chunks) where block_sets[i] is the
    pile_set_name of block i's first token, stats mirrors chunk_stream's
    stats dict, and chunks are the 256-token id lists.
    """
    stats = dict(docs_consumed=0, tokens_consumed=0, leftover_tokens=0,
                 pile_set_counts={})
    counts = Counter()
    pending_ids = []
    pending_sets = []  # parallel to pending_ids: set of each token's doc
    block_sets = []
    chunks = []
    completed = 0
    for row in rows:
        ids = tok(row["text"], add_special_tokens=False)["input_ids"]
        if not ids:
            continue
        stats["docs_consumed"] += 1
        set_name = row["meta"]["pile_set_name"]
        counts[set_name] += 1
        stats["pile_set_counts"] = dict(counts)
        pending_ids.extend(ids)
        pending_sets.extend([set_name] * len(ids))
        while len(pending_ids) >= BLOCK_TOKENS:
            block_sets.append(pending_sets[0])
            chunks.append(pending_ids[:BLOCK_TOKENS])
            pending_ids = pending_ids[BLOCK_TOKENS:]
            pending_sets = pending_sets[BLOCK_TOKENS:]
            completed += 1
            stats["tokens_consumed"] = completed * BLOCK_TOKENS
            stats["leftover_tokens"] = len(pending_ids)
            if completed == blocks:
                return block_sets, stats, chunks
        stats["leftover_tokens"] = len(pending_ids)
    raise ValueError(f"shard 0 insufficient: {completed}/{blocks} full blocks")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--blocks", type=int, default=500)
    p.add_argument("--verify-first", type=int, default=30)
    p.add_argument("--out", default=str(ROOT / "runs/EXP-26/pile-block-sets.json"))
    a = p.parse_args()

    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)

    # 1) Original chunk_stream for the first N blocks (ground truth).
    check_stats = {}
    check_chunks = list(chunk_stream(pile_rows(PARQUET), tok, a.verify_first,
                                     BLOCK_TOKENS, check_stats))
    # 2) Annotated replay for the first N blocks: chunks and stats must match.
    sets_n, stats_n, chunks_n = replay(pile_rows(PARQUET), tok, a.verify_first)
    assert chunks_n == check_chunks, "annotated replay chunks != chunk_stream chunks"
    assert stats_n == check_stats, f"annotated replay stats != chunk_stream stats: {stats_n} vs {check_stats}"
    # 3) Full annotated replay.
    block_sets, stats, _ = replay(pile_rows(PARQUET), tok, a.blocks)
    assert block_sets[:a.verify_first] == sets_n, "full replay prefix != verified prefix"

    block_counts = dict(Counter(block_sets))
    result = dict(
        tokenizer=dict(model=MODEL, revision=REVISION),
        blocks=a.blocks,
        block_tokens=BLOCK_TOKENS,
        docs_consumed=stats["docs_consumed"],
        tokens_consumed=stats["tokens_consumed"],
        leftover_tokens=stats["leftover_tokens"],
        pile_set_counts=stats["pile_set_counts"],  # doc counts, as in evalsets26
        block_counts=block_counts,  # block counts per set (first-token rule)
        block_sets=[dict(index=i, pile_set_name=s)
                    for i, s in enumerate(block_sets)],
        verify_first_n=a.verify_first,
        verify_first_docs_consumed=check_stats["docs_consumed"],
        verify_first_pile_set_counts=check_stats["pile_set_counts"],
        verify_first_tokens_consumed=check_stats["tokens_consumed"],
        verify_first_leftover_tokens=check_stats["leftover_tokens"],
    )
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(out)
    print(json.dumps({k: v for k, v in result.items() if k != "block_sets"},
                     indent=2, ensure_ascii=False))
    print("wrote", out)


if __name__ == "__main__":
    main()
