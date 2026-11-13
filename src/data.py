"""WikiText-2 raw chunks; immutable split files and shared seeded order."""
import hashlib
import json
import subprocess
import zipfile
import numpy as np
import torch
from transformers import AutoTokenizer
from .ponder import ROOT, MODEL, REVISION
URL = "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-raw-v1.zip"
DATA_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"

def chunks(split="train", seq=256):
    data = ROOT / "data"
    data.mkdir(exist_ok=True)
    archive = data / "wikitext-2-raw-v1.zip"
    cache = data / (split + "-tokens.npy")
    parquet = data / (split + ".parquet")
    if not archive.exists() and not parquet.exists():
        subprocess.run(["curl","-fL","--max-time","120",URL,"-o",str(archive)],check=True)
    if zipfile.is_zipfile(archive):
        source = archive
        source_url = URL
        name = "valid" if split == "validation" else split
        with zipfile.ZipFile(archive) as z:
            text = z.read("wikitext-2/wiki."+name+".raw").decode("utf-8")
    else:
        source_url = f"https://huggingface.co/datasets/Salesforce/wikitext/resolve/{DATA_REVISION}/wikitext-2-raw-v1/{split}-00000-of-00001.parquet"
        if not parquet.exists():
            subprocess.run(["curl","-fL","--max-time","120","--retry","2",source_url,"-o",str(parquet)],check=True)
        import pyarrow.parquet as pq
        source = parquet
        # Rows already retain their terminating newlines; empty rows represent blank lines.
        rows = pq.read_table(parquet)["text"].to_pylist()
        text = "".join(row if row else "\n" for row in rows)
    if not cache.exists():
        tok = AutoTokenizer.from_pretrained(MODEL,revision=REVISION)
        ids = tok(text,add_special_tokens=False)["input_ids"]
        np.save(cache,np.asarray(ids,dtype=np.int64))
    ids = np.load(cache)
    (ROOT/"runs").mkdir(exist_ok=True)
    metadata = dict(split=split,tokens=len(ids),seq=seq,chunks=len(ids)//seq,
                    dropped_tail_tokens=len(ids)%seq,source_url=source_url,
                    source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    text_sha256=hashlib.sha256(text.encode()).hexdigest(),
                    tokens_sha256=hashlib.sha256(ids.tobytes()).hexdigest(),
                    tokenizer=MODEL,revision=REVISION,special_tokens=False,
                    row_join="empty rows -> newline; nonempty rows verbatim")
    (ROOT/"runs"/("data-"+split+".json")).write_text(json.dumps(metadata,indent=2))
    return torch.from_numpy(ids[:len(ids)//seq*seq].copy()).reshape(-1,seq)

def order(n,count,seed=42):
    path = ROOT/"runs"/"exp01-order.json"
    path.parent.mkdir(exist_ok=True)
    if path.exists():
        result = json.loads(path.read_text())
        assert result["chunks"] == n and len(result["indices"]) == count and result["seed"] == seed
        return result["indices"]
    rng = np.random.default_rng(seed)
    indices = []
    while len(indices) < count:
        indices.extend(rng.permutation(n).tolist())
    indices = indices[:count]
    path.write_text(json.dumps(dict(seed=seed,chunks=n,indices=indices)))
    return indices
