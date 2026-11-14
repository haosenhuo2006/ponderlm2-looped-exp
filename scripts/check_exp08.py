"""Configuration identity and deterministic seeded K/data sampling."""
import hashlib
import json
import random
import tempfile
from pathlib import Path
import os
from src.data import order
BASE = {"T0":"exp01_t0", "T1":"exp01_t1", "T2":"exp03_t2", "T4":"exp05_t4"}
def ks(seed):
    rng=random.Random(seed)
    return [rng.choice([2,3]) for _ in range(600)]
def main():
    records={}
    for seed in (43,44):
        assert ks(seed)==ks(seed)
        for model,base in BASE.items():
            c=json.loads(Path(f"configs/exp08_seed{seed}_{model.lower()}.json").read_text())
            b=json.loads(Path(f"configs/{base}.json").read_text())
            assert c["seed"]==seed and c["run_dir"]==f"runs/EXP-08-seed{seed}-{model}"
            assert {k:v for k,v in c.items() if k not in ("seed","run_dir")}=={k:v for k,v in b.items() if k not in ("seed","run_dir")}
        with tempfile.TemporaryDirectory() as d:
            os.environ["EXP08_RUN_DIR"]=d
            first=order(9411,9600,seed)
            Path(d,"data-order.json").unlink()
            assert first==order(9411,9600,seed)
        records[str(seed)]={"K_counts":{str(k):ks(seed).count(k) for k in (2,3)}, "K_sha256":hashlib.sha256(bytes(ks(seed))).hexdigest(), "order_sha256":hashlib.sha256(json.dumps(first).encode()).hexdigest()}
    os.environ.pop("EXP08_RUN_DIR",None)
    assert ks(43)!=ks(44)
    Path("reports/exp08-check.json").write_text(json.dumps(records,indent=2)+"\n")
    print(json.dumps(records,indent=2))
if __name__=="__main__": main()
