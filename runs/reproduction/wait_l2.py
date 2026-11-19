import json
import sys
import time
from pathlib import Path

ROOT = Path("/home/lenovo_jason/ponderlm2-looped-exp")
DRIVER_LOG = ROOT / "runs/reproduction/run_l2_resume.log"

def main():
    group = sys.argv[1]
    deadline_s = int(sys.argv[2]) if len(sys.argv) > 2 else 3600
    marker = "=== " + group + " DONE ==="
    waited = 0
    while waited < deadline_s:
        try:
            text = DRIVER_LOG.read_text()
        except OSError:
            text = ""
        if "FAIL" in text:
            print("DRIVER_FAIL")
            print(text[-2000:])
            return 1
        if marker in text:
            ev = ROOT / "runs/reproduction" / group / "evaluation.json"
            try:
                res = json.loads(ev.read_text())
                row = res["per_r"][0]
                print(group + " NLL=" + str(row.get("nll", row)))
            except Exception as e:
                print(group + " DONE but evaluation.json unreadable: " + str(e))
            return 0
        time.sleep(30)
        waited += 30
    print("TIMEOUT waiting for " + group)
    try:
        print(DRIVER_LOG.read_text()[-2000:])
    except OSError:
        pass
    return 1

if __name__ == "__main__":
    raise SystemExit(main())
