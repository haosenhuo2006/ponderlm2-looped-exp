"""D4: independent sampler simulation (CPU, NumPy).

Huginn log-normal Poisson sampler, nominal t=2, s=2, sigma=0.5, cap 12:
  log lambda ~ N(ln(t+s) - sigma^2/2, sigma^2),  E[lambda] = t+s = 4
  r = 1 + Poisson(lambda);  r = min(r, 12);  k = min(r, 2)
Draws: default_rng(42), N=2,000,000, batched (log-normal first, then
Poisson, per the design spec). Reports E[r] (uncapped and capped), E[k],
P(r=1), P(r>=8), P(r>12, pre-cap), and the full capped distribution.
Independent re-run of the C3 check in the Codex verification report.
"""
import json
from pathlib import Path

import numpy as np

OUT = Path("runs/reproduction/d4_sampler.json")
N = 2_000_000
BATCH = 200_000
T, S, SIGMA, CAP = 2, 2, 0.5, 12


def main():
    rng = np.random.default_rng(42)
    mu = np.log(T + S) - SIGMA ** 2 / 2
    sum_r = 0.0
    sum_r_cap = 0.0
    sum_k = 0.0
    c1 = c8 = c12 = 0
    counts = np.zeros(CAP + 1, dtype=np.int64)  # index 1..12 capped
    done = 0
    while done < N:
        n = min(BATCH, N - done)
        lam = rng.lognormal(mean=mu, sigma=SIGMA, size=n)
        r = 1 + rng.poisson(lam)
        c1 += int((r == 1).sum())
        c8 += int((r >= 8).sum())
        c12 += int((r > 12).sum())
        sum_r += float(r.sum())
        r_cap = np.minimum(r, CAP)
        sum_r_cap += float(r_cap.sum())
        sum_k += float(np.minimum(r_cap, S).sum())
        for v in range(1, CAP + 1):
            counts[v] += int((r_cap == v).sum())
        done += n
    out = dict(
        method="log-normal Poisson: log lam ~ N(ln(4)-0.125, 0.25); "
               "r = 1 + Poisson(lam); r = min(r, 12); k = min(r, 2); "
               "default_rng(42), batched log-normal then Poisson",
        t=T, s=S, sigma=SIGMA, cap=CAP, n_draws=N,
        E_r_uncapped=sum_r / N,
        E_r_capped=sum_r_cap / N,
        E_k=sum_k / N,
        P_r_eq_1=c1 / N,
        P_r_ge_8=c8 / N,
        P_r_gt_12_precap=c12 / N,
        P_r_le_12=1 - c12 / N,
        capped_counts={str(v): int(counts[v]) for v in range(1, CAP + 1)},
        capped_probs={str(v): counts[v] / N for v in range(1, CAP + 1)},
        theory="E[r] uncapped = 1 + E[lam] = 5 exactly; "
               "cap 12 truncates the tail (P(r>12) ~ 2%)")
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
