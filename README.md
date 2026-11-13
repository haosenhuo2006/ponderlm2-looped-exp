# ponderlm2-looped-exp

Tiny-scale reproduction of **PonderLM-2** (Zeng et al., arXiv:2509.23184v4) and
diagnostics for Looped Transformer comparisons — on a single RTX 4060 Laptop (8 GB).

This is **not** a full-scale reproduction of the paper (1.4B / 300B tokens /
73,047 GPU hours). The goal is to verify mechanisms at tiny scale:
does Jacobi iteration converge, does a latent thought help at matched
inference FLOPs, and how much of the gain comes from the current thought
vs. previous thoughts (D0–D2 diagnostics)?

## Plan

See [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md). In short:

- **EXP-01** — T0 vanilla LM baseline vs. T1 full PonderLM-2
  (same base model, same data order, same token budget).
- **EXP-02** — Diagnostics D0–D2 on the T1 checkpoint, inference-only:
  D0 full model, D1 mask previous thoughts, D2 mask previous tokens
  (keep the current token).

T2/T3/T4/T5 (trained masking variants, Looped comparison, intermediate
supervision) are **not** in this stage; they wait for Haosen's decision
after EXP-01/EXP-02.

## Repo layout

```
├── EXPERIMENT_PLAN.md
├── src/          # minimal implementation / diagnostic scripts
├── configs/      # per-experiment configs (seed, data, hyperparams)
├── scripts/      # train / eval / diagnose entry points
├── tests/        # mask unit tests, Jacobi convergence tests
├── runs/         # logs and small results (no large checkpoints in git)
└── reports/      # one detailed report per experiment
```

## Reproducibility

Every report records the method, commands, config, git hash, seed,
hardware, wall time, and limitations. Failed runs are reported as
failures — no invented numbers, no claiming an unrun experiment as done.
