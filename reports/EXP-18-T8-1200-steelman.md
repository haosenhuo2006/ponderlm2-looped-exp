# EXP-18 — T8 × 1200 steps, multi-seed (steelman test)

## 0. Why this experiment was necessary

T8 was the best ponder variant at 600 steps (NLL 4.9944, only 0.067 behind T0×600 at seed 42). The open question was whether the best thought model can tie vanilla given a longer budget. This experiment is the steelman test: it gives thought every advantage — the best variant (T8: aggregated intermediate supervision + split updates), 1200 data steps, and roughly 8× the training FLOPs per step of vanilla — and asks whether it can win. If it still loses, the thought mechanism has no value beyond its compute at this scale.

Decision rule (pre-registered in `~/workspace/phase5-design.md`, before any run): three-seed mean of T8×1200 worse than T0×1200 by more than 0.05 → thought loses; close the thought line. Mean difference ≤ 0.05 with no losing direction → mechanism is real; next phase digs into why.

## 1. Method

- Base configs: `configs/exp17_t8.json` for T8, `configs/exp12_t0.json` for T0. Only `steps`, `run_dir`, `seed` changed (T8: steps 600 → 1200; T0: steps already 1200).
- Unified settings unchanged: Pythia-70M (`EleutherAI/pythia-70m`, revision `a39f36b100fe8a5377810d56c3f4789b9c53ac42`), WikiText-2 (Salesforce/wikitext raw-v1 mirror, fixed revision), seq 256, microbatch 1, grad_accum 16, AdamW lr 1e-4, betas (0.9, 0.95), weight decay 0.01, 20-step warmup, cosine decay to 10% floor, grad clip 1.0, bf16 autocast + fp32 master weights, SDPA. Main metric: validation-set NLL (977 chunks, 249135 predicted tokens).
- T8×1200, seeds 42/43/44; T0×1200, seeds 43/44. T0×1200 seed 42 is reused from EXP-12 (4.7428, previously validated; not re-run).
- K sequences are drawn per seed from `random.Random(seed)` over {2,3} for 1200 steps. Data orders are 19200 indices per seed from `numpy.random.default_rng(seed)`; the seed-42 order's first 9600 indices match `runs/exp01-order.json`.
- Evaluation: validation NLL at the step-1200 checkpoint, plus a step-600 midpoint evaluation for T8 (same staging trick as EXP-16). The midpoint LR has not decayed, so it is not directly comparable to the EXP-17 600-step endpoint.
- Validation is done by `scripts/summarize_exp18.py` (1200 ordered steps, K sequences, data orders regenerated independently, source hashes, evaluation fields) and the result is written to `runs/exp18-summary.json`.

## 2. Results

### 2.1 Endpoints (validation NLL / PPL, step-1200 checkpoint)

| Model | seed 42 | seed 43 | seed 44 | Mean ± SD |
|---|---|---|---|---|
| T8×1200 | 4.8207 / 124.05 | 4.8810 / 131.76 | 4.7941 / 120.79 | **4.8319 ± 0.0445** / 125.54 ± 5.63 |
| T0×1200 | 4.7428 / 114.75 | 4.7322 / 113.55 | 4.6691 / 106.60 | **4.7147 ± 0.0399** / 111.63 ± 4.40 |

### 2.2 Paired differences (T8 − T0, NLL)

| seed 42 | seed 43 | seed 44 | Mean |
|---|---|---|---|
| +0.0779 | +0.1488 | +0.1250 | **+0.1172** |

Direction consistent: T8 is worse on all three seeds.

### 2.3 Step-600 midpoints (T8 only, LR not decayed)

| seed 42 | seed 43 | seed 44 |
|---|---|---|
| 5.0979 / 163.67 | 5.1093 / 165.55 | 5.1177 / 166.95 |

### 2.4 Training cost and K counts

| Model | Wall (s) per seed | Peak memory (MiB) |
|---|---|---|
| T8×1200 | 1493.8 / 1391.2 / 1393.7 | ~1822 |
| T0×1200 | 507.8 (EXP-12) / 454.9 / 453.1 | ~1393 |

K counts (T8, 1200 steps): seed 42: 587×2, 613×3 (3013 optimizer updates); seed 43: 596×2, 604×3 (3004 updates); seed 44: 598×2, 602×3 (3002 updates).

### 2.5 Context

- T8×1200 (mean 4.8319) beats T6×1200 seed 42 (4.8870) and T8×600 seed 42 (4.9944): the longer budget helped T8 by 0.16 over its own 600-step result.
- T0 also improved with budget: T0×600 multi-seed mean 4.900 ± 0.040 → T0×1200 mean 4.7147 ± 0.0399 (−0.185). The gap did not close; it is +0.117 at 1200 steps.
- Training FLOPs are analytical estimates, not profiler measurements: T8 costs ~8× vanilla per step (~2× the data steps here means ~8× the total training FLOPs of T0×1200); inference ~1.47× vanilla.

## 3. Conclusions

- **Decision (pre-registered rule): "thought loses; close thought line."** Mean gap +0.1172 exceeds the 0.05 threshold, and the direction is consistent across all three seeds. This is not single-seed noise.
- The best thought model, given more compute than vanilla (best variant + 1200 steps + ~8× training FLOPs per step), still loses to vanilla by a stable margin. At this scale, thought adds nothing vanilla cannot get more cheaply.
- The 1200-step budget helped both sides; it did not rescue thought. The resource story from EXP-10/EXP-13 holds: thought's deficit is not fixed by more optimization budget, because vanilla converts the same budget into more improvement.

## 4. Confounds and limitations

- T8 clips separately for each microbatch group; T1 clips once per data step (same caveat as EXP-17).
- Cosine LR is stretched over 1200 steps; the step-600 midpoints are not directly comparable to the EXP-17 600-step endpoint.
- WikiText-2 only; 70M tiny-scale; three seeds; no downstream benchmark in this experiment.
- Training FLOPs (~8×) are analytical estimates, not profiler measurements.
- This closes the thought line at tiny scale. It does not test larger models, other corpora, or downstream tasks; and it does not test the looped variant (see EXP-19).
