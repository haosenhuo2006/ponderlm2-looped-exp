# EXP-22 — 160M middle-layer looped variant (scale check)

## 0. Why this experiment was necessary

The main question ("does repeated computation beat vanilla at matched compute?") was answered NO at 70M: thought lost (EXP-18), the middle-layer loop tied at 600 steps and lost at 1200 (EXP-19/EXP-20), and the downstream check found no comeback (EXP-21). But the biggest hole in that conclusion was scale: the positive looped results in the literature are all at 730M–3.5B (Huginn, Done Right, SMELT), while we had only tested 70M. 160M is the largest base that fits on this 8GB card without changing the memory profile (EXP-14). The question here is narrow and pre-registered: does the 600-step tie survive — or grow into a win — when the model is bigger?

Decision rules (pre-registered before launch, control = 160M T0×600 multi-seed mean): (a) looped mean better than T0 by more than 0.05 with a consistent direction → "scale revives Looped", conclusion rewritten; (b) mean difference within ±0.05 → "tie persists at scale"; (c) looped mean worse than T0 by more than 0.05 → "NO hardens across scales".

## 1. Method

### 1.1 Code change (the only `src/` change in phase 8)

- `src/evaluate_looped.py`: one line. `inference_flops_per_token_estimate` now calls `looped.flops(c.loop_block_size, c.loop_repetitions, d=c.hidden_size, ff=c.intermediate_size, layers=c.num_hidden_layers)` instead of relying on the 70M defaults (d=512, ff=2048, layers=6). Without this fix, a 160M looped evaluation would have silently reported the 70M FLOPs number. Backward-compatible: for the 70M model both calls return 130154496. `src/looped.py` and `src/train.py` are untouched. Full suite: 43/43 passed.

### 1.2 Variant and training

- Schedule: Huginn-style sandwich scaled to 12 layers by the 25% / 50% / 25% split — prelude [0,1,2] once, core [3,4,5,6,7,8] looped, coda [9,10,11] once. No input injection, no hyperconnections, no change to supervision — only the scale and R change relative to EXP-19.
- R was chosen by the FLOPs table before training (EXP-01 convention: 2 FLOPs/MAC, cached context mean 128 real tokens). 160M T0 inference = 251,854,848 FLOPs/token. R=2 → 18 layer calls, 339,148,800 FLOPs/token, ratio 1.3466 (5.8% error vs the 1.43 target — the looped-vs-vanilla inference ratio at 70M); R=3 → 24 calls, ratio 1.6932 (18.4% error). R=2 is closer, so R=2. Training FLOPs ≈ 1.5× vanilla (18 vs 12 calls).
- 600 steps × seeds 42/43/44, plus 160M T0 seeds 43/44 (EXP-14 had only seed 42). Unified settings unchanged: WikiText-2, seq 256, microbatch 1, grad_accum 16, AdamW lr 1e-4, betas (0.9, 0.95), weight decay 0.01, 20-step warmup, cosine decay to 10% floor, grad clip 1.0, bf16 autocast + fp32 master weights, SDPA. Main metric: validation-set NLL (977 chunks, 249135 predicted tokens). Looped runs evaluated with `src/evaluate_looped.py`, T0 runs with `src/evaluate.py`.
- Pilot gate (100 steps, seed 42) before the full runs: passed with decision "continue" — 100 ordered steps, finite loss/grad_norm, no divergence (last-20 loss mean 4.99 vs first-20 5.25), grad-norm median ratio vs 160M T0's first 100 steps 0.7515 (gate range [0.1, 10]), data-order prefix matches.
- Validation by `scripts/summarize_exp22.py` (600 ordered steps, K=0 rows, data orders regenerated independently per seed, source hashes, evaluation fields including `block_size == 6`, `repetitions == 2`, `loop_block_start == 3`, `layers == [3, 4, 5, 6, 7, 8]`, inference FLOPs 339148800). T0 seed 42 is read from `runs/EXP-14-T0` (NLL 4.7960, re-verified); seeds 43/44 are the new runs.

## 2. Results

### 2.1 Endpoints (validation NLL / PPL, step-600 checkpoint)

| Model | seed 42 | seed 43 | seed 44 | Mean ± SD |
|---|---|---|---|---|
| Loop-middle 160M (new) | 4.7946 / 120.85 | 4.7739 / 118.38 | 4.7662 / 117.47 | **4.7782 ± 0.0147** / 118.90 ± 1.75 |
| T0 160M | 4.7960 / 121.02 | 4.7762 / 118.66 | 4.9438 / 140.31 | **4.8387 ± 0.0916** / 126.33 ± 11.94 |

### 2.2 Paired differences (new variant minus T0, NLL)

| Control | seed 42 | seed 43 | seed 44 | Mean | Direction consistent |
|---|---|---|---|---|---|
| T0 160M | −0.0014 | −0.0023 | −0.1776 | **−0.0605** | yes (all better) |

### 2.3 Cost

Wall time per looped seed: 635.0 / 615.4 / 640.3 s (160M T0: 529 / 476 / 485 s). Peak memory ~3179 MiB. Training FLOPs ≈ 1.5× vanilla; inference 339,148,800 FLOPs/token ≈ 1.35× vanilla (T0: 251,854,848).

## 3. Conclusions

- **Decision (pre-registered rule): "scale revives Looped."** The paired mean is −0.0605 (threshold −0.05) with all three seeds in the looped direction, so the formal rule fires and the conclusion is rewritten as stated.
- **The honest reading is narrower than the verdict.** Seeds 42 and 43 are ties (−0.0014, −0.0023 — two-thousandths of an NLL point). The entire mean effect rides on seed 44 (−0.1776), and it rides on the T0 side, not the looped side: looped seed 44 (4.7662) is only 0.01–0.03 better than looped seeds 42/43, while T0 seed 44 (4.9438) is 0.15–0.17 worse than T0 seeds 42/43 — worse, in fact, than 70M T0's seed-42 result (4.9271). T0 seed 44 is a legitimate run (single attempt, exit 0, all validations passed; its training loss was also the worst of the six runs and its grad-norm median 93.3 vs 54–68 for the other five — a noisy optimization draw, not a bug), but it is one seed. The defensible summary: **looped ties T0 on 2 of 3 seeds at 160M, and the mean win depends on one noisy T0 seed.** A fourth or fifth seed would tell whether T0 seed 44 is a tail event; without it, "revives" overstates the evidence.
- **The robust secondary finding is stability.** Looped 160M has SD 0.0147 across seeds vs T0 160M's 0.0916. At 70M the same pattern held in reverse for placement (middle loop SD 0.066 vs top loop 0.309). The looped variant is the more seed-stable of the two at both scales tested.
- **Scale helps both, slightly more for looped.** 70M → 160M: looped 4.8710 → 4.7782 (−0.093), T0 4.8998 → 4.8387 (−0.061). The looped-vs-T0 gap widened from −0.029 (70M, a tie) to −0.060 (160M) — in the looped direction, but again, the widening is the seed-44 effect.
- **What this means for the main question.** The NO is no longer clean. At 70M the answer was NO (tie at 600, loss at 1200, no downstream comeback). At 160M the formal rule says the looped variant wins — but the win is not robust to the honest reading above. The main question is now: is the 160M advantage real or a one-seed artifact? That is a question for more seeds, not for a new variant.

## 4. Confounds and limitations

- WikiText-2 only; 160M tiny-scale; three seeds; no downstream benchmark in this experiment.
- The "scale revives Looped" decision follows the pre-registered mean rule; the per-seed breakdown (two ties + one T0 outlier) must be reported alongside it. The "better" branch of the rule required direction consistency, which holds, but the magnitude is carried by a single seed.
- No 1200-step version of the 160M middle loop; no test of other placements or R at 160M; no 410M (8GB VRAM).
- Inference FLOPs are analytical estimates (cached sequential convention), not profiler measurements; the measured prefix-recomputation cost is higher.
- Training FLOPs (≈ 1.5× vanilla) are analytical estimates, not profiler measurements.
