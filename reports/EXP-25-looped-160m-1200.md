# EXP-25 — 160M middle-layer loop × 1200 steps (does the revival survive longer training?)

> **Addendum — 2026-11-18 (Step 0 remediation).** The verdict in this report ("revival holds at 1200 steps") is now restated as: "EXP-25 shows a reportable positive signal — paired mean −0.19, all three seeds crossing the 0.05 line in the same direction — but not statistical confirmation (paired-difference 95% interval [−0.46, +0.08], two-sided p≈0.10); and the equal-FLOPs advantage is untested." The "1.5× training FLOPs" figures in §1.1/§2/§4 were the layer-call ratio (18/12), not the FLOPs ratio; the recomputed ratio is **1.3466** (corrected in place below). The equal-FLOPs control (T0×1616) is Step 1, pre-registered in `reports/STEP1-equal-flops-prereg.md`, not yet run. See the remediation plan (`remediation-plan.md`) and `first-principles-review.md` in the research notes.

## 0. Why this experiment was necessary

EXP-22 (160M×600) formally said "scale revives Looped" (paired mean −0.0605, threshold −0.05), but the honest reading was narrow: seeds 42 and 43 were ties (−0.0014, −0.0023), and the entire mean effect rode on one noisy T0 seed (seed 44: 4.9438, 0.15–0.17 worse than T0's other seeds). At 70M, the precedent pointed the other way: the middle-layer loop tied at 600 steps (EXP-19, paired mean −0.029) and reversed into a loss at 1200 steps (EXP-20, +0.2327). So the open question was sharp: is the 160M advantage real, or a 600-step artifact that reverses with budget like the 70M one did?

Decision rules (pre-registered before launch, control = 160M T0×1200 multi-seed mean): (a) looped paired mean better than T0 by more than 0.05 with a consistent direction → "revival holds at 1200 steps"; (b) looped paired mean worse than T0 by more than 0.05 → "reverses with budget" (the 70M pattern); (c) otherwise → "600-step artifact".

## 1. Method

### 1.1 Variant and training

- Identical to EXP-22 except steps 600 → 1200. Schedule: Huginn-style sandwich scaled to 12 layers by the 25% / 50% / 25% split — prelude [0,1,2] once, core [3,4,5,6,7,8] looped (R=2), coda [9,10,11] once. No input injection, no hyperconnections, no change to supervision.
- Cost (analytical, same convention as EXP-01/EXP-19/EXP-22): 18 layer calls; inference 339,148,800 FLOPs/token ≈ 1.35× vanilla (T0 160M: 251,854,848). Training FLOPs ≈ 1.3466× vanilla (recomputed 2026-11-18; the earlier figure was the layer-call ratio 18/12, not the FLOPs ratio).
- 1200 steps × seeds 42/43/44 for the looped variant; T0 seeds 43/44 are new runs, T0 seed 42 is reused from EXP-15 (runs/EXP-15-T0, NLL 4.8659, re-verified from its evaluation.json — not re-run). Unified settings unchanged: WikiText-2, seq 256, microbatch 1, grad_accum 16, AdamW lr 1e-4, betas (0.9, 0.95), weight decay 0.01, 20-step warmup, cosine decay to 10% floor, grad clip 1.0, bf16 autocast + fp32 master weights, SDPA. Main metric: validation-set NLL (977 chunks). Looped runs evaluated with `src/evaluate_looped.py`, T0 runs with `src/evaluate.py`.
- Pilot gate (100 steps, seed 42) before the full runs: passed with decision "continue" — gradient ratio 0.75 vs 160M T0's first 100 steps, data-order prefix matches EXP-01, no divergence.
- Validation by `scripts/summarize_exp25.py`; summary `runs/exp25-summary.json`. Configs: `configs/exp25_loopmid160_s42/s43/s44.json`, `configs/exp25_t0_s43/s44.json`. Check `check_exp25.py` passed (18 layer calls, 339,148,800 FLOPs/token).

## 2. Results

### 2.1 Endpoints (validation NLL / PPL, step-1200 checkpoint)

| Model | seed 42 | seed 43 | seed 44 | Mean ± SD |
|---|---|---|---|---|
| Loop-middle 160M (new) | 4.5726 / 96.80 | 4.6125 / 100.74 | 4.6743 / 107.16 | **4.6198 ± 0.0512** / 101.56 ± 5.23 |
| T0 160M | 4.8659 / 129.79 | 4.8076 / 122.44 | 4.7487 / 115.43 | **4.8074 ± 0.0586** / 122.56 ± 7.18 |

### 2.2 Paired differences (looped minus T0, NLL)

| Control | seed 42 | seed 43 | seed 44 | Mean | Direction consistent |
|---|---|---|---|---|---|
| T0 160M | −0.2933 | −0.1951 | −0.0744 | **−0.1876** | yes (all better) |

### 2.3 Cost

Wall time per looped seed: 1214.2 / 1185.5 / 1182.6 s (T0 seed 44: 965.2 s). Peak memory ~3333 MiB (looped), ~3336 MiB (T0). Training FLOPs ≈ 1.3466× vanilla (recomputed 2026-11-18; the earlier figure was the layer-call ratio 18/12, not the FLOPs ratio); inference 339,148,800 FLOPs/token ≈ 1.35× vanilla.

## 3. Conclusions

- **Decision (pre-registered rule): "revival holds at 1200 steps."** The paired mean is −0.1876 (threshold −0.05) with all three seeds in the looped direction, so the formal rule fires.
- **Unlike EXP-22, this verdict survives the honest reading.** All three seeds agree, and the smallest win (−0.0744, seed 44) alone exceeds the 0.05 threshold. The seed-44 story flipped: at 600 steps T0 seed 44 was the noisy outlier (4.9438) that carried the whole EXP-22 effect; at 1200 steps T0 seed 44 recovered to normal (4.7487, the best of the three T0 seeds), and the looped variant still won on that seed. The win does not depend on any single seed.
- **The 70M pattern does not repeat at 160M.** At 70M the looped advantage went from a tie at 600 steps (−0.029) to a loss at 1200 (+0.2327). At 160M it goes the other way: the paired mean grows from −0.0605 (600 steps) to −0.1876 (1200 steps). Longer training helps the looped variant at 160M and hurts it at 70M.
- **Seed stability is now comparable.** Looped 160M SD 0.0512 vs T0 160M SD 0.0586 at 1200 steps. (At 600 steps looped was far more stable, 0.0147 vs 0.0916 — that gap closed because T0's noisy seed recovered, not because looped got noisier.)
- **What this means for the main question.** The NO is now scale-dependent. At 70M the answer stays NO (thought loses, looped ties then loses, no downstream comeback). At 160M the looped variant beats vanilla on NLL at 1200 steps — but at ~1.35× training FLOPs, so the strict equal-compute reading still needs that caveat stated: this is a win at matched steps, not at matched FLOPs. The looped line is reopened at 160M; the thought line stays closed (EXP-18). The next question is whether the 160M looped advantage holds on downstream tasks (EXP-26) and at larger scales (not testable on this card).

## 4. Confounds and limitations

- WikiText-2 only; 160M tiny-scale; three seeds; no downstream benchmark in this experiment (EXP-26 follows).
- Training FLOPs (≈ 1.3466× vanilla) are analytical estimates, not profiler measurements; the comparison is at matched steps (1200), not matched FLOPs.
- Inference FLOPs are analytical estimates (cached sequential convention), not profiler measurements; the measured prefix-recomputation cost is higher.
- No 1200-step test of other placements or R at 160M; no 410M (8GB VRAM).
- T0 seed 42 was reused from EXP-15 rather than re-run (re-verified NLL 4.8659 from its evaluation.json).
