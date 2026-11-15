# EXP-20 — Middle-layer loop × 1200 steps, multi-seed

## 0. Why this experiment was necessary

EXP-19 showed the Huginn-style middle-layer loop (2,2,2) R=4 tying vanilla at 600 steps (4.8710 ± 0.0657 vs T0×600 4.8998 ± 0.0397) — the first repeated-computation variant to tie vanilla at all, and a 0.904 improvement over the top loop T4. But a tie within noise is not a win, and the honest reading was "tie, not win". The question here is narrow and pre-registered: does the tie hold or grow with budget? The loop gets the same 1200-step budget that T0 and T8 already had in EXP-18.

Decision rules (fixed before running, in the phase-6 task brief): (a) loop-middle×1200 mean better than T0×1200 by more than 0.05 with all three seeds better → Looped wins at 1200 steps, main question YES on the Looped side; (b) mean difference within ±0.05 → tie persists; (c) mean worse than T0×1200 by more than 0.05 → Looped loses at 1200 steps.

## 1. Method

### 1.1 What changed relative to EXP-19

Nothing except the step budget. Same configuration (Huginn-style (2,2,2): Prelude [0,1], core [2,3] looped R=4, Coda [4,5]; 12 layer calls; inference FLOPs/token 130,154,496; training FLOPs ≈ 2× vanilla), same seeds 42/43/44, steps 600 → 1200. No `src/` changes. New files: `configs/exp20_loopmid_s42/s43/s44.json` (derived from the EXP-19 configs with steps and run_dir changed), `configs/exp20_loopmid_pilot.json`, `scripts/check_exp20.py`, `scripts/summarize_exp20.py`, `scripts/run_exp20.sh` (all adapted from the EXP-19 versions). `check_exp20.py` verified the configs against the T4 baseline (steps 1200, loop {block_size 2, repetitions 4, start 2}, 12 calls, 130154496 FLOPs/token) before training started.

### 1.2 Training and evaluation

- 1200 steps × seeds 42/43/44. Unified settings unchanged: Pythia-70M, WikiText-2, seq 256, microbatch 1, grad_accum 16, AdamW lr 1e-4, betas (0.9, 0.95), weight decay 0.01, 20-step warmup, cosine decay to 10% floor, grad clip 1.0, bf16 autocast + fp32 master weights, SDPA. Main metric: validation-set NLL (977 chunks, 249135 predicted tokens), evaluated with `src/evaluate_looped.py` on the step-1200 checkpoint.
- Pilot gate (100 steps, seed 42) before the full runs: passed with decision "continue" — 100 ordered steps, finite loss/grad_norm, no divergence (last-20 loss mean 7.82 vs first-20 10.04), grad-norm median ratio vs T4's first 100 steps 0.158 (gate range [0.1, 10]), data-order prefix matches.
- Validation by `scripts/summarize_exp20.py` (1200 ordered steps, K=0 rows, data orders regenerated independently per seed, seed-42 first-9600 indices match EXP-01, source hashes, evaluation fields including `block_size == 2`, `repetitions == 4`, `loop_block_start == 2`, `layers == [2, 3]`, inference FLOPs 130154496). Controls (T0×1200, T8×1200, loop-middle×600) are read from the validated phase-5 summaries `runs/exp18-summary.json` and `runs/exp19-summary.json`, with recomputed means asserted against the reported values (4.7147 / 4.8319 / 4.8710, tolerance 1e-4).
- Execution notes: pipeline ran 23:01–23:39 CST on 2026-11-15, zero failures, zero retries. Two SSH monitor sessions dropped at 23:17 (transient tunnel failure, not laptop sleep); the nohup'd pipeline was unaffected and the watcher reconnected. The laptop did not sleep during the run.

## 2. Results

### 2.1 Endpoints (validation NLL / PPL, step-1200 checkpoint unless noted)

| Model | seed 42 | seed 43 | seed 44 | Mean ± SD |
|---|---|---|---|---|
| Loop-middle × 1200 (new) | 4.9340 / 138.93 | 4.8612 / 129.18 | 5.0470 / 155.56 | **4.9474 ± 0.0936** / 141.22 ± 13.34 |
| T0 × 1200 | 4.7428 / 114.75 | 4.7322 / 113.55 | 4.6691 / 106.60 | **4.7147 ± 0.0399** / 111.63 ± 4.40 |
| T8 × 1200 | 4.8207 / 124.05 | 4.8810 / 131.76 | 4.7941 / 120.79 | **4.8319 ± 0.0445** / 125.54 ± 5.63 |
| Loop-middle × 600 (EXP-19) | 4.9433 / 140.23 | 4.8551 / 128.39 | 4.8148 / 123.32 | **4.8710 ± 0.0657** / 130.65 ± 8.68 |

### 2.2 Paired differences (new variant minus control, NLL)

| Control | seed 42 | seed 43 | seed 44 | Mean | Direction consistent |
|---|---|---|---|---|---|
| T0 × 1200 | +0.191 | +0.129 | +0.378 | **+0.2327** | yes (all worse) |
| T8 × 1200 | +0.113 | −0.020 | +0.253 | **+0.1155** | no |
| Loop-middle × 600 | −0.009 | +0.006 | +0.232 | **+0.0764** | no |

### 2.3 Cost

Wall time per seed: 778.2 / 730.3 / 768.9 s (EXP-19's 600-step runs were ~370 s). Peak memory 1,458,087,424 bytes (~1390 MiB). Training FLOPs ≈ 2× vanilla; inference 130,154,496 FLOPs/token ≈ 1.43× vanilla (T0: 90,832,896).

## 3. Conclusions

- **Decision (pre-registered rule): "looped_loses_1200."** The mean gap to T0×1200 is +0.2327, above the +0.05 threshold, with all three seeds worse. Looped loses at 1200 steps. The tie from EXP-19 did not survive a bigger budget — it reversed.
- **Longer training made the loop itself worse.** Loop-middle×1200 (4.9474) is worse than loop-middle×600 (4.8710) by 0.076 on average, and its SD grew from 0.066 to 0.094. The regression is driven by seed 44 (+0.232; 4.8148 → 5.0470); seeds 42 and 43 are essentially flat (−0.009 / +0.006). A plausible mechanism — stated as a hypothesis, not a conclusion — is second-epoch overfitting on repeated WikiText-2 data, the same pattern as 160M T0's epoch-2 regression in EXP-15 (4.7960 → 4.8659). But note T0×1200 *improved* over T0×600 (4.8998 → 4.7147), so repeated data alone does not explain it: the looped variant specifically fails to benefit from the second epoch. Isolating this would need a fresh-data or LR-schedule control, which was not run.
- **At 1200 steps the ranking flips against the loop.** T8×1200 (4.8319) now beats loop-middle×1200 (4.9474) by 0.115 on average, reversing the 600-step order where the loop (4.8710) beat T8 (4.9944). Thought already lost to vanilla (EXP-18), so this is a contest for second place — but it shows the loop's advantage over thought was budget-dependent, not structural.
- **What this means for the main question.** On the Looped side, the answer is now NO at both budgets tested (tie at 600 steps, loss at 1200). Combined with EXP-18 (thought loses even as the strongest variant with ~8× training FLOPs), repeated computation does not beat vanilla at matched compute in this tiny-scale setup. The main question — "in matched compute, can repeated computation beat vanilla?" — is answered **NO** within the tested budgets and configurations. The loop's one remaining distinction is cost: it ties vanilla at 600 steps for ~2× training FLOPs, while thought needs ~8× to lose by less — but a tie at higher cost is not a win, and the tie does not survive a bigger budget.

## 4. Confounds and limitations

- WikiText-2 only; 70M tiny-scale; three seeds; no downstream benchmark in this experiment.
- The second-epoch regression mechanism is a hypothesis; no fresh-data control was run to isolate it.
- One loop placement ((2,2,2) R=4), one R, one scale; other placements, loop counts, or scales could behave differently — the "no" is scoped to what was tested.
- Inference FLOPs are analytical estimates (cached sequential convention), not profiler measurements; the measured prefix-recomputation cost is higher.
- Training FLOPs (≈ 2× vanilla) are analytical estimates, not profiler measurements.
