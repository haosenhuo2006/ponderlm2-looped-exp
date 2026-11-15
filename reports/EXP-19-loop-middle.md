# EXP-19 — Middle-layer looped variant (Huginn-style (2,2,2), R=4)

## 0. Why this experiment was necessary

T4 (top block [3,4,5] × 3) beat T1 by 0.52 at matched inference FLOPs, but it was unstable across seeds (multi-seed mean 5.775 ± 0.309; seed 43 flipped against T1) and still lost to T0. The literature says middle-layer looping is the strongest placement: Huginn's sandwich (prelude / looped core / coda), Done Right's controlled ablation attributing the gain to the sandwich itself, and SMELT's middle-50% loop. The question here is narrow and pre-registered: is loop placement the lever? Only one thing changes relative to T4 — where the loop sits.

Decision rules (pre-registered in `~/workspace/phase5-design.md`): (a) new-variant mean better than T4 by more than 0.1 with a consistent direction → middle > top; (b) new-variant mean ties or beats T0×600 (4.900 ± 0.040) → major progress; (c) otherwise loop position is not the lever.

## 1. Method

### 1.1 Code change (the only `src/` change in phase 5)

- `src/looped.py`: `configure(model, block_size, repetitions, start=None)`. `start=None` defaults to the top block (`len(layers) - block_size`), so T4 behavior is unchanged. A given `start` must be an integer with `0 <= start <= len(layers) - block_size`; it is stored as `model.config.loop_block_start`. `layer_schedule` reads it with a `getattr` fallback to the top block, so old checkpoints without the field still evaluate as before. `flops()`, `candidates()`, `selected()` are untouched (FLOPs depend only on the number of layer calls, not on position).
- `src/evaluate_looped.py`: passes `start=getattr(c, "loop_block_start", None)` through to `configure`; the result dict records `layers` as the actual block indices and adds `loop_block_start`. All other fields are unchanged.
- New test `test_middle_block_schedule` in `tests/test_looped.py` (GPU): schedule `[0,1] + [2,3]×4 + [4,5]`, parameter sharing, 70,426,624 parameters, `flops(2,4) == 130154496`, invalid `start` values rejected, default `start` is 3, old-checkpoint fallback. Full suite: 42/42 passed (`reports/exp18-exp19-tests.txt`).

### 1.2 Variant and training

- Schedule: Prelude layers [0,1] once, core layers [2,3] looped R=4, Coda layers [4,5] once. Layer calls: 2 + 8 + 2 = 12, same as T4. Inference FLOPs/token: 130,154,496, same as T4 (0.976× T1). Training FLOPs ≈ 2× vanilla. No input injection, no hyperconnections, no change to supervision — only the loop position changes.
- 600 steps × seeds 42/43/44 (same budget as T4). Unified settings unchanged: Pythia-70M, WikiText-2, seq 256, microbatch 1, grad_accum 16, AdamW lr 1e-4, betas (0.9, 0.95), weight decay 0.01, 20-step warmup, cosine decay to 10% floor, grad clip 1.0, bf16 autocast + fp32 master weights, SDPA. Main metric: validation-set NLL (977 chunks, 249135 predicted tokens), evaluated with `src/evaluate_looped.py`.
- Pilot gate (100 steps, seed 42) before the full runs: passed with decision "continue" — 100 ordered steps, finite loss/grad_norm, no divergence (last-20 loss mean 7.82 vs first-20 10.04), grad-norm median ratio vs T4's first 100 steps 0.158 (gate range [0.1, 10]), data-order prefix matches.
- Validation by `scripts/summarize_exp19.py` (600 ordered steps, K=0 rows, data orders regenerated independently per seed, source hashes, evaluation fields including `block_size == 2`, `repetitions == 4`, `loop_block_start == 2`, `layers == [2, 3]`, inference FLOPs 130154496). T4 and T0 control means are recomputed from `runs/EXP-05-T4`, `runs/EXP-08-seed43/44-T4`, `runs/EXP-01-T0`, `runs/EXP-08-seed43/44-T0` — not hardcoded.

## 2. Results

### 2.1 Endpoints (validation NLL / PPL, step-600 checkpoint)

| Model | seed 42 | seed 43 | seed 44 | Mean ± SD |
|---|---|---|---|---|
| Loop-middle (new) | 4.9433 / 140.23 | 4.8551 / 128.39 | 4.8148 / 123.32 | **4.8710 ± 0.0657** / 130.65 ± 8.68 |
| T4 (top loop) | 5.4855 / 241.17 | 6.0996 / 445.68 | 5.7387 / 310.67 | **5.7746 ± 0.3086** / 332.51 ± 103.99 |
| T0×600 | 4.9271 / 137.98 | 4.8542 / 128.28 | 4.9181 / 136.75 | **4.8998 ± 0.0397** / 134.34 ± 5.28 |

### 2.2 Paired differences (new variant minus control, NLL)

| Control | seed 42 | seed 43 | seed 44 | Mean | Direction consistent |
|---|---|---|---|---|---|
| T4 | −0.542 | −1.245 | −0.924 | **−0.9036** | yes (all better) |
| T0 | +0.016 | +0.001 | −0.103 | **−0.0288** | no |

### 2.3 Cost

Wall time per seed: 374.1 / 375.9 / 368.2 s (T4×600 was 411 s). Peak memory ~1390 MiB. Training FLOPs ≈ 2× vanilla; inference 130,154,496 FLOPs/token ≈ 1.43× vanilla (T0: 90,832,896).

## 3. Conclusions

- **Decision (pre-registered rule): "major progress."** Both conditions fired: middle beats top by 0.904 on average (threshold 0.1, all three seeds better), and the mean ties-or-beats T0×600 (−0.0288 ≤ 0).
- **The robust finding is the placement effect.** Moving the loop from the top block to the middle improves NLL by 0.90 on average, in the same direction on all three seeds. It also stabilizes the variant: SD drops from 0.309 (T4) to 0.066. T4's instability was not an inherent property of looping; it was a property of looping the top block.
- **The tie with vanilla must not be overclaimed.** The mean beats T0×600 by only 0.029, which is smaller than either SD (0.066 / 0.040), and the paired direction is inconsistent (seeds 42 and 43 are slightly worse, seed 44 is better by 0.103). The honest reading: looped-middle *ties* vanilla at 600 steps, at ~2× training FLOPs and ~1.43× inference FLOPs. It does not clearly beat vanilla.
- **What this means for the main question.** Thought is closed (EXP-18). Looped-middle is the first repeated-computation variant to tie vanilla at all — but a tie at higher cost is not a win. The main question ("does repeated computation beat vanilla at matched compute?") is still open: the next test is whether the tie holds or grows with budget (looped-middle × 1200 steps, multi-seed) and whether it survives a downstream check.

## 4. Confounds and limitations

- WikiText-2 only; 70M tiny-scale; three seeds; no downstream benchmark in this experiment.
- The vs-T0 comparison is a tie within noise; the "major progress" decision follows the pre-registered mean rule and should be read with the per-seed direction in mind.
- No 1200-step version of the middle loop yet; no test of other placements (e.g. a larger middle core) or of input injection.
- Inference FLOPs are analytical estimates (cached sequential convention), not profiler measurements; the measured prefix-recomputation cost is higher.
- Training FLOPs (≈ 2× vanilla) are analytical estimates, not profiler measurements.
