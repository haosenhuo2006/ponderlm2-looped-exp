# Step 1 — Equal-FLOPs control, direction A (T0 × 1616 steps): pre-registration

**Frozen 2026-11-18, before launch. This file is the decision rule of record; it must not be edited after the first run starts.**

## 0. Question

EXP-25 found a reportable positive signal for the 160M middle-layer loop at matched steps (looped 4.6198 ± 0.0512 vs T0 4.8074 ± 0.0586, paired mean −0.1876, all three seeds in the same direction) — but the comparison was at matched steps, not matched FLOPs, and the looped variant uses ~1.35× the training FLOPs per step. Step 1 asks: does the revival survive when the FLOPs are matched?

## 1. Design (frozen)

- Base: Pythia-160M, `EleutherAI/pythia-160m`, revision `50f5173d932e8e61f858120bcb800b97af589f46` (same revision as EXP-22/EXP-25).
- Corpus: WikiText-2 (Salesforce/wikitext raw-v1 mirror, pinned revision; train 2,409,267 tokens / 9,411 chunks; validation 250,139 tokens / 977 chunks). seq 256, microbatch 1, grad_accum 16 (4096 tokens/step — identical to EXP-25).
- New runs only: **T0 × 1616 steps, seeds 42/43/44** (3 runs). Looped arm **reuses the EXP-25 checkpoints** (no re-run):
  - `runs/EXP-25-loopmid160-s42/checkpoints/step-1200` — sha256 `1a4429380d28dccc7ae4cbb1a09de7882060e47fdffaf9d5aef7392bbd215a57`
  - `runs/EXP-25-loopmid160-s43/checkpoints/step-1200` — sha256 `f0723e8ca8a8a575b91bb4e60d318c6ce7ed3b25b67d15b519b2c622e0cbfc96`
  - `runs/EXP-25-loopmid160-s44/checkpoints/step-1200` — sha256 `0b7f9eae12f74ab280c87091c6f7a70995d467304bd891fc5ad6a78f12952824`
  - Verified loadable on CPU, 12 layers, hidden 768, vocab 50304, `loop_block_size 6 / loop_block_start 3 / loop_repetitions 2` (Step 0 evidence inventory §1).
- Hyperparameters: peak lr 1e-4, warmup 20 steps, cosine decay to 10% floor **rescheduled to 1616 steps** (reported explicitly), AdamW betas (0.9, 0.95), weight decay 0.01, grad clip 1.0, bf16 autocast + fp32 master weights, SDPA. Identical to EXP-25 except the schedule horizon.
- Data order: `order()` is a prefix extension — the first 1200 steps of each T0×1616 run have the **identical data order** to the same-seed T0×1200 run (verified in Step 0 evidence retrieval).
- Metric: validation-set NLL (977 chunks), same evaluation code as EXP-25 (`src/evaluate.py` for T0; looped values reused verbatim from `runs/exp25-summary.json`). Checkpoint selection: **final (step-1616) checkpoint only**. New T0 group passes the same symmetry check as EXP-25.

## 2. FLOPs accounting (frozen)

- Training FLOPs = 3 × parsed forward FLOPs, **including the lm_head**, excluding Adam updates, checkpoint saving, and evaluation. Wall-clock time is not a FLOPs proxy.
- Ratio (recomputed 2026-11-18): forward FLOPs/token T0 160M = 251,854,848; looped (3,6,3) R=2 = 339,148,800; **ratio 1.3466** (the audit's 1.5 was the layer-call ratio 18/12, not the FLOPs ratio).
- 1616 = round(1200 × 1.3466) = round(1615.92). Profiler measurement is a post-hoc check; if it deviates by more than ±5% the conclusion is annotated.
- Expected cost: ~71 minutes GPU for the three T0 runs (T0×1200 ≈ 1055 s each; 1616/1200 × 1055 ≈ 1420 s each). If measured time deviates by more than 10% from the 1055 s/run basis, the cost table is rescaled proportionally.

## 3. Decision rule (frozen before launch)

Paired comparison, looped minus T0, per seed (looped values from EXP-25, T0 values from the new runs):

- **Looped paired mean wins by more than 0.05 and all three seeds agree in direction** → "revival holds at equal FLOPs (within this recipe)".
- **Otherwise** → "revival is a compute artifact".

Report alongside: paired mean, per-seed differences, SD/SE of the differences, and the 95% paired-t interval. If the interval crosses 0 but the direction is consistent, write "operational threshold passed, statistically uncertain" — never "confirmed". **Three pairs is three pairs**: no fourth pair may be added after seeing the results on the grounds of "almost".

## 4. Branches (judge only — do not execute)

- (a) Looped still wins ≥ 0.05 with consistent direction → Step 2.
- (b) T0 catches up or overtakes (paired mean ≥ −0.05) → stop; looped line closes.
- (c) T0×1616 is clearly worse than T0×1200 (mean difference > +0.05, 2.75-epoch rollback) and looped still leads → Step 1B.
- (d) Mixed directions / very wide interval → "undecided". No automatic seed top-up (top-up of seeds 45/46 requires separate user authorization).

## 5. Coverage check (done 2026-11-18)

T0×1616 has never been run: the config step census across all runs shows only 100/600/1200/2000/8000 steps, and no data-order file exists for more than 1200 steps. EXP-25's three looped checkpoints and the three T0×1200 seed results are all present (see Step 0 evidence inventory).
