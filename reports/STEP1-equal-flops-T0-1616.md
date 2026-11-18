# Step 1 — Equal-FLOPs control, direction A (T0 × 1616 steps)

**Date:** 2026-11-18. **Pre-registration:** `reports/STEP1-equal-flops-prereg.md`
(frozen 2026-11-18, before launch). **Remediation plan:**
`~/workspace/research_notes/ponderlm2-looped-20261113/remediation-plan.md`, Step 1 section.
**Summary:** `runs/step1-summary.json`. **Pipeline:** `scripts/run_step1.sh`.

## 1. Question

EXP-25 found a reportable positive signal for the 160M middle-layer loop at matched
steps (looped 4.6198 ± 0.0512 vs T0 4.8074 ± 0.0586, paired mean −0.1876, all three
seeds in the same direction) — but the comparison was at matched steps, not matched
FLOPs, and the looped variant uses ~1.35× the training FLOPs per step. Step 1 asks:
does the revival survive when the FLOPs are matched?

## 2. Design (as pre-registered)

- Base: Pythia-160M, `EleutherAI/pythia-160M`, revision
  `50f5173d932e8e61f858120bcb800b97af589f46` (same revision as EXP-22/EXP-25).
- Corpus: WikiText-2 (Salesforce/wikitext raw-v1 mirror, pinned revision; train
  2,409,267 tokens / 9,411 chunks; validation 250,139 tokens / 977 chunks).
  seq 256, microbatch 1, grad_accum 16 (4096 tokens/step — identical to EXP-25).
- New runs only: T0 × 1616 steps, seeds 42/43/44 (3 runs). Looped arm reuses the
  EXP-25 checkpoints (no re-run).
- Hyperparameters: peak lr 1e-4, warmup 20 steps, cosine decay to 10% floor
  **rescheduled to 1616 steps** (schedule change is part of the design, reported
  explicitly), AdamW betas (0.9, 0.95), weight decay 0.01, grad clip 1.0, bf16
  autocast + fp32 master weights, SDPA. Identical to EXP-25 except the schedule
  horizon.
- Data order: `order()` is a prefix extension — the first 1200 steps of each
  T0×1616 run have the identical data order to the same-seed T0×1200 run
  (verified by `scripts/summarize_step1.py`: first 19200 indices match the
  EXP-25 T0 data-order for seeds 43/44; for seed 42 the EXP-25 T0 run reused
  EXP-15, so the prefix was verified against `runs/EXP-15-T0/data-order.json`).
- Metric: validation-set NLL (977 chunks), same evaluation code as EXP-25
  (`src/evaluate.py`, mode vanilla, split validation). Checkpoint selection:
  final (step-1616) checkpoint only.
- Runs: `runs/STEP1-T0-s42`, `runs/STEP1-T0-s43`, `runs/STEP1-T0-s44`.
  Pipeline: `scripts/run_step1.sh` (lock `runs/step1.lock`, preflight parameter
  check, 2 attempts per run; all three runs succeeded on the first attempt).
  Configs: `configs/step1_t0_s{42,43,44}.json` (verified by
  `scripts/check_step1.py` to equal the EXP-15 T0 baseline except
  steps/seed/run_dir).

## 3. FLOPs accounting (frozen)

- Training FLOPs = 3 × parsed forward FLOPs, including the lm_head, excluding
  Adam updates, checkpoint saving, and evaluation. Wall-clock time is not a
  FLOPs proxy.
- Ratio (analytic, `src/looped.py:flops`): forward FLOPs/token T0 160M =
  251,854,848; looped (3,6,3) R=2 = 339,148,800; **ratio 1.3466** (the audit's
  1.5 was the layer-call ratio 18/12, not the FLOPs ratio).
- 1616 = round(1200 × 1.3466) = round(1615.92).
- Profiler post-hoc check (`scripts/profile_flops_step1.py`, torch.profiler,
  microbatch 1, seq 256, CUDA; `runs/step1-flops-profile.json`): measured
  ratio = **1.349455**, deviation from analytic **+0.212%**, within ±5%: **yes**.
  No annotation required.

## 4. Results

| seed | looped (EXP-25, reused) | T0×1616 (new) | paired diff (looped − T0) | T0×1200 (EXP-25) |
|---|---|---|---|---|
| 42 | 4.5726 | 4.6055 | −0.0329 | 4.8659 |
| 43 | 4.6125 | 4.6841 | −0.0716 | 4.8076 |
| 44 | 4.6743 | 4.6423 | +0.0320 | 4.7487 |
| **mean ± SD** | **4.6198 ± 0.0512** | 4.6440 ± 0.0394 | −0.0242 ± 0.0524 | 4.8074 ± 0.0586 |

- Paired mean: **−0.0242**; SD of differences: 0.0524; SE: 0.0302;
  95% paired-t interval (t = 4.303, df = 2): **[−0.1543, +0.1060]** (crosses 0).
- Direction consistency: **no** — seed 44 flips sign (+0.0320).
- T0×1616 mean minus T0×1200 mean (rollback check): **−0.1635**. T0 improved
  by 0.16 with the extra 416 steps; there is no 2.75-epoch rollback, so branch
  (c) does not apply.
- PPL: T0×1616 mean 104.01 ± 4.10 (vs looped 101.56 ± 5.23, T0×1200 122.56 ± 7.18).
- Wall time per run (train only, `complete.json`): 1197.8 / 1185.2 / 1204.8 s
  (mean 1195.9 s). The pre-registered expectation was 1420 s from the 1055 s/run
  basis — measured time is 15.8% faster, beyond the ±10% band, so the cost
  table is rescaled proportionally per the pre-registration: Step 1 training
  took **59.8 minutes** of GPU, not the estimated 71 minutes. (The 1055 s basis
  itself was already flagged as unverified provenance; this measurement is now
  the recorded per-step rate, 0.74 s/step for T0-160M.)
- Evidence: `scripts/summarize_step1.py` re-validated every new run before
  aggregating (train.jsonl 1616 rows, data-order exact recomputation + prefix
  check, metadata source hash, complete.json, evaluation.json: finite NLL/PPL,
  exp(NLL) = PPL, predicted_tokens = 249135, seq = 256, mode vanilla, split
  validation). Looped values were reused verbatim from `runs/exp25-summary.json`
  and cross-checked against the three EXP-25 looped `evaluation.json` files.

## 5. Verdict (frozen decision rule)

- Branch: **(b)**
- Decision: **"revival is a compute artifact"**
- Annotation: none.

The paired mean (−0.0242) does not clear the 0.05 operational threshold, and
the direction is not consistent across seeds. The EXP-25 signal does not
survive matched FLOPs: giving vanilla the same training compute closes almost
the entire gap (0.1876 → 0.0242), and the remaining 0.02 is within noise
(95% interval [−0.1543, +0.1060]).

## 6. Limitations

- WikiText-2 only; 160M tiny-scale; three seeds; validation set reused for
  model selection (no independent test-set confirmation was run — branch (b)
  does not call for Step 2).
- Cosine schedule was rescheduled to the 1616-step horizon — the schedule
  change is part of the treatment, not a confound to remove.
- Equal FLOPs ≠ equal fresh data: T0×1616 ≈ 2.75 epochs vs looped×1200 ≈ 2.04
  epochs (9411 training chunks). Data repetition is part of the comparison,
  not a reason to cancel it.
- Three pairs is three pairs: no fourth pair was added after seeing the results.

## 7. What happens next

Branch (b): **stop, do not go to Step 2.** The conclusion is now "the EXP-25
signal is a compute artifact; at equal FLOPs the looped variant does not beat
vanilla on 160M". The looped line closes; no further GPU is spent on it
(no Step 1B, no seed top-up, no Step 2/3/4 for this line).
