# EXP-23: Multi-digit addition pilot (single seed)

## Purpose

The main question (does repeated computation beat vanilla at matched compute?) was answered NO on language-modeling NLL, but WikiText-2, ARC, and LAMBADA never gave loop a chance to show its advantage: in the literature, loop wins on multi-step reasoning tasks. This pilot tests loop on its home turf — iterative algorithms plus length generalization — and validates the full pipeline (data, training, evaluation) before any multi-seed run.

## Design (fixed before running)

- **Task**: decimal addition. Train on short operands only (1–3 digits); test in two bands: same lengths 1–3 (did it learn the task) and 4–6 digits (length generalization: did it learn the algorithm).
- **Format**: `{a}+{b}={sum}` + EOS; loss only on answer tokens (including EOS). Verified tokenization: `123+456=579` → `["123", "+", "456", "=", "5", "79"]`, EOS id 0.
- **Train data**: operands sampled independently, digit-length uniform in {1,2,3}, value uniform in [10^(d−1), 10^d−1] (d=1: 0..9). Online generation, seed 42. 2000 steps × batch 256 = 512,000 examples.
- **Test data**: L=1..6, 500 examples each, both operands exactly L digits (L=1: 0..9), fixed seed 1234, 3000 examples total.
- **Models**: Pythia-70M architecture (6 layers, hidden 512, vocab 50304), **trained from scratch** (random init via the vendored GPTNeoX class; no pretrained weights — cleanest comparison, same as Saunshi et al.). Vanilla and middle-layer loop (2,2,2) R=4, the EXP-19/20 configuration, share the same `looped.forward` path and differ only in the layer schedule.
- **Hyperparameters**: AdamW lr 1e-3, betas (0.9, 0.95), weight decay 0.01, 100-step warmup, cosine decay to 10%, clip 1.0, bf16 autocast + fp32 master weights, SDPA. Batch 256, single microbatch. Seed 42.
- **Evaluation**: greedy decoding from the prompt `{a}+{b}=`, max 8 new tokens, stop at EOS, exact match on the decoded number, per-length accuracy.

## Decision rule (pre-registered)

Looped beats vanilla by > 5pp on long lengths (4–6 digits) → loop advantage holds at tiny scale on its home turf. Otherwise (including both at ~0) → null result (equally meaningful: even on its home turf, tiny-scale loop confers no length generalization).

Sanity gate (pre-registered): if either model scores < 90% in-distribution (1–3 digits), the OOD comparison is uninterpretable; the fix is more steps, not reinterpretation.

## Results

| Operand length | Vanilla (correct/500) | Looped (correct/500) |
|---|---|---|
| 1 digit | 500 (100%) | 500 (100%) |
| 2 digits | 500 (100%) | 485 (97.0%) |
| 3 digits | 59 (11.8%) | 42 (8.4%) |
| 4 digits | 0 (0%) | 0 (0%) |
| 5 digits | 0 (0%) | 0 (0%) |
| 6 digits | 0 (0%) | 0 (0%) |
| **In-distribution mean (1–3)** | **70.6%** | **68.5%** |
| **OOD mean (4–6)** | **0%** | **0%** |
| Overall (3000) | 35.3% | 34.2% |

## Training curves

| Step | Vanilla loss | Looped loss |
|---|---|---|
| 100 | 2.756 | 2.815 |
| 600 | 1.209 | 1.440 |
| 1100 | 0.787 | 0.979 |
| 1600 | 0.432 | 0.579 |
| 2000 | 0.360 | 0.460 |

Wall time: vanilla 283 s, looped 302 s. Final grad norms: 0.68 / 0.78. Loss was still decreasing at step 2000 for both models.

## Diagnosis: why 3 digits fail

Greedy predictions on sampled test cases (8 per length):

- **3 digits**: both models output the right magnitude but make carry errors in the middle/low digits, e.g. vanilla `333+227=560` → `519`, `384+396=780` → `868`, `231+318=549` → `548`; looped `333+227=560` → `561`, `231+318=549` → `577`. The format and magnitude are learned; exact carry propagation is not.
- **4 digits**: both models output short 2–3 digit answers (e.g. `7218+1952=9170` → `88` / `616`), i.e. answers in the familiar training length range. The model never saw 4-digit operands, so it does not even produce the right output length.

Both failure modes are the same for vanilla and looped and are consistent with "2000 steps was not enough to learn 3-digit addition", not with a loop-vs-vanilla difference. The 3-digit gap (vanilla 11.8% vs looped 8.4%, 3.4pp on n=500) is within noise (~1.8 SE); do not overclaim it.

## Pilot verdict

1. **Pipeline: PASS.** End-to-end validated: 50/50 tests (43 existing + 7 new), 2-step smoke runs for both configs, eval smoke on smoke checkpoints, full 2000-step runs, checkpoint save/load, per-length evaluation.
2. **Signal: null + undertrained.** OOD is 0% / 0%, and in-distribution is only ~70% because 3-digit addition was not learned in 2000 steps. Per the pre-registered sanity gate, the OOD comparison is **uninterpretable** — this pilot cannot answer whether looped generalizes better.
3. **What the full run needs**: more steps until in-distribution accuracy is ≥ 90–95% (e.g. 8000 steps; at ~5 min per 2000 steps this is ~20 min per model). Note looped's train loss is higher than vanilla's at equal steps (0.460 vs 0.360) — looped optimizes slower, the same pattern as on WikiText. The full run should compare at matched in-distribution accuracy (or at least note the confound), or train both to convergence.

## Limitations

- Single seed (42); no multi-seed replication.
- 2000 steps was chosen as a pilot starting point and turned out to be too few for 3-digit addition.
- Addition is one algorithmic task; parity, subtraction, or multiplication would test generality of any finding.
- Greedy decoding only; no sampling-based evaluation.

## Environment note (worth recording)

Tests and training must run with the repo's own `.venv/bin/python` (transformers 4.46.1). The GPT2_Latent_Reasoning venv has transformers 5.18.0, where `GPTNeoXConfig` no longer has `rotary_pct`, breaking the vendored GPT-NeoX class (27 test errors). Nothing was wrong with the code; the wrong interpreter was used.

## Files

- Code: `src/addition.py`, `src/train_addition.py`, `src/evaluate_addition.py`; tests `tests/test_addition.py`; configs `configs/exp23_pilot_vanilla.json`, `configs/exp23_pilot_looped.json`.
- Runs (gitignored): `runs/EXP-23-pilot-vanilla/`, `runs/EXP-23-pilot-looped/` (config.json, metadata.json, train.jsonl, checkpoints/final, results.json).
- Design: `~/workspace/phase9-exp23-design.md`.
- Base commit: `de18cd2`.
