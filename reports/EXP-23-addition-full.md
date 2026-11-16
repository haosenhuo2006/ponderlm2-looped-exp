# EXP-23: Multi-digit addition, full run (single seed)

## Purpose

The main question (does repeated computation beat vanilla at matched compute?) was answered NO on language-modeling NLL, but WikiText-2, ARC, and LAMBADA never gave loop a chance to show its advantage: in the literature, loop wins on multi-step reasoning tasks. The EXP-23 pilot (2000 steps) came back null + undertrained — in-distribution accuracy was only ~70%, below the 90% sanity gate, so the generalization comparison was uninterpretable. This full run trains both models to the sanity gate first, then compares generalization.

## Design (fixed before running; identical to the pilot except steps)

- **Task**: decimal addition. Train on short operands only (1–3 digits); test in two bands: same lengths 1–3 (did it learn the task) and 4–6 digits (length generalization: did it learn the algorithm).
- **Format**: `{a}+{b}={sum}` + EOS; loss only on answer tokens (including EOS). Verified tokenization: `123+456=579` → `["123", "+", "456", "=", "5", "79"]`, EOS id 0.
- **Train data**: operands sampled independently, digit-length uniform in {1,2,3}, value uniform in [10^(d−1), 10^d−1] (d=1: 0..9). Online generation, seed 42. 8000 steps × batch 256 = 2,048,000 examples.
- **Test data**: L=1..6, 500 examples each, both operands exactly L digits (L=1: 0..9), fixed seed 1234, 3000 examples total. Same test set as the pilot.
- **Models**: Pythia-70M architecture (6 layers, hidden 512, vocab 50304), **trained from scratch** (random init via the vendored GPTNeoX class; no pretrained weights — cleanest comparison, same as Saunshi et al.). Vanilla and middle-layer loop (2,2,2) R=4, the EXP-19/20 configuration, share the same `looped.forward` path and differ only in the layer schedule.
- **Hyperparameters**: AdamW lr 1e-3, betas (0.9, 0.95), weight decay 0.01, 100-step warmup, cosine decay to 10%, clip 1.0, bf16 autocast + fp32 master weights, SDPA. Batch 256, single microbatch. Seed 42.
- **Evaluation**: greedy decoding from the prompt `{a}+{b}=`, max 8 new tokens, stop at EOS, exact match on the decoded number, per-length accuracy.
- **Pre-run checks**: full test suite 50/50 passed with the repo's own `.venv/bin/python` (transformers 4.46.1) before training; configs differ only in `run_dir` and `loop`.

## Decision rule (pre-registered, same as the pilot)

Looped beats vanilla by > 5pp on long lengths (4–6 digits) → loop advantage holds at tiny scale on its home turf. Otherwise (including both at ~0) → null result (equally meaningful: even on its home turf, tiny-scale loop confers no length generalization).

Sanity gate (pre-registered): both models in-distribution (1–3 digits) ≥ 90%. If the gate failed after 8000 steps, report the numbers and the confound and stop — do not add steps unilaterally.

Confound note (pre-registered): if the two models' in-distribution accuracy is not aligned, say so in the OOD comparison.

## Results

| Operand length | Vanilla (correct/500) | Looped (correct/500) |
|---|---|---|
| 1 digit | 500 (100%) | 500 (100%) |
| 2 digits | 500 (100%) | 500 (100%) |
| 3 digits | 455 (91.0%) | 431 (86.2%) |
| 4 digits | 0 (0%) | 0 (0%) |
| 5 digits | 0 (0%) | 0 (0%) |
| 6 digits | 0 (0%) | 0 (0%) |
| **In-distribution mean (1–3)** | **97.0%** | **95.4%** |
| **OOD mean (4–6)** | **0%** | **0%** |
| Overall (3000) | 48.5% | 47.7% |

## Training curves

| Step | Vanilla loss | Looped loss |
|---|---|---|
| 100 | 2.7565 | 2.8149 |
| 1000 | 0.9359 | 1.2736 |
| 2000 | 0.4021 | 0.5339 |
| 4000 | 0.2035 | 0.2329 |
| 6000 | 0.0412 | 0.0596 |
| 8000 | 0.0124 | 0.0397 |

Wall time: vanilla 1068.5 s, looped 1211.0 s. Both losses drop sharply between steps 4000 and 6000 (vanilla 0.20 → 0.04, looped 0.23 → 0.06) — the point where 3-digit addition clicks. Looped's final loss is higher than vanilla's at equal steps (0.0397 vs 0.0124): looped optimizes slower, the same pattern as on WikiText.

## Diagnosis: what the errors look like

Greedy predictions on the first 6 test examples of each length (test seed 1234, same examples as the pilot diagnosis):

- **3 digits**: both models are mostly right now; the remaining errors are near-misses on carries, e.g. vanilla `333+227=560` → `559`, `384+396=780` → `782`; looped `918+272=1190` → `1193`. Same failure mode as the pilot, much rarer.
- **4 digits**: both models output short 2–3 digit answers in the familiar training length range, e.g. vanilla `7218+1952=9170` → `389`, `4717+1563=6280` → `109`; looped `7218+1952=9170` → `500`, `2837+3516=6353` → `72`. Neither model learned the algorithm; both learned the output-length distribution of the training data.

The 3-digit gap (vanilla 91.0% vs looped 86.2%, 4.8pp on n=500, ~2.4 SE) is suggestive but not conclusive on a single seed; do not overclaim it. The in-distribution gap overall (97.0% vs 95.4%, 1.6pp) is the same story: looped is somewhat less trained at equal steps, consistent with its higher final loss.

## Verdict

1. **Sanity gate: PASSED.** In-distribution accuracy is 97.0% (vanilla) and 95.4% (looped), both ≥ 90%. The OOD comparison is interpretable this time.
2. **OOD: 0% vs 0%.** The pre-registered rule (> 5pp win for looped on 4–6 digits) is not met. **Verdict: null — loop advantage not demonstrated on its home turf at tiny scale.**
3. **Confound, stated plainly:** in-distribution accuracy is not aligned — looped trails vanilla by 1.6pp overall (4.8pp on 3 digits), consistent with slower optimization. This confound cannot explain a 0%-vs-0% tie (even at equal in-distribution accuracy there is no sign looped would generalize), but the comparison carries it and any follow-up should train both to matched in-distribution accuracy or full convergence.
4. **What it means:** both models learned 1–3 digit addition and neither learned the algorithm — they learned the answer-length distribution instead. At tiny scale, the middle-layer loop confers no length generalization on addition, its home-turf task. Combined with EXP-18/20/21, the main question's NO now holds on language modeling NLL, on downstream tasks, and on an algorithmic reasoning task.

## Limitations

- Single seed (42); no multi-seed replication.
- Addition is one algorithmic task; parity, subtraction, or multiplication would test generality of any finding.
- Greedy decoding only; no sampling-based evaluation.
- In-distribution accuracy not aligned between the two models (see verdict point 3); a matched-accuracy comparison would be cleaner.
- 8000 steps passed the gate, but looped's 3-digit accuracy (86.2%) still trails vanilla's (91.0%) — training both to full convergence is the stricter version of this comparison.

## Files

- Code (unchanged from the pilot): `src/addition.py`, `src/train_addition.py`, `src/evaluate_addition.py`; tests `tests/test_addition.py`; new configs `configs/exp23_full_vanilla.json`, `configs/exp23_full_looped.json`.
- Runs (gitignored): `runs/EXP-23-full-vanilla/`, `runs/EXP-23-full-looped/` (config.json, metadata.json, train.jsonl, checkpoints/final, results.json).
- Design: `~/workspace/phase9-exp23-design.md`.
- Base commit: `9b7be71`.
