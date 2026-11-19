# A1 Addition Experiment Report: Strict 3-Digit Training, 4-Digit Testing

**Arm**: A1 (full reproduction, Q5) | **Date**: 2026-11-19 | **Status**: Complete (4/4 runs) | **Verdict**: Sanity gate failed; Q5 uninterpretable; one key finding established (see §4).

## 1. Question (Q5)

Under strict 3-digit-only training, can the new looped implementation (looped2, with input injection) generalize to 4-digit addition better than vanilla?

- Sanity gate (design §3.4): per-model 3-digit accuracy ≥90%. If the gate fails, the generalization comparison is ruled "uninterpretable".
- Main verdict (design §3.4): looped − vanilla > 5pp on 4-digit AND same direction in both seeds → "looped generalizes better on 4-digit under the new implementation". 5–6 digit reported only, no verdict.
- Background: the teacher's question (train on 3-digit, test on 4-digit). EXP-23 (mixed 1–3 digit training, old implementation) reached 97.0%/95.4% in-distribution but 0%/0% on 4–6 digit — neither model learned the algorithm, only the answer-length distribution. A1 re-tests this with the new implementation and strict 3-digit-only training.

## 2. Methods

- **Models**: Pythia-70M architecture (6 layers, hidden 512), trained from scratch (vendored initialization, from_scratch_init). Vanilla: plain 6 layers. Looped: looped2 (2,2,2) (prelude 2 layers, core 2 layers looped, coda 2 layers), input injection (e added every round), sandwich norms, random initial state s0 (truncated normal, std=√(2/5)), random R sampler (looped2 defaults), emb_scale=√512≈22.627.
- **Data**: strict 3-digit addition, both operands 100–999 (strict_3digit=true, max_train_digits=3), no curriculum (no 1–2 digit warm-up). This is the key difference from EXP-23's mixed training.
- **Training**: 8000 steps, batch 256, AdamW lr 1e-3, warmup 100, weight_decay 0.01, betas (0.9, 0.95). Seeds 42, 43.
- **Evaluation**: 3/4/5/6-digit, 500 items per length (eval seed 1234, generated on the fly), exact match. Looped evaluated at r=4, s0-seed 0. Evaluator: src.evaluate_looped2 addition (corrected: do NOT use src.evaluate_addition.py — it silently evaluates looped2 checkpoints as vanilla).
- **Cost**: ~18–21 min per run (vanilla ~17.5–17.7 min, looped ~20.8–21.2 min); 4 runs ≈ 77 min GPU total.
- **Execution**: the A1 driver was killed by SIGTERM at 10:12 (exec session terminated, cause unknown); the s43 vanilla training process survived as an orphan and kept running; a resume driver (setsid, detached) finished the rest. All 4 runs completed and committed locally; push pending (github.com DNS failure).

## 3. Results

| Model | Seed | Final loss | 3-digit (/500) | 4-digit | 5–6 digit | Time |
|---|---|---|---|---|---|---|
| vanilla | 42 | 0.6999 | 161 (0.322) | 0 | 0 | 17.5 min |
| looped | 42 | 0.7230 | 161 (0.322) | 0 | 0 | 20.8 min |
| vanilla | 43 | 0.7239 | 133 (0.266) | 0 | 0 | 17.7 min |
| looped | 43 | 0.6804 | 139 (0.278) | 0 | 0 | 21.2 min |

- 3-digit mean: vanilla 0.294 (147/500), looped 0.300 (150/500). Paired difference (looped − vanilla): seed 42: 0.000; seed 43: +0.012 (+6/500). Mean difference +0.006.
- 4-digit: 0/500 in all 4 runs. 5–6 digit: 0/500 in all 4 runs.
- Per-length detail: looped-s42 scored 6/500 (0.012) on 1-digit; the other three runs 0/500; 2-digit 0/500 in all runs. (1–2 digit are outside the training distribution and outside the sanity gate; for reference only.)
- Final loss: vanilla mean 0.712, looped mean 0.702. Loss plateau ~0.70 (EXP-23 mixed training reached 0.012).
- Sanity gate: best run 0.322, far below 0.90. Gate failed.

## 4. Key finding (established)

1. **Sanity gate failed; Q5 uninterpretable.** Best 3-digit accuracy across four runs is 0.322, far below 0.90. Per the frozen rule, the 4-digit generalization comparison is ruled "uninterpretable" — this is neither a looped loss nor a looped win; the experimental setup failed to establish a comparable baseline. The 4-digit zeros are reported as numbers only, with no mechanistic conclusion.
2. **The failure is in the training setup, not the looped architecture.** Vanilla (no loop, no recurrence) also only reached a 0.294 mean, essentially the same as looped's 0.300. Vanilla's failure cannot be explained by the looped architecture; the shared cause must lie in what the two models share: the training setup (strict 3-digit-only, no curriculum; and/or the new from-scratch initialization + emb_scale √512).
3. **Both models learned the same partial digit-wise skill.** Seed 42: both models scored exactly 161/500 on 3-digit. A per-digit accuracy of ~70% implies a full-match probability of ~0.7³ ≈ 0.34, consistent with the observed 0.322 — suggesting both models learned "each digit roughly right, but all three correct only a third of the time", not that one learned the algorithm and the other did not.
4. **Seed 43 replicates; the finding is solid.** Seed 43: 0.266/0.278, in the same "far below 90%" range as seed 42's 0.322; paired differences are inconsistent in direction (0 and +0.012). Two seeds, two models, four runs point to the same conclusion: the strict 3-digit-only from-scratch setup itself cannot learn 3-digit addition.
5. **Comparison with EXP-23 reinforces "setup problem".** EXP-23 mixed training (1–3 digit) reached 97.0%/95.4% in-distribution with final loss 0.012; A1 strict 3-digit-only reached final loss ~0.70 and 3-digit ~0.30. Same task, same model scale: mixed training learns it, strict 3-digit-only does not — the difference is in the data curriculum (and/or initialization), not in looped vs vanilla. Caveat: EXP-23 and A1 differ in TWO respects at once (curriculum: mixed vs strict 3-digit-only; implementation/initialization: old implementation vs new implementation + emb_scale), so this cannot be attributed to curriculum alone — see §5.

## 5. Problems and limitations

- **Two changes happened at once and cannot be separated**: strict 3-digit-only (no curriculum) and the new from-scratch initialization + emb_scale √512 were introduced together; vanilla's failure could come from either or both. Follow-up experiments must separate them one factor at a time (see §6).
- **Whether 8000 steps is under-training**: final loss plateaued at ~0.70 (EXP-23 mixed training reached 0.012) and loss fell slowly over 8000 steps — this looks more like "cannot learn" than "has not learned enough", but longer training or a different lr cannot be ruled out. Strictly, under-training is not excluded.
- **Evaluation uses exact match only**: per-digit accuracy was not measured directly. The "~70% per digit" figure is back-calculated from 0.322 (0.7³≈0.34), not a direct measurement. Follow-ups should measure per-digit accuracy directly (see §6 D-A1-3).
- **Two seeds is a small sample**: paired differences (0 and +0.012) show no direction, but two seeds cannot statistically confirm "no difference between looped and vanilla". However, this report's conclusions (sanity gate failed, Q5 uninterpretable) do not depend on seed count — both seeds are far below the gate.
- **1–2 digit results are for reference only**: training contained only 3-digit items; 1–2 digit are out of distribution. Looped-s42's 1-digit 6/500 is noise-level and is not interpreted.
- **The 4/5/6-digit zeros are consistent with the gate failure and do not mean "generalization failed"**: with no baseline established, the out-of-distribution numbers carry no information.

## 6. Follow-up experiment designs (design only, do not run)

Per the user's rule: keep designing follow-up experiments when an important finding appears, but think them through and verify them before running. Designs only, no runs; each design must pass the "question, cheaper alternative, method check, literature, preregistration" gate before running.

- **D-A1-1 (top priority): curriculum control — vanilla single arm.** Question: is the strict 3-digit-only failure due to the missing curriculum? Design: vanilla from scratch with a curriculum 1-digit → 2-digit → 3-digit (about 2000 steps per stage, 8000 total, same budget as A1), everything else identical to A1 (same initialization, same emb_scale, same lr). Verdict: 3-digit ≥90% → "curriculum is the key factor"; still <90% → the problem is in initialization/hyperparameters (go to D-A1-4). Cheaper alternative: first probe with a 2000-step short run (a curriculum effect should show early), rather than burning the full 8000 steps. Cost: 1 run, ~18 min.
- **D-A1-2: mixed-training control — re-run EXP-23 with the new implementation.** Question: can the new implementation (looped2/vanilla from scratch + emb_scale) itself reproduce EXP-23's 97%? Design: vanilla and looped, 1 run each (seed 42), mixed 1–3 digit training, 8000 steps, otherwise identical to A1. Verdict: 3-digit ≥90% → the new implementation is fine and A1's failure is pinned on "strict 3-digit-only" (curriculum); still <90% → the new implementation/initialization is at fault (go to D-A1-4). Cost: 2 runs, ~40 min. Note: D-A1-1 and D-A1-2 are alternatives (both test "curriculum vs initialization"; D-A1-1 is more direct and cheaper; if D-A1-1 is done, D-A1-2 can be skipped).
- **D-A1-3: per-digit accuracy diagnosis (zero training cost).** Question: does the failure sit in carry propagation or in digit-wise mapping? Design: use the existing A1 checkpoints (s42/s43) and measure per-digit accuracy on the 3-digit test set (hundreds/tens/ones separately) plus error rates at carry positions. No training, evaluation only. Verdict: high ones-digit accuracy with low hundreds-digit accuracy points to carry propagation as the bottleneck; uniform ~70% across digits points to the digit-wise mapping itself. Cost: evaluation only, minutes. — This is the cheapest diagnosis and should be done first.
- **D-A1-4: initialization/hyperparameter ablation — vanilla single arm.** Question: do emb_scale √512 or lr 1e-3 block learning? Design: single-factor changes on vanilla (one at a time): (a) emb_scale 1.0 (remove the √512 scaling); (b) lr 3e-4; (c) steps 8000→16000. One run each (seed 42). Verdict: any change that pushes 3-digit past 90% identifies the key factor. Cost: ~18–36 min per run. Note: do D-A1-3 first (cheap diagnosis) before choosing which ablation to run, to avoid burning GPU blindly.
- **D-A1-5: formal re-test of Q5 (only after the 3-digit gate passes).** Question: does looped generalize to 4-digit better than vanilla? Design: under the setting that pushes 3-digit past 90% (from D-A1-1 or D-A1-2), vanilla and looped, 2 runs each (seeds 42/43), evaluate 3/4/5–6 digit with A1's verdict rule (4-digit looped − vanilla > 5pp with consistent direction in both seeds). Cost: 4 runs, ~80 min. Note: do not run this while 3-digit is below the gate (it would repeat A1's uninterpretable outcome).

## 7. Conclusion

- A1 does not answer Q5: the sanity gate (3-digit ≥90%) failed in all four runs, so per the frozen rule the 4-digit generalization comparison is uninterpretable. The 4-digit zeros are reported as numbers only, with no mechanistic conclusion.
- A1 establishes: the strict 3-digit-only from-scratch setup itself cannot learn 3-digit addition (two models, two seeds, four runs agree; vanilla also only ~0.29). The problem is in the training setup (no curriculum, and/or the new initialization + emb_scale), not in the looped architecture. The identical 161/500 in seed 42 suggests both models learned the same partial digit-wise skill.
- Next steps: first the zero-cost per-digit diagnosis (D-A1-3), then the curriculum control (D-A1-1) to push 3-digit past 90%, and only then re-test Q5 (D-A1-5). Do not run experiments that have not passed the design gate.

---

**Data and reproduction**: configs, training logs, eval results, commits: runs/reproduction/a1-*.json, a1-*-train.out, a1-*-eval.out, a1-*/results.json, a1-*/complete.json; commits a87d3b2 (vanilla-s42), b8634ef (looped-s42), 83915bd (vanilla-s43), e32efca (looped-s43). Eval seed 1234, 500 items per length, looped evaluated at r=4. Push pending (github.com DNS failure).
