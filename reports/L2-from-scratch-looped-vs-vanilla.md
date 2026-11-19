# L2 From-Scratch Experiment Report: Injected Looped vs Vanilla (Same Token Count)

**Arm**: L2 (full reproduction, Q3) | **Date**: 2026-11-19 | **Status**: Complete (6/6 runs) | **Conclusion**: Branch (b) — looped does not win (paired mean −0.0453, below the −0.05 threshold; direction consistent 3/3)

## 1. The question (Q3)

Is CPT grafting the reason looped loses? Remove the graft, train from scratch — can the paper's method (input injection + sandwich norms + random R + random initial state) beat vanilla?

- Decision rule (design §7.2): per-seed paired difference `d_i = NLL_looped,i − NLL_vanilla,i` (i=42,43,44). (a) Looped wins: paired mean < −0.05 **and** all three seeds in the same direction; (b) otherwise looped does not win. **Note on the metric**: L2 is an "architecture comparison at the same token count", not "equal FLOPs" — both arms run 1200 steps, but looped costs more FLOPs per step (measured ratio ≈ 1.676, see §2). This distinction must be stated in the report.
- Relation to Q2 (design §1): Q2 (L1, CPT at equal FLOPs) asks whether the paper's method wins at equal FLOPs; Q3 asks whether CPT grafting is the cause of looped losing. The two conclusions can differ. But **L1 is frozen** (D3 zero-step damage +5.65, decided 2026-11-18 evening), so there is no CPT-side result to compare against — the Q2×Q3 joint interpretation is therefore incomplete, see §4.

## 2. Method

- **Model**: Pythia-160M architecture (12 layers, hidden 768), trained from scratch. Vanilla: plain 12 layers. Looped: looped2 (3,6,3) (prelude 3 layers, core 6 layers looped, coda 3 layers), input injection (add e every round), sandwich norms (core layers only, LayerNorm kept; vendored parallel-residual layers changed to serial), random initial state s0 (truncated normal, pre-truncation std=√(2/5), truncation ±3σ, decoupled from emb_scale), random R sampling (log-normal Poisson, nominal t=s=2, actual E[r]≈4.93, E[k]≈1.94, cap 12), emb_scale=√768≈27.71.
- **Initialization** (design §3.3, Huginn recipe): weights truncated normal std=√(2/(5h)); bias=0; LayerNorm weight=1, bias=0; attention/MLP output projections std=1/√(5·h·l_eff), nominal l_eff 30 (looped) / 12 (vanilla); lm_head at the plain scale √(2/(5h)). (The CPT arm does not use this initialization; L1 is frozen and unrelated to this arm.)
- **Data**: WikiText-2 raw v1, seq 256, 9411 train blocks / 977 validation blocks; each run 1200 steps × 16 × 256 ≈ 4.92M tokens (same token count for both arms).
- **Training**: 1200 steps, AdamW lr 3e-4, warmup 100, cosine to 10%, betas (0.9, 0.95), wd 0.01, clip 1.0, bf16 autocast. Seeds 42, 43, 44 (three seeds).
- **Evaluation**: validation-set NLL (977 blocks), eval r=4 (near the training mean), s0-seed 0. Evaluation entry point: src.evaluate_looped2 lm (corrected: do not use src.evaluate_addition.py; the from-scratch vanilla arm goes through the emb_scale plain forward, identified by config_from_checkpoint).
- **Compute verification**: cumulative training FLOPs for the s44 pair: vanilla 3.714×10¹⁵, looped 6.225×10¹⁵, ratio **1.676** (design value 1.6721, deviation ≈ +0.3%). Sampler statistics for looped-s44 (19,200 microbatches): E[r]≈4.97, E[k]≈1.94, matching the design (4.93 / 1.94) — the sampler behaved as designed.
- **Cost**: 6 runs, training ≈ 7832 s (≈ 2.18 h GPU) + evaluation. Vanilla per run 936/950/1128 s (15.6/15.8/18.8 min); looped per run 1544/1552/1722 s (25.7/25.9/28.7 min).
- **Execution**: the original driver run_l2.py (PID 5261) was killed when the machine slept/restarted at 2026-11-19 12:48; the first 4 groups (s42/s43) were already done. The 5th group, vanilla-s44, first died at step 1197/1200 (killed by the restart, no complete.json); the interrupted log is preserved (train.interrupted.jsonl / train.interrupted.out). The resume driver run_l2_resume.py (setsid, detached, log-only) re-ran it from scratch for 1200 steps (train.py has no resume entry), then ran looped-s44 back-to-back. All 6 groups completed and were committed; the push succeeded (git push rc=0, github.com DNS recovered).
- **Storage**: per the 2026-11-19 rule, each group's intermediate checkpoints were deleted right after its evaluation, keeping only the final step-1200 (each s44 group freed 9.74 GB). An earlier repo-wide trim had already deleted 199 intermediate checkpoints (repo 345G→87G). Note: the intermediate checkpoints' config files remain in git history (they were added at commit time, then deleted from disk); only step-1200 remains on disk.

## 3. Results

| Model | Seed | Final train loss | Validation NLL (r=4) | Train time |
|---|---|---|---|---|
| vanilla | 42 | 4.5122 | 5.2810 | 15.6 min |
| looped | 42 | 4.5031 | 5.2338 | 25.7 min |
| vanilla | 43 | 4.4646 | 5.2785 | 15.8 min |
| looped | 43 | 4.4552 | 5.2353 | 25.9 min |
| vanilla | 44 | 4.5864 | 5.2709 | 18.8 min |
| looped | 44 | 4.5565 | 5.2253 | 28.7 min |

Exact values (evaluation.json, per_r[0]["nll"]): vanilla s42 5.281025649528619, s43 5.278519584745641, s44 5.270879471976946; looped s42 5.233832521461758, s43 5.235324693164913, s44 5.225309218822908.

- **Means**: vanilla 5.2768, looped 5.2315.
- **Paired differences** (looped − vanilla): s42 −0.0472, s43 −0.0432, s44 −0.0456. **Paired mean −0.0453**, SD 0.0020, SE 0.0012, t=−39.0 (df=2), two-sided p=0.00066, 95% paired t interval [−0.0503, −0.0403].
- **Verdict (design §7.2)**: branch **(b), looped does not win**. The paired mean −0.0453 does not reach the −0.05 threshold (short by 0.0047); the direction is consistent 3/3 (all <0), but consistent direction cannot substitute for the threshold. The lower edge of the 95% interval (−0.0503) just touches −0.05; the interval itself does not change the operational conclusion (the threshold is the pre-registered operational decision; statistical uncertainty is reported separately).
- **Final train loss**: vanilla mean 4.521, looped mean 4.505. Both models converged to ~4.5 (neither failed to train).

## 4. Interpretation (what Q3 answered, and what it did not)

1. **Q3's answer is (b): from-scratch looped does not win.** By the frozen rule, the paired mean −0.0453 is short of 0.05, so looped does not win. This is not "looped is worse" — looped beat vanilla on all three seeds by ~0.045 NLL, and the difference is highly significantly non-zero (p=0.00066). The accurate statement is: **the advantage is real and consistent in direction, but its magnitude is below the pre-registered bar for "winning"**.
2. **A near miss by 0.0047.** The −0.05 threshold is a pre-registered operational decision and is not changed after the fact; but it must be honestly reported how close it was (0.0047 short) and that the lower edge of the 95% interval just touches −0.05. A different threshold (e.g., −0.04) would flip the conclusion — which is exactly why thresholds are pre-registered: one cannot pick a favorable threshold after seeing the result.
3. **The "grafting is the main cause" hypothesis is weakened, but not refuted.** The logic of Q3 is: if CPT grafting were the main cause of looped losing, removing the graft (from scratch) should produce a clear win. From scratch only won by 0.045 (below 0.05) — no clear win appeared. So grafting is not the sole, decisive explanation: even without grafting, looped's NLL advantage is small. But note: **this inference is incomplete**, because L1 (the CPT side) has no number (frozen), so no formal Q2×Q3 comparison can be made. Strictly, "the grafting effect" remains unmeasured; one cannot conclude "grafting is not the cause", only that "from scratch alone did not produce a large win".
4. **Per unit of compute, looped looks worse.** L2 is a same-token-count comparison, and looped costs 1.676× the FLOPs per step — the 0.045 NLL advantage was bought with 68% more compute. At equal FLOPs (vanilla running ~1.68× more steps, ~2006 steps), vanilla's NLL would keep dropping and looped's advantage would likely shrink or reverse (the Step 1 lesson: under an equal-FLOPs comparison, the old looped's −0.19 advantage shrank to −0.024). This report does not present that as a measured result (L2 ran no equal-FLOPs control), only as a caveat on the metric: **L2's −0.045 was achieved while looped spent 68% more compute**.
5. **Relation to the earlier looped line**: the old implementation (fixed R, no injection) at 160M×1200 reported looped better by −0.188 in EXP-25 (not statistically confirmed) and −0.024 at equal FLOPs in Step 1. The new from-scratch −0.045 sits between the two — but the baselines differ (EXP-25/Step 1 compared against CPT T0; L2 compares against from-scratch vanilla), so the numbers cannot be directly differenced. The only comparable conclusion is: **at our scale, looped's NLL advantage has consistently been "small, direction-consistent, but not large enough"**, whether with the old or the new implementation, whether CPT or from scratch.

## 5. Problems and limitations

- **Threshold sensitivity**: the conclusion depends on the pre-registered −0.05; it missed by 0.0047. Only three seeds; the lower edge of the 95% interval just touches the threshold. Adding seeds (45/46) could move the mean, but the design states "do not add seeds automatically; adding seed 45/46 needs separate authorization" (following the EXP-22 precedent), so none were added here.
- **L1 has no result, so the joint interpretation is incomplete**: Q2 (CPT, equal FLOPs) was frozen by D3 and has no number; "whether grafting is the cause" cannot be tested with a formal Q2×Q3 comparison. Point 3 of §4 is therefore an incomplete inference, offered only as a pointer.
- **LR not scanned**: the from-scratch lr 3e-4 follows Pythia-160M's own training LR and was not scanned; if that LR is suboptimal for looped (or vanilla), the comparison could shift. Changing the LR requires re-registration (design §11 risk 5).
- **Same token count, not equal FLOPs**: looped spends 68% more compute; the equal-FLOPs conclusion is unknown (would need a T0×2006-style control, which this design did not schedule).
- **Scale gap**: 160M, ~5M tokens; the papers are 1.4B–3.5B, 300B–800B tokens. Conclusions do not extrapolate to large scale.
- **NLL is not the papers' winning metric**: Yang won on in-context function MSE, Huginn on math/code reasoning; PPL is where looping helps least (Huginn Sec 5.2). Looped not winning on NLL does not mean the paper's method fails at large scale; conversely, winning here would not imply winning at large scale.
- **vanilla-s44 was re-run once**: the first attempt died at 1197/1200 (machine restart); the second ran from scratch with the same config and seed 44 — a clean redo (not a resume). The re-run introduces no bias (same config and data order), but it is recorded here.
- **Evaluation only at r=4**: the main metric is NLL at r=4; the R-scaling curve (r=1/2/8) was not measured, so whether looped's advantage varies with r is unknown (see §6 D-L2-4, cheap to add).

## 6. Follow-up experiment designs (design only — do not run)

Per the user's instruction: when an important finding appears, keep designing follow-up experiments, but think them through and verify them before running. The designs below are written only, not run; each must pass the "question, cheaper alternative, method verification, literature, preregistration" gate before it runs.

- **D-L2-1 (most direct, needs explicit authorization): add seeds 45/46.** Question: the paired mean −0.0453 is only 0.0047 short of −0.05; would two more seeds push the mean past the threshold? Design: vanilla and looped each add seeds 45 and 46 (1200 steps, everything else identical to L2), recompute the paired mean and interval over five seeds. Decision: five-seed mean < −0.05 with consistent direction → re-judge as (a); otherwise keep (b). Cost: 4 runs, ~1.5 h GPU. Note: the design states "adding seeds needs separate authorization" (EXP-22 precedent: do not add seeds 45/46); and adding seeds carries a "picking favorable seeds" risk — the preregistration must state "report the five-seed mean regardless of the outcome" and never report a favorable subset alone.
- **D-L2-2: from-scratch LR scan (single factor).** Question: is lr 3e-4 unfavorable to looped (or vanilla)? Design: seed 42 only, vanilla and looped each run at lr 1e-4 and 1e-3 (1200 steps), compared against L2's 3e-4. Decision: if some lr clearly widens looped's advantage (e.g., >0.08), L2's conclusion is LR-dependent; if the advantage stays ~0.04–0.05 across all three LRs, the conclusion is robust to LR. Cost: 4 runs, ~1.5 h. Note: first think through whether changing the LR changes the fairness of the comparison (both arms use the same lr; this design keeps that).
- **D-L2-3: longer training (2400 steps, seed 42).** Question: does looped's advantage grow or shrink with longer training? Design: vanilla and looped, one run each (seed 42), 2400 steps, everything else as in L2. Decision: if the paired difference at 2400 steps exceeds that at 1200 steps, looped may need longer training to show its advantage; if it shrinks, the advantage is converging. Cost: 2 runs, ~1 h. Note: the FLOPs ratio stays 1.676 at 2400 steps (same-token-count metric unchanged).
- **D-L2-4 (cheapest, do first): R-scaling diagnostic (zero training cost).** Question: does looped's advantage vary with eval r? Design: use the existing L2 checkpoints (both arms, seeds 42/43/44) and evaluate NLL at r=1/2/8 on the validation set (following the D1 R-scaling protocol). Decision: if the advantage vanishes at r=1 and grows at r=8, it comes from "more compute" itself; if it stays ~0.045 across r, it comes from the architecture. Cost: evaluation only, ~10 min. — This diagnostic is the cheapest and should be done first.

## 7. Conclusion

- L2 did not let looped win: from-scratch, injected looped (looped2) beat vanilla on validation NLL on all three seeds (paired differences −0.047/−0.043/−0.046), paired mean −0.0453, p=0.00066, 95% interval [−0.0503, −0.0403]; but by the frozen rule (mean < −0.05 to win), the verdict is **branch (b), looped does not win** — short by 0.0047.
- Removing the CPT graft did not produce a large win: from scratch only won by 0.045, below 0.05. This weakens the "grafting is the main cause" hypothesis, but with no L1 result, no formal Q2×Q3 comparison can be made, so "grafting is not the cause" cannot be concluded.
- Per unit of compute it looks worse: the 0.045 advantage cost 1.68× the compute; at equal FLOPs, vanilla running ~68% more steps would likely shrink or reverse the advantage (Step 1 precedent).
- Next steps: run the zero-cost R-scaling diagnostic first (D-L2-4), then decide whether to add seeds (D-L2-1, needs authorization) or scan the LR (D-L2-2). Do not run any experiment that has not passed the design gate.

---

**Data and reproduction**: configs, training logs, evaluation results and commits for the 6 runs: runs/reproduction/l2-*.json, l2-*-train.out, l2-*-eval.out, l2-*/evaluation.json, l2-*/complete.json; commits bc0e2ae (vanilla-s42), a3f3bbd (looped-s42), 8a3f874 (vanilla-s43), 7f05311 (looped-s43), 4c253d9 (vanilla-s44), a72c3c6 (looped-s44). Evaluation at r=4, s0-seed 0, 977 validation blocks. Push succeeded (git push rc=0). Statistics computed in ~/workspace/l2_stats.py. Interrupted vanilla-s44 log: runs/reproduction/l2-vanilla-s44-train.interrupted.out / train.interrupted.jsonl.
