# EXP-06 — T5: Intermediate supervision on T1

## 1. Method

T5 starts from the same pretrained EleutherAI/pythia-70m revision
a39f36b100fe8a5377810d56c3f4789b9c53ac42 as T1. This is 70M tiny-scale
continued pretraining, not paper-scale reproduction. It retains one thought per
token, scale_embeds=true (sqrt(2.5*d)), interpolation=false, full causal D0
visibility, official backbone/interleave/extraction helpers, and SDPA.
No vendor or GPT2_Latent_Reasoning files were edited.

Exact indexing: the token-only pass creates H^(0). For k=1..K, refinement k runs
the backbone on [e(x_i), H_i^(k-1)] interleaved across the sequence, with repeated
rotary position IDs. Its even/token outputs create H^(k), and its odd/thought
outputs go through embed_out to predict the next real token. These K thought-logit
sets are supervised. Token-only initialization is unsupervised because it has
no thought positions. After refinement K, a separate interleaved pass using
H^(K) produces the final prediction, exactly as T1. That separate pass is evaluated
and monitored but is **not** a K+1 supervised term: the specified objective
averages only the K refinement passes.

L = (1/K) sum_(k=1..K) CE(z_thought^(k)[:, :-1], x[:, 1:]).
All rounds have equal weight. All states remain connected; no detach is added.
Each microbatch backpropagates its averaged loss divided by accumulation=16;
there is exactly one AdamW update after all 16 microbatches per optimizer step.
No per-round optimizer updates, no moving parameters mid-Jacobi, and no separate
backward per supervised round were done. Final-prediction CE is logged without
contributing gradients. This exact indexing supervises the K refinement
predictions; it does not add a separate final-pass term to the specified mean.

Implementation: opt-in intermediate_supervision=true in src/train.py,
sharing src/masked.py with D0 and unmodified official helpers. Existing
T1/T2/T3/T4 defaults retain their former objectives. Primary evaluation uses
the unchanged official src.evaluate path, fixed K=3. New diagnostics return
refinement logits separately and verify final logits against the official path.

Heads use bf16 autocast, with FP32 master weights/Adam state. Round CE explicitly
converts contiguous shifted logits to FP32. An initial implementation copied
EXP-01's bf16-input CE under autocast, but independent FP32 recomputation differed
by 0.000297546 on the tiny test, failing 1e-5. Explicit FP32 conversion fixes this.
Shift, targets and mean reduction match EXP-01; the small training-CE numerical
difference from its implicit autocast call is recorded. Old paths preserve old CE.
Primary validation recomputes explicit FP32 CE identically for both models.

## 2. What was done

Started with git pull --ff-only on clean main at 1a18eb619c07e9e6730049176b63f7f284715131.
The final implementation revision is the commit containing this report; launch
source SHA256 digests are in runs/EXP-06-T5/metadata.json.
Config: configs/exp06_t5.json, copied to the run directory. Seed 42.
Python 3.12.3, torch 2.11.0+cu130; versions:
{"numpy": "1.26.4", "pyarrow": "18.1.0", "tokenizers": "0.20.3", "transformers": "4.46.1"}.
GPU: NVIDIA GeForce RTX 4060 Laptop GPU, 8585216000 bytes;
70426624 trainable parameters. Dataset package is unused.

Seq 256, microbatch 1, accumulation 16, 600 steps = 2,457,600 input tokens /
2,448,000 shifted targets. AdamW lr 1e-4, betas (0.9,0.95), weight decay 0.01 on
all parameters, warmup 20, cosine to 1e-5, clip 1.0. Independent random.Random(42)
samples K from {2,3} once per optimizer step, shared across microbatches.
All 600 logged values match T1 in order: 306 twos, 294 threes, mean K=2.49.
Training wall includes checkpoint saves; model/data loading is excluded.

Same WikiText-2 raw splits, tokenizer and runs/exp01-order.json as EXP-01.
Train 2,409,267 tokens → 9411 chunks, 51 dropped tail tokens. Validation 250,139
tokens → 977 chunks, 27 dropped, 249,135 scored targets. Uses the previously
recorded Salesforce/wikitext mirror at dataset revision
b08601e04326c79dfdd32d625aee71d232d685c3: nonempty rows verbatim, empty rows
newline, no EOS. Original S3 substitution and unavailable ZIP byte-equivalence
verification remain limitations. Data, order, prior EXP-01–05 checkpoints, and
vendor Python hashes were checked unchanged afterward (186 protected
files). Verification is saved in verification.json.

Commands from /home/lenovo_jason/ponderlm2-looped-exp on WSL, reached with
the workspace tssh wrapper and explicit /home/hatch/.ssh/id_ed25519 key:

~~~bash
git pull --ff-only
HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 .venv/bin/python -m unittest discover -s tests -v
nohup bash scripts/run_exp06.sh > runs/exp06-pipeline.log 2>&1 < /dev/null &
# Pipeline exports HF_HUB_OFFLINE=1, OMP_NUM_THREADS=4 and executes:
timeout 3650 .venv/bin/python -m src.train --config configs/exp06_t5.json
.venv/bin/python -m src.evaluate --checkpoint runs/EXP-06-T5/checkpoints/step-600 --mode ponder
.venv/bin/python -m src.evaluate_rounds --checkpoint runs/EXP-06-T5/checkpoints/step-600 --output runs/EXP-06-T5/rounds-T5.json
.venv/bin/python -m src.evaluate_rounds --checkpoint runs/EXP-01-T1/checkpoints/step-600 --output runs/EXP-06-T5/rounds-T1.json
.venv/bin/python scripts/summarize_exp06.py
~~~

All 16 tests (old + new) passed before full training; output:
reports/exp06-tests.txt. New tests: two unified optimizer steps, random
1×64 batches, K=2 then 3, finite loss/components, backward, clipping and AdamW;
independent FP32 shifted CE from the same round logits, mean/components matching
atol=1e-5, rtol=0 for both K values; official final-logit parity atol=1e-6; seeded
600-step schedule. Attribution test uses all six layers, FP32, fixed 1×8 input.
It adds a fixed random tensor to all backbone output states of refinement 2
(third backbone call including initialization), before both vocab head and H^(2)
extraction. Round-1 CE stays exactly unchanged, round-2 CE and averaged loss
change by >1e-5. Later rounds may change and are not constrained by the assertion.
This tests output-state attribution, not individual attention edges.
No tests were run concurrently with full training.

Checkpoints and optimizer/RNG state at steps 200/400/600 remain local, gitignored
under runs/EXP-06-T5/checkpoints/; checkpoint-index.jsonl records paths.
The background pipeline evaluated immediately after training.

## 3. Results

| Metric | T1 (EXP-01) | T5 |
|---|---|---|
| Primary final-prediction validation NLL | 6.006961 | 5.835645 |
| PPL | 406.246846 | 342.285574 |
| Sequential NLL 8×128 | 5.897509 | 5.759782 |
| Parallel targets/s | 7119.060528 | 8762.368802 |
| Sequential targets/s | 89.624143 | 88.649763 |
| Inference FLOPs/output-token estimate | 133300224.000000 | 133300224.000000 |
| KV-cache bytes, 127 real tokens | 4681728.000000 | 4681728.000000 |
| Training wall seconds | 905.811560 | 763.252314 |
| Whole-run input tokens/s | 2713.147091 | 3219.905076 |
| Peak training allocated MiB | 1685.385254 | 1748.990723 |
| Peak primary validation allocated MiB | 621.267578 | 621.267578 |

Primary ΔNLL (T5 − T1) = -0.171315570; T5 has lower
final validation loss at this frozen token budget. This single seed
supports a benefit at this endpoint.
Wall difference = -142.559 s (-15.74%, ratio 0.8426).
These are historical T1/current T5 timings, not synchronized paired overhead.

### Per-round prediction NLL

Both checkpoints were recomputed with the same new diagnostic on the full
977-chunk validation split, seq 256, batch 1, fixed K=3, 249,135 targets.
These are not reused EXP-01 quantities. Rounds exclude initialization and
the separate final pass.

| Refinement k | T1 checkpoint NLL (recomputed) | T5 NLL |
|---|---|---|
| 1 | 6.144837 | 5.841570 |
| 2 | 6.006974 | 5.841433 |
| 3 | 6.007826 | 5.836916 |

T5 refinement NLL decreases monotonically
across k=1,2,3; T1 does not decrease monotonically.
Manual separate-final NLLs: 5.835645400 (T5),
6.006960971 (T1), agreeing with official primary within 1e-6.
First-chunk final logits also match official within 1e-6.
Per-round evaluation costs 35.197 s (T5), 35.891 s (T1),
additional to primary evaluation and excluded from its throughput.

### Training loss curves

EXP-01 convention: last 20 steps ending at each 100-step boundary. T1 loss is
its supervised separate-final-pass CE. T5 mean is its new objective; CE k=K is
the last supervised refinement; separate final CE monitors the T1-compatible
output and retains the original CE call. Different averaged objectives alone
cannot establish faster convergence.

| Step | Input tokens | T1 final loss | T5 mean CE | T5 CE k=K | T5 separate final CE |
|---|---|---|---|---|---|
| 100 | 409600 | 8.470191 | 8.317098 | 8.336108 | 8.334428 |
| 200 | 819200 | 7.672082 | 6.773438 | 6.753981 | 6.744712 |
| 300 | 1228800 | 6.840076 | 6.111191 | 6.109580 | 6.107532 |
| 400 | 1638400 | 6.124055 | 5.927492 | 5.927353 | 5.929100 |
| 500 | 2048000 | 5.889502 | 5.711048 | 5.710715 | 5.717965 |
| 600 | 2457600 | 5.865784 | 5.673470 | 5.671113 | 5.669012 |

Full 100-step block means with per-round components:

| Step block | Mean objective | CE k=1 | CE k=2 | CE k=3 (K=3 only) | K=3 steps | CE k=K | Separate final CE |
|---|---|---|---|---|---|---|---|
| 1–100 | 9.167046 | 9.141890 | 9.185284 | 9.049582 | 44 | 9.184880 | 9.188628 |
| 101–200 | 7.533920 | 7.541097 | 7.530078 | 7.519474 | 43 | 7.527267 | 7.527911 |
| 201–300 | 6.272105 | 6.273999 | 6.269993 | 6.263960 | 45 | 6.270240 | 6.269039 |
| 301–400 | 5.990756 | 5.995132 | 5.987752 | 5.967430 | 54 | 5.987750 | 5.988042 |
| 401–500 | 5.839980 | 5.843476 | 5.837588 | 5.818864 | 53 | 5.836900 | 5.838467 |
| 501–600 | 5.738186 | 5.742139 | 5.735127 | 5.741351 | 55 | 5.735644 | 5.737185 |

k=3 includes only sampled K=3 steps; k=1/2 use all 100 steps. Their data and
conditioning differ, so k=3 training gaps are descriptive; fixed-K validation
provides the matched-input comparison. Read early-round agreement from these
measured gaps rather than assuming it from supervision. Validation was measured
only at step 600, so faster validation convergence/time-to-threshold remains
unestablished. The training curves document optimization progress. T5's T1-compatible
separate-final training CE is lower at 6/6 reported boundaries,
supporting faster training-loss reduction in this run, while validation
convergence speed remains unmeasured. The matched-step mean CE(k=2)-CE(k=1)
is +0.043394 in steps 1–100 and -0.007012 in steps 501–600;
the sign/gap is reported rather than assuming early rounds must catch up.

### Compute accounting

Forward schedule matches T1. T5 adds K head forward matmuls/CE calls per
microbatch on refinement thought states, retaining the separate final head for
monitoring: 23,904 additional head forward matmuls over 9600 microbatches.
With d=512, vocab=50304, seq=256 and 2 FLOPs/MAC, each costs
2*256*512*50304 = 13,186,891,776 FLOPs; added head forward work totals
315,219,461,013,504 FLOPs (~315.22 TFLOPs),
excluding CE/backward. Backward flows through the K supervised heads and
refinement graph; the separate final-pass branch has no objective and receives
no backward traversal. Thus wall-time difference includes changed backward
graph, kernels, Python logging, and hardware load; it is not extra head cost
alone. A wall-time reduction does not mean negative added head work.
Paper-style backbone-only 3+2K=7.98× vanilla is not an exact T5 training FLOPs
count; no profiler-measured training FLOPs are claimed.

Inference unchanged: one head and two cached backbone passes per output token.
Analytical formula matches EXP-01: 2*layers*(8*d*d+4*d*ff) + 2*d*vocab +
4*layers*d*4*128; 2 FLOPs/MAC, mean cached context 128 real tokens, excluding
norms/softmax/embedding lookup. It describes cached sequential inference, not
parallel validation. Sequential subset is chunks 0..7, first 128 tokens
(1016 targets). Peak memory is torch allocated memory, excluding other processes.

## 4. Problems and limitations

Completed 600 steps at seq 256. No OOM, nonfinite loss/gradient, timeout,
recovery, sequence fallback or shortened run. Initial averaging test failed
before training; explicit FP32 logits fixed it and the full suite passed
before launch. The precision difference from old training CE is recorded above.

Single seed, WikiText-2 only, pretrained 70M tiny-scale CPT, no downstream
benchmark; no broad generalization/paper-scale claim. Primary validation and
short sequential subset have different contexts/computation. Timings are
historical and hardware/kernel/load dependent; FLOPs analytical. Full per-round
evaluation adds heads and CE costs. Training k=3 statistics are conditional on
sampled K=3. The separate final prediction remains indirectly trained through
shared weights but is not directly in this objective. Only endpoint validation
was run, so faster validation convergence remains unproven.

## 5. Next steps

Repeat matched T1/T5 with more seeds and synchronized timing; record checkpoint
validation curves for time-to-NLL. Extend corpus and training budget before
mechanism/generalization claims. Inspect FP32 versus bf16 round dynamics; add
a separately prespecified downstream benchmark. A per-round optimizer-update
variant remains a separate experiment and was not run.
