# EXP-01 — T0/T1 PonderLM-2 tiny-scale baseline

## 1. Method

This is a mechanism check using pretrained EleutherAI/pythia-70m revision `a39f36b100fe8a5377810d56c3f4789b9c53ac42`, rather than from-scratch pretraining. Both runs load identical pretrained tensors through the same unmodified official GPTNeoXForCausalLM class (importlib; vendor commit `fa784bece621b989fb008c59b0fd8d282fa9c73c`). No LLaMA-Factory trainer, DeepSpeed, flash-attn or wandb is needed.

T0: num_latent_thoughts=0, recurrent_model=false, scale_embeds=false. The official forward then selects original-token logits and performs ordinary shifted next-token CE. T1: num_latent_thoughts=1, recurrent_model=true, scale_embeds=true, interpolation=false. Token embeddings are multiplied by sqrt(2.5*d), retaining the official code-only scale convention. The loss uses only thought-position logits to predict the next real token.

T1 uses an independent Python RNG seeded 42 to choose num_jacobi_iterations from {2,3} once per optimizer step, shared across all 16 microbatches. The wrapper sets random_jacobi_iterations=false and overrides config.num_jacobi_iterations, avoiding the upstream GPT-NeoX {2,3,4} discrepancy without modifying vendor. There is an initial token-only thought initialization pass, then the sampled refinement passes, then a final thought-logit pass. Autograd spans all passes; no intermediate supervision or detach is added.

Validation for both modes uses the same official forward, no grad, fixed num_jacobi_iterations=3, seq 256, batch 1, bf16 autocast and FP32 CE. This measures parallel Jacobi prefill NLL for T1. Separately, a shared teacher-forced sequential inference path runs on the first 128 tokens of validation chunks 0..7. It feeds each real token, then (for T1) its output hidden state as a thought, sharing the token's rotary position ID, and retaining the KV cache. Thus sequential inference and parallel Jacobi validation throughput measure different computation.

The official forward does not expose intermediate H^(k). The separate diagnostic manually invokes the official backbone, interleave helper and hidden-extraction helper: H^(0)=f(E), S^(k)=[e(x1),h1^(k-1),...,e(xT),hT^(k-1)], H^(k)=f(S^(k))[:,0::2]. H_seq comes from one-token-at-a-time cached sequential computation. It reports FP32 RMSE differences of mixed-precision-produced states over 128*512 elements, averaged over eight fixed sequence prefixes. It does not fabricate internal forward states.

## 2. What was done

Run base git hash: `039fc7e87c9e74aab4f8eff0ca11ca683febf185` (main, pulled before work). The final harness revision is the commit containing this report; per-source SHA256 digests at training launch are in metadata.json. Seed 42. Python 3.12.3; torch 2.11.0+cu130; transformers 4.46.1; tokenizers 0.20.3; NumPy 1.26.4; pyarrow 18.1.0. The datasets package is not installed/used.

Hardware: WSL2 Ubuntu, NVIDIA GeForce RTX 4060 Laptop GPU, 8,585,216,000 bytes (8188 MiB). Prelaunch desktop memory baseline was 1262 MiB. Model has 70,426,624 trainable parameters. FP32 master weights/gradients/Adam state and bf16 autocast compute; SDPA; no gradient checkpointing. This is mixed precision, not pure-bf16 optimizer state.

Frozen configs: configs/exp01_t0.json and configs/exp01_t1.json. Each uses seq 256, microbatch 1, accumulation 16, 4096 input tokens/step, 600 steps = 2,457,600 input tokens and 2,448,000 scored next-token training targets. AdamW: lr 1e-4, betas (0.9,0.95), weight decay 0.01, linear warmup steps 1..20, cosine decay steps 21..600 to 1e-5, global gradient clipping 1.0. Weight decay applies to all parameters.

Data: WikiText-2 raw train and validation. The specified S3 URL returned XML PermanentRedirect (301 without a usable Location); the regional bucket endpoint returned 403. Used Salesforce/wikitext's same wikitext-2-raw-v1 split from Hugging Face, pinned dataset revision `b08601e04326c79dfdd32d625aee71d232d685c3`, downloaded using curl. This source substitution is a recorded deviation. The unavailable ZIP prevents bytewise equivalence verification. Nonempty Parquet text rows retain newlines; empty rows are restored to a newline. No EOS or other special tokens are added. Source/text/token SHA256 digests and exact URLs are in runs/data-*.json.

Train: 2,409,267 tokens, 9411 nonoverlapping chunks of 256, 51 tail tokens dropped. Validation: 250,139 tokens, 977 chunks, 27 tail tokens dropped, 249,135 scored within-chunk next-token targets. NumPy default_rng(42) chunk permutations are repeated to obtain 9600 examples (slightly over one epoch), saved once in runs/exp01-order.json and loaded by both runs. Chunk boundaries do not overlap and have no carried context.

All WSL work used the assistant VM wrapper, exporting `TUNNEL_PROXY="${HTTPS_PROXY%:*}:3130"` before `~/workspace/bin/tssh wsl '<command>'`. Commands below ran from /home/lenovo_jason/ponderlm2-looped-exp:

```bash
git pull --ff-only
# Initial pyarrow 26.0.1 required NumPy 2; corrected without changing model dependencies:
/home/lenovo_jason/.local/bin/uv pip install --python .venv/bin/python pyarrow==18.1.0
HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 .venv/bin/python -m unittest discover -s tests -v
# Tokenizer was missing despite expected cache; fetched at the exact pinned model revision:
HF_HUB_DISABLE_XET=1 HF_HUB_DOWNLOAD_TIMEOUT=60 OMP_NUM_THREADS=4 timeout 240 .venv/bin/python -c 'from src.data import chunks; print(chunks().shape,chunks("validation").shape)'
nohup bash scripts/run_exp01.sh > runs/exp01-pipeline.log 2>&1 < /dev/null &
# Pipeline exports HF_HUB_OFFLINE=1 and OMP_NUM_THREADS=4, then runs sequentially:
timeout 3650 .venv/bin/python -m src.train --config configs/exp01_t0.json
.venv/bin/python -m src.evaluate --checkpoint runs/EXP-01-T0/checkpoints/step-600 --mode vanilla --split validation
timeout 3650 .venv/bin/python -m src.train --config configs/exp01_t1.json
.venv/bin/python -m src.evaluate --checkpoint runs/EXP-01-T1/checkpoints/step-600 --mode ponder --split validation
.venv/bin/python -m src.jacobi_check --checkpoint runs/EXP-01-T1/checkpoints/step-600
HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 timeout 180 .venv/bin/python -m src.jacobi_check --checkpoint runs/EXP-01-T1/checkpoints/step-600 --precision fp32 --output-dir runs/EXP-01-T1/fp32-diagnostic
.venv/bin/python scripts/summarize_exp01.py
```

Smoke: two optimizer steps per mode, batch 1 x 64 random tokens, finite official loss, logits [1,64,50304], backward/gradient clipping/optimizer successful; unittest passed in 6.082 seconds (reports/exp01-smoke.txt). D1/D2 mask tests are out of scope. Checkpoints and optimizer/RNG state are saved at steps 200, 400, 600 under each run's ignored checkpoints/ directory; small checkpoint-index.jsonl files contain absolute WSL paths. Each step logs loss, iterations, lr, throughput, preclip gradient norm, peak memory and elapsed wall time.

## 3. Results

T1 underperforms T0 at this frozen budget: validation NLL increases from 4.927077 to 6.006961 and PPL from 137.98 to 406.25. The last-20-step mean training loss falls to 4.7942 for T0 and 5.8658 for T1. The runs complete without OOM/nonfinite losses, but this does not demonstrate a thought-related quality benefit.

The requested mixed-precision Jacobi diagnostic drops sharply over the first three rounds, then plateaus around 0.12–0.14 rather than reproducing the paper's near-zero curve. A bounded additional FP32 diagnostic on the same trained checkpoint and sequences gives round-8 r_k=0.000183506 and d_k=0.000199217. This strong precision sensitivity explains much of the observed plateau; it does not establish the exact cause of every remaining residual or reproduce the paper's absolute errors. Both datasets of measurements are retained. The optional diagnostic CLI was added after training; it changes neither training nor the default bf16 diagnostic path.


<!--RESULTS-->
| Measurement | T0 | T1 |
|---|---:|---:|
| Training wall seconds | 242.964 | 905.812 |
| Whole-run train input tok/s | 10115.1 | 2713.1 |
| Peak train allocated MiB | 1392.42 | 1685.39 |
| nll | 4.927077 | 6.006961 |
| ppl | 137.975574 | 406.246846 |
| tokens_s | 27115.938154 | 7119.060528 |
| sequential_nll_8x128 | 4.560739 | 5.897509 |
| sequential_tokens_s | 235.192010 | 89.624143 |
| inference_flops_per_token_estimate | 90832896.000000 | 133300224.000000 |
| kv_cache_bytes_127_tokens | 2340864.000000 | 4681728.000000 |
| Peak validation allocated MiB | 517.25 | 621.27 |

Equal-input-token loss curves (mean over up to last 20 steps):

| Step | Input tokens | T0 loss | T1 loss |
|---:|---:|---:|---:|
| 1 | 4096 | 6.083247 | 11.636139 |
| 20 | 81920 | 6.143621 | 10.276195 |
| 100 | 409600 | 6.781144 | 8.470191 |
| 200 | 819200 | 6.325436 | 7.672082 |
| 400 | 1638400 | 5.652151 | 6.124055 |
| 600 | 2457600 | 4.794231 | 5.865784 |

Approximately matched training FLOPs (paper estimate T1 step = 3+2*num_jacobi_iterations T0 steps; excludes output-head and sequence-dependent corrections):

| T1 step | T0-equivalent step | T0 last-20 loss | T1 last-20 loss |
|---:|---:|---:|---:|
| 20 | 150 | 6.572875 | 10.276195 |
| 40 | 310 | 6.022220 | 10.070837 |
| 60 | 470 | 5.234648 | 9.075374 |

num_jacobi_iterations counts: {2: 306, 3: 294}. Mean=2.490000; estimated total training FLOPs ratio=7.980000. Measured whole-run wall ratio=3.728 (wall time is not FLOPs). Inference FLOPs ratio=1.468. Combined training wall=1148.776s.

Jacobi convergence (mean of eight sequence RMSEs):

| Refinement k | r_k | d_k to sequential |
|---:|---:|---:|
| 1 | 1.81736842 | 0.32385242 |
| 2 | 0.311193429 | 0.149037011 |
| 3 | 0.136963565 | 0.138454218 |
| 4 | 0.128487946 | 0.128616437 |
| 5 | 0.130829365 | 0.132344875 |
| 6 | 0.126813856 | 0.141658156 |
| 7 | 0.128263625 | 0.141007504 |
| 8 | 0.122818545 | 0.130986349 |

Additional FP32 diagnostic (same checkpoint and sequences; training/evaluation unchanged):

| Refinement k | FP32 r_k | FP32 d_k |
|---:|---:|---:|
| 1 | 1.80690223 | 0.243136117 |
| 2 | 0.239833223 | 0.0286851602 |
| 3 | 0.0283047131 | 0.00340336718 |
| 4 | 0.00336719358 | 0.000345153287 |
| 5 | 0.000338720829 | 0.000201508919 |
| 6 | 0.000196074 | 0.000195172832 |
| 7 | 0.000184897784 | 0.000192225577 |
| 8 | 0.000183505808 | 0.000199216776 |
<!--END_RESULTS-->

Inference FLOPs are analytical estimates, not profiler/hardware-counter FLOPs. With factor=1 for T0 or 2 for T1: factor*layers*(8*d^2+4*d*d_ff) + 2*d*vocab + 4*layers*d*factor^2*128. Convention: 2 FLOPs/MAC, mean cached context 128 real tokens (reference seq 256), dense transformer projections/MLP plus causal QK/AV and one vocabulary head; excludes norms, softmax, embedding lookup. The once-per-real-token vocabulary head makes the total ratio smaller than exactly 2. This estimate applies to sequential inference, not multi-pass parallel Jacobi validation. KV bytes count actual cached tensors after 127 real tokens (254 positions for T1). torch memory peaks exclude other processes/desktop.

Jacobi values are saved per sequence in runs/EXP-01-T1/jacobi_convergence.csv; additional FP32 values are in runs/EXP-01-T1/fp32-diagnostic/. Parallel and cached sequential attention have different shapes/kernels; mixed precision can produce a nonzero d_k numerical floor even once successive Jacobi states stop changing. Compare the measured trend to the paper qualitatively rather than claiming its reported absolute errors.

## 4. Problems and limitations

Pretrained-70M CPT is not from-scratch pretraining; WikiText-2 is small, one seed and 600 steps are a mechanism check rather than paper-scale reproduction. No downstream benchmark was added: this task freezes evaluation to WikiText validation, so the broader plan's downstream evaluation remains untested. No EXP-02 or later experiment was launched.

No OOM fallback, sequence reduction, or batch reduction was used. All 1200 logged losses and preclip gradient norms are finite. Large preclip gradient norms and unfavorable metrics remain in the raw logs. A finite smoke test does not prove convergence or benefits from thoughts. Full validation parallel-Jacobi NLL and the small sequential subset NLL use different context and cannot be directly compared as an iteration-only effect. No generation-quality or serving benchmark is claimed.

The S3 source failure required the documented canonical mirror and newline reconstruction. Missing tokenizer files were fetched at the pinned revision. pyarrow 26 was incompatible with NumPy 1.26; pinned 18.1.0 instead. No vendor code or GPT2_Latent_Reasoning files were modified. Training is bounded by an internal 3600-second check and external timeout; no silent step truncation is allowed.

## 5. Next steps

Use the saved T1 checkpoint for EXP-02 only when requested, with exact masks, meaningful mask tests and distribution-shift caveats. More seeds, training tokens and a larger corpus are needed for performance conclusions. The additional FP32 diagnostic already shows a large reduction in the residual; repeat with controlled attention kernels and more sequences before attributing the full mixed-precision plateau to a specific numerical source.
