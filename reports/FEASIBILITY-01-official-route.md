# FEASIBILITY-01: official PonderLM-2 route

**Recommendation (a): retain the official modeling route with downsizing.**
Start a later, separately budgeted run at microbatch 1, cutoff 256/512, SDPA,
no DeepSpeed, and report_to none. The official modeling code demonstrably
works on this GPU. The full LLaMA-Factory CLI and optimizer/backward stability
remain unvalidated; this is a feasibility probe, not EXP-01 or paper-scale
reproduction. No training or optimizer updates were run.

## Provenance and method

Date: 2026-11-13 UTC. Experiment base commit ba97dd1 (main).
Official source: `fa784bece621b989fb008c59b0fd8d282fa9c73c`; shallow, unchanged, ignored clone.
Plan and official README, quickstart, requirements and modeling code were read
before WSL mutations; setup.py and ZeRO-0 config were also inspected.
All WSL edits and commands used `~/workspace/bin/tssh wsl` with
`export TUNNEL_PROXY="${HTTPS_PROXY%:*}:3130"` set on the assistant VM.
Existing GPT2 repository/environment were not modified.

Hardware: NVIDIA GeForce RTX 4060 Laptop GPU, total 8585216000 bytes
(8188 MiB). Baseline nvidia-smi reported 1534 MiB already used, leaving about
6.5 GiB; allocated peak below excludes other processes/desktop usage.
Software: Python 3.12.3; torch 2.11.0+cu130; transformers 4.46.1;
datasets not installed in probe environment (5.1.0 in existing environment).
Full versions and install commands: ../ENV.md; dependency audit: ../VENDOR.md.
Seed 42. Random valid token IDs, no dataset/tokenizer, labels equal input IDs.
70,426,624 trainable parameters; pretrained Pythia-70M, not scratch training.

Reproducible probe: `scripts/probe_official.py`. Commands: `reports/probe-commands.txt`.

```bash
timeout 300 .venv/bin/python scripts/probe_official.py
# Same checkpoint, bounded transport retry begun while first fetch was pending:
HF_HUB_DISABLE_XET=1 HF_HUB_DOWNLOAD_TIMEOUT=30 timeout 240 .venv/bin/python scripts/probe_official.py
# Isolated cached rerun, after both earlier invocations exited:
HF_HUB_OFFLINE=1 timeout 60 .venv/bin/python scripts/probe_official.py
```

Modeling file is imported directly via importlib.util.spec_from_file_location.
The full training entry point is **not required for this forward**. The exact
upstream GPTNeoXForCausalLM class is used, with SDPA replacing quickstart fa2.
No model code is copied, patched or reimplemented. `from_pretrained` uses bf16;
train mode retains the autograd graph, no checkpointing, no cache, no backward.
num_latent_thoughts=1; num_jacobi_iterations=2; random_jacobi_iterations=false;
scale_embeds=true; interpolation=false; recurrent_model=true.

## Results

First checkpoint fetch completed: **166,029,852 bytes** model.safetensors,
567-byte config; snapshot/download call **274.242354 s**. HF revision
`a39f36b100fe8a5377810d56c3f4789b9c53ac42`. First model load to GPU **0.547670 s**.
The fetch looked stalled but completed before the 300-second timeout; it did
not fail. A standard-HTTP retry had already begun and completed in 8.790040 s,
possibly benefiting from the shared cache/first transfer. Do not interpret
that retry as an independent 166 MB download benchmark. Those two smoke invocations may overlap. To remove that uncertainty, a third
invocation ran alone with HF_HUB_OFFLINE=1 and a 60-second timeout, using the
now-pinned HF revision. The table below uses this isolated rerun: cache lookup
0.002273 s, load 0.517096 s; all memory peaks exactly match the first run.
The first isolated 64-token forward took 0.684455 s. These single forwards
remain diagnostic measurements, not a sustained throughput benchmark.

All five shapes returned finite logits [B,S,50304]. Random-token losses were
16.5, 17.5, 21.75, 18.625 and 24.125; these prove finite computation, not
learning quality. Cold 64-token forward includes CUDA startup; later shapes
are warm, single measurements. Times synchronize CUDA before and after forward.
Peak is torch.cuda.max_memory_allocated, reset for each shape, including model.

| Batch | Cutoff | Forward wall s | Peak allocated MiB | Peak reserved MiB | Estimated training GiB, J=2 to worst J=4 |
|---|---|---|---|---|---|
| 1 | 64 | 0.684455 | 282.9 | 304.0 | 1.71–2.27 |
| 1 | 256 | 0.042351 | 466.0 | 484.0 | 1.93–2.69 |
| 1 | 512 | 0.056297 | 713.9 | 734.0 | 2.23–3.26 |
| 2 | 256 | 0.058545 | 761.8 | 784.0 | 2.29–3.37 |
| 2 | 512 | 0.055481 | 1294.0 | 1306.0 | 2.94–4.59 |

Raw output: probe-output.txt, probe-http-output.txt, probe-isolated-output.txt.
No OOM, import failure or nonfinite result occurred in the official model probe.

## Memory estimate and decision

See memory-method.md for formulas. The B=2 measurements give slope
**1.0396 MiB per original token**, intercept **229.5 MiB**.
A linear memory-efficient-attention extrapolation to batch 8, cutoff 2048
(16384 original tokens; interleaved attention length 4096) gives
**16.86 GiB forward allocation at J=2**, already over the 8 GiB card.
Adding AMP fp32 weights/gradients/Adam state difference **0.918 GiB**,
backward temporary allowance, J=2 through upstream worst-case J=4, and
0.5–1 GiB runtime headroom gives approximately **22.4–41.4 GiB**.
This is an estimate, not an attempted large-shape run; FA2 vs SDPA,
sequence-dependent kernels, LLaMA-Factory AMP/loss casting and optimizer
implementation can change it. No claim that exact CLI peak was measured.
ZeRO stage 0 on one GPU does not shard optimizer state.
Eager attention would be worse: a single B=8,H=8,4096² bf16 attention matrix
is 2 GiB per layer/pass, before softmax and backward intermediates.

The measured small-shape estimates above are well below total memory and
leave room for the observed ~1.5 GiB desktop baseline, especially B=1 at
256/512. B=2,512 has less margin under J=4; start at B=1 and measure a bounded
backward/optimizer check before any full run. Gradient accumulation (e.g. 8–16
as a later budget choice, not tested here) increases effective batch without
multiplying activation peak. Do not blindly match the original eight-GPU
262144-token update (B=1,S=512 would need accumulation 512).

## Errors, deviations and limits

- Existing environment import errors, verbatim, are in existing-import-check.txt:
  `ModuleNotFoundError: No module named 'accelerate'` (also flash_attn,
  deepspeed, wandb, llamafactory, peft, trl absent). Transformers 5.18.0,
  datasets 5.1.0, tokenizers 0.23.2 and NumPy 2.5.3 are outside upstream bounds.
- First torch setup was deliberately terminated after ~11 minutes of slow
  downloads (exit 143), not a package compatibility error. Offline cache check
  failed with `No solution found when resolving dependencies` because
  `torch==2.11.0 needs to be downloaded from a registry` (setup-notes.txt).
  Resumed online cu130 install succeeded. No full training stack was fought.
- `nvidia-smi` was absent from PATH; `/usr/lib/wsl/lib/nvidia-smi` worked.
- The full quickstart CLI, data loading, flash-attn 2.7.2.post1,
  DeepSpeed <=0.14.4 and wandb are not installed/tested. Compatibility with
  torch 2.11/CUDA13 is unknown; README tested torch 2.4/CUDA12.1.
  These are avoided by the plan-permitted SDPA/no-DeepSpeed/no-wandb route.
  CLI registration would require installing the upstream package and its
  declared dependencies; package/CLI success must not be inferred here.
- Upstream GPT-NeoX uses `torch.randint(2,5)` → {2,3,4}, versus the paper/plan
  {2,3}. The smoke disables randomness and fixes two; upstream is unchanged.
  Before EXP-01, explicitly review a minimal {2,3} correction and record it.
- scale_embeds=true uses sqrt(2.5*d), a code-only detail retained in this probe.
  Each real token shares its position ID with the interleaved thought.
- No adaptive stopping, convergence diagnostic, sequential decoding test,
  backward, optimizer step, dataset pass, or model-quality claim is made.
  This probe does not resolve the paper/code inference asymmetry.

Proceed with official modeling, not a new minimal implementation. If the
later full CLI is required and blocked, report that decision gate before
switching routes, as the experiment plan requires.
