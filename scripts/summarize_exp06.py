"""Build EXP-06 tables and verify the completed experiment."""
import hashlib
import json
import math
import random
from pathlib import Path
from statistics import mean
D = Path("runs/EXP-06-T5")
def read(p):
    return json.loads(Path(p).read_text())
def rows(p):
    return [json.loads(s) for s in Path(p).read_text().splitlines()]
def table(headers, values):
    return "\n".join(["| "+" | ".join(map(str,headers))+" |",
                      "|"+"|".join("---" for _ in headers)+"|"]+
                     ["| "+" | ".join(map(str,row))+" |" for row in values])
def f(x):
    return f"{x:.6f}"
def main():
    t,b = rows(D/"train.jsonl"), rows("runs/EXP-01-T1/train.jsonl")
    c,bc = read(D/"complete.json"),read("runs/EXP-01-T1/complete.json")
    e,be = read(D/"evaluation.json"),read("runs/EXP-01-T1/evaluation.json")
    r,br = read(D/"rounds-T5.json"),read(D/"rounds-T1.json")
    meta,prov = read(D/"metadata.json"),read(D/"provenance.json")
    rng=random.Random(42)
    expected=[rng.choice([2,3]) for _ in range(600)]
    assert len(t)==c["steps"]==600
    assert [v["step"] for v in t]==list(range(1,601))
    assert [v["K"] for v in t]==expected==[v["K"] for v in b]
    assert set(expected)=={2,3}
    for v in t:
        assert len(v["round_ce"])==v["K"]
        assert all(math.isfinite(x) for x in [v["loss"],v["grad_norm"],v["final_prediction_ce"],*v["round_ce"]])
        assert abs(v["loss"]-mean(v["round_ce"]))<1e-5
    protected={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in prov["protected_sha256"]}
    assert protected==prov["protected_sha256"],"Protected files changed"
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in meta["source_sha256"].items())
    assert abs(e["nll"]-r["final_prediction_nll"])<1e-6
    assert abs(be["nll"]-br["final_prediction_nll"])<1e-6
    assert (e["chunks"],e["predicted_tokens"])==(977,249135)
    checks=dict(steps=600,K_counts={str(k):expected.count(k) for k in (2,3)},K_matches_T1=True,
                finite_losses_and_gradients=True,averaged_logged_CE_matches=True,
                protected_files_unchanged=len(protected),training_sources_unchanged=True,
                final_nll_manual_official_difference=abs(e["nll"]-r["final_prediction_nll"]))
    (D/"verification.json").write_text(json.dumps(checks,indent=2)+"\n")
    values=[]
    for key,label in [("nll","Primary final-prediction validation NLL"),("ppl","PPL"),
        ("sequential_nll_8x128","Sequential NLL 8×128"),("tokens_s","Parallel targets/s"),
        ("sequential_tokens_s","Sequential targets/s"),
        ("inference_flops_per_token_estimate","Inference FLOPs/output-token estimate"),
        ("kv_cache_bytes_127_tokens","KV-cache bytes, 127 real tokens")]:
        values.append([label,f(be[key]),f(e[key])])
    values += [["Training wall seconds",f(bc["wall_s"]),f(c["wall_s"])],
               ["Whole-run input tokens/s",f(2457600/bc["wall_s"]),f(2457600/c["wall_s"])],
               ["Peak training allocated MiB",f(max(v["peak_memory_bytes"] for v in b)/2**20),f(max(v["peak_memory_bytes"] for v in t)/2**20)],
               ["Peak primary validation allocated MiB",f(be["peak_memory_bytes"]/2**20),f(e["peak_memory_bytes"]/2**20)]]
    results=table(["Metric","T1 (EXP-01)","T5"],values)
    rounds=table(["Refinement k","T1 checkpoint NLL (recomputed)","T5 NLL"],
                 [[k+1,f(br["round_nll"][k]),f(r["round_nll"][k])] for k in range(3)])
    curves,components=[],[]
    for end in range(100,601,100):
        last=t[end-20:end]
        curves.append([end,end*4096,f(mean(v["loss"] for v in b[end-20:end])),
            f(mean(v["loss"] for v in last)),f(mean(v["round_ce"][-1] for v in last)),
            f(mean(v["final_prediction_ce"] for v in last))])
        block=t[end-100:end]
        components.append([f"{end-99}–{end}",f(mean(v["loss"] for v in block)),
            *[f(mean(v["round_ce"][k] for v in block if len(v["round_ce"])>k)) for k in range(3)],
            sum(v["K"]==3 for v in block),f(mean(v["round_ce"][-1] for v in block)),
            f(mean(v["final_prediction_ce"] for v in block))])
    curve_table=table(["Step","Input tokens","T1 final loss","T5 mean CE","T5 CE k=K","T5 separate final CE"],curves)
    component_table=table(["Step block","Mean objective","CE k=1","CE k=2","CE k=3 (K=3 only)","K=3 steps","CE k=K","Separate final CE"],components)
    delta=e["nll"]-be["nll"]
    cost=c["wall_s"]-bc["wall_s"]
    ratio=c["wall_s"]/bc["wall_s"]
    improving=all(a>b for a,b in zip(r["round_nll"],r["round_nll"][1:]))
    baseline_improving=all(a>b for a,b in zip(br["round_nll"],br["round_nll"][1:]))
    lower_blocks=sum(mean(v["final_prediction_ce"] for v in t[end-20:end]) <
                     mean(v["loss"] for v in b[end-20:end]) for end in range(100,601,100))
    first_gap=mean(v["round_ce"][1]-v["round_ce"][0] for v in t[:100])
    last_gap=mean(v["round_ce"][1]-v["round_ce"][0] for v in t[500:])
    text=f"""# EXP-06 — T5: Intermediate supervision on T1

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

Started with git pull --ff-only on clean main at {meta["git_hash"]}.
The final implementation revision is the commit containing this report; launch
source SHA256 digests are in runs/EXP-06-T5/metadata.json.
Config: configs/exp06_t5.json, copied to the run directory. Seed 42.
Python {meta["python"]}, torch {meta["torch"]}; versions:
{json.dumps(meta["versions"],sort_keys=True)}.
GPU: {meta["gpu"]}, {meta["gpu_total_bytes"]} bytes;
{meta["parameters"]} trainable parameters. Dataset package is unused.

Seq 256, microbatch 1, accumulation 16, 600 steps = 2,457,600 input tokens /
2,448,000 shifted targets. AdamW lr 1e-4, betas (0.9,0.95), weight decay 0.01 on
all parameters, warmup 20, cosine to 1e-5, clip 1.0. Independent random.Random(42)
samples K from {{2,3}} once per optimizer step, shared across microbatches.
All 600 logged values match T1 in order: 306 twos, 294 threes, mean K=2.49.
Training wall includes checkpoint saves; model/data loading is excluded.

Same WikiText-2 raw splits, tokenizer and runs/exp01-order.json as EXP-01.
Train 2,409,267 tokens → 9411 chunks, 51 dropped tail tokens. Validation 250,139
tokens → 977 chunks, 27 dropped, 249,135 scored targets. Uses the previously
recorded Salesforce/wikitext mirror at dataset revision
b08601e04326c79dfdd32d625aee71d232d685c3: nonempty rows verbatim, empty rows
newline, no EOS. Original S3 substitution and unavailable ZIP byte-equivalence
verification remain limitations. Data, order, prior EXP-01–05 checkpoints, and
vendor Python hashes were checked unchanged afterward ({len(protected)} protected
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

{results}

Primary ΔNLL (T5 − T1) = {delta:+.9f}; T5 has {"lower" if delta<0 else "higher"}
final validation loss at this frozen token budget. This single seed
{"supports a benefit at this endpoint" if delta<0 else "does not show a final-loss benefit"}.
Wall difference = {cost:+.3f} s ({(ratio-1)*100:+.2f}%, ratio {ratio:.4f}).
These are historical T1/current T5 timings, not synchronized paired overhead.

### Per-round prediction NLL

Both checkpoints were recomputed with the same new diagnostic on the full
977-chunk validation split, seq 256, batch 1, fixed K=3, 249,135 targets.
These are not reused EXP-01 quantities. Rounds exclude initialization and
the separate final pass.

{rounds}

T5 refinement NLL {"decreases monotonically" if improving else "does not decrease monotonically"}
across k=1,2,3; T1 {"does" if baseline_improving else "does not"} decrease monotonically.
Manual separate-final NLLs: {r["final_prediction_nll"]:.9f} (T5),
{br["final_prediction_nll"]:.9f} (T1), agreeing with official primary within 1e-6.
First-chunk final logits also match official within 1e-6.
Per-round evaluation costs {r["seconds"]:.3f} s (T5), {br["seconds"]:.3f} s (T1),
additional to primary evaluation and excluded from its throughput.

### Training loss curves

EXP-01 convention: last 20 steps ending at each 100-step boundary. T1 loss is
its supervised separate-final-pass CE. T5 mean is its new objective; CE k=K is
the last supervised refinement; separate final CE monitors the T1-compatible
output and retains the original CE call. Different averaged objectives alone
cannot establish faster convergence.

{curve_table}

Full 100-step block means with per-round components:

{component_table}

k=3 includes only sampled K=3 steps; k=1/2 use all 100 steps. Their data and
conditioning differ, so k=3 training gaps are descriptive; fixed-K validation
provides the matched-input comparison. Read early-round agreement from these
measured gaps rather than assuming it from supervision. Validation was measured
only at step 600, so faster validation convergence/time-to-threshold remains
unestablished. The training curves document optimization progress. T5's T1-compatible
separate-final training CE is lower at {lower_blocks}/6 reported boundaries,
supporting faster training-loss reduction in this run, while validation
convergence speed remains unmeasured. The matched-step mean CE(k=2)-CE(k=1)
is {first_gap:+.6f} in steps 1–100 and {last_gap:+.6f} in steps 501–600;
the sign/gap is reported rather than assuming early rounds must catch up.

### Compute accounting

Forward schedule matches T1. T5 adds K head forward matmuls/CE calls per
microbatch on refinement thought states, retaining the separate final head for
monitoring: 23,904 additional head forward matmuls over 9600 microbatches.
With d=512, vocab=50304, seq=256 and 2 FLOPs/MAC, each costs
2*256*512*50304 = {2*256*512*50304:,} FLOPs; added head forward work totals
{23904*2*256*512*50304:,} FLOPs (~{23904*2*256*512*50304/1e12:.2f} TFLOPs),
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
"""
    Path("reports/EXP-06-T5-intermediate-supervision.md").write_text(text)
    print(json.dumps(checks,indent=2))
    print("NLL delta",delta,"wall delta",cost,"wall ratio",ratio)

if __name__=="__main__":
    main()
