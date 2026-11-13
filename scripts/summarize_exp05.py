"""Generate the EXP-05 report from measured artifacts and verify provenance."""
import hashlib
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines()]


def main():
    run = Path("runs/EXP-05-T4")
    meta = read(run/"metadata.json")
    done = read(run/"complete.json")
    e = [read(f"runs/{name}/evaluation.json") for name in ["EXP-01-T0","EXP-01-T1","EXP-05-T4"]]
    logs = [rows(f"runs/{name}/train.jsonl") for name in ["EXP-01-T0","EXP-01-T1","EXP-05-T4"]]
    walls = [read(f"runs/{name}/complete.json")["wall_s"] for name in ["EXP-01-T0","EXP-01-T1","EXP-05-T4"]]
    assert done["steps"] == 600 and len(logs[2]) == 600
    assert [r["step"] for r in logs[2]] == list(range(1,601))
    assert all(math.isfinite(r[k]) for r in logs[2] for k in ["loss","grad_norm"])
    assert [r["step"] for r in rows(run/"checkpoint-index.jsonl")] == [200,400,600]
    assert e[2]["predicted_tokens"] == 249135 and e[2]["sequential_predicted_tokens"] == 1016
    before = read("runs/exp05-preserved-before.json")
    after = {p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in before}
    changed = [p for p in before if before[p]!=after[p]]
    assert changed == ["src/train.py"], changed
    provenance = dict(base_git_hash=meta["git_hash"],preserved_before=before,preserved_after=after,
        changed_existing_files=changed,
        launch_sources_match={p:hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
                              for p,h in meta["source_sha256"].items()},
        t4_checkpoint_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in run.glob("checkpoints/step-*/*") if p.is_file()},
        config_sha256=hashlib.sha256(Path("configs/exp05_t4.json").read_bytes()).hexdigest(),
        vendor_status=subprocess.check_output(["git","-C","vendor/PonderLM-2","status","--porcelain"],text=True))
    assert all(provenance["launch_sources_match"].values()) and not provenance["vendor_status"]
    Path("runs/exp05-provenance.json").write_text(json.dumps(provenance,indent=2)+"\n")
    table = read("runs/exp05-flops-table.json")
    report = """# EXP-05 — T4 simple Looped Transformer control

## 1. Method

T4 continues pretraining from the identical EleutherAI/pythia-70m revision
a39f36b100fe8a5377810d56c3f4789b9c53ac42 used by T0/T1. Zero-based layers 0,1,2
run once, then the entire original block [3,4,5] runs R=3 times, followed by the
final norm and vocabulary head once. The call schedule is
0,1,2,3,4,5,3,4,5,3,4,5. R counts total block passes, including the first.
No new modules or parameters: all calls reuse the original parameter objects.

Top layers preserve a single pretrained lower-layer feature pass while adding
depth near prediction, following the teacher's top-block suggestion. This is a
design rationale, not evidence that top loops outperform other placements.
Top-5 versus a smaller block was decided using FLOPs before training, without
validation-based selection.

Forward path: src/looped.py is a thin wrapper over the original unmodified
official vendor GPTNeoXForCausalLM used by EXP-01. It calls gpt_neox.embed_in,
emb_dropout, the original gpt_neox.layers modules, gpt_neox.final_layer_norm and
model.embed_out directly, using the original _update_causal_mask and rotary_emb
helpers. Installed transformers 4.46.1 GPTNeoXModel.forward and GPTNeoXLayer.forward
were inspected and compared with the official vendor backbone. No layer code
was copied. The top-level official forward cannot express this schedule.
The same looped.forward runs in training and evaluation; sequential evaluation
projects only the final position through the same vocabulary head.

Standard causal attention over real-token positions 0..L-1; normal rotary
positions in every layer call. No thought positions/interleaving/position reuse,
input injection, random state, sandwich norm, auxiliary supervision or learned
exit gate. Ordinary shifted autoregressive next-token CE. SDPA; bf16 autocast
with FP32 master weights/gradients/Adam state and FP32 CE accumulation.

### FLOPs matching

Exact EXP-01 convention: 2 FLOPs/MAC, mean cached context C=128 real tokens;
dense transformer projections/MLP + causal QK/AV + one vocabulary head; excludes
norms, softmax and embedding lookup. d=512, ff=2048, V=50304, L=6.
For block size B and total passes R:

N = 6+B*(R-1)
F_T4 = N*(8*d*d+4*d*ff+4*d*128)+2*d*V

One layer call costs 6,291,456 dense + 262,144 attention = 6,553,600 FLOPs.
One head costs 51,511,296. T0 = 90,832,896.
T1 = 2*6*(8*d*d+4*d*ff)+2*d*V+4*6*d*4*128 = 133,300,224.
T1 has sequence factor 2; T4 has factor 1. The head executes once per output,
so T1/T0 is about 1.468 rather than 2.

| Top block (zero-based) | R | Layer calls | FLOPs/output | Ratio vs T1 | Error |
|---|---:|---:|---:|---:|---:|
"""
    for c in table["candidates"]:
        label = str(c["layers"])+(" **chosen**" if (c["block_size"],c["repetitions"])==(3,3) else "")
        report += f'| {label} | {c["repetitions"]} | {c["layer_calls"]} | {c["flops_per_token"]:,} | {c["ratio_vs_t1"]:.6f} | {100*c["relative_error"]:.3f}% |\n'
    report += """
Chosen B=3,R=3: 130,154,496 FLOPs/output, 0.976401× T1 (2.360% lower),
and about 1.433× T0. B=2,R=4 ties; smaller R breaks the tie.
The teacher-suggested top-5,R=2 costs 123,600,896 (7.276% lower than T1):
within 10%, but farther from the target. The exhaustive considered grid is
B=1..5 (at least one bottom layer remains), R=2..4. Tests assert the closest
selection and save runs/exp05-flops-table.json.

This matched number is analytical arithmetic for an exact cached implementation,
not the executed FLOPs of the measured uncached evaluator. Repeated physical
layers represent different effective depths and require careful cache state
separation/scheduling. To avoid conflating repetition states, T4 exposes no cache.
Sequential NLL and speed use a full-context forward for every prefix 1..127,
as allowed by the task, with a single last-position vocabulary head. No
kv_cache_bytes is reported. This makes the measured sequential implementation
more expensive than the architecture's matched cached arithmetic.
"""
    report += f'Actual prefix-recomputation arithmetic averages {e[2]["measured_prefix_flops_per_output_estimate"]:,.0f} FLOPs/output on the evaluated subset.\n'
    report += """
## 2. What was done

"""
    report += f'Clean main was pulled with git pull --ff-only; base {meta["git_hash"]}. '
    report += 'Final source revision is the commit containing this report. Exact launch source digests are in runs/EXP-05-T4/metadata.json. '
    report += 'Data/order and EXP-01..04 checkpoints are verified byte-identical in runs/exp05-provenance.json. Only src/train.py adds a loop-config branch; original branches remain unchanged. vendor/ and GPT2_Latent_Reasoning were not modified.\n\n'
    report += f'Training started {datetime.fromtimestamp(meta["start_unix"],timezone.utc).isoformat()}, seed 42. Python {meta["python"]}, torch {meta["torch"]}; '
    report += '; '.join(f'{k} {v}' for k,v in meta["versions"].items())+'. datasets is unused. '
    report += f'WSL2 Ubuntu, {meta["gpu"]}, {meta["gpu_total_bytes"]:,} bytes GPU memory; {meta["parameters"]:,} trainable parameters.\n'
    report += """
Config configs/exp05_t4.json, copied into the run directory: seq 256, microbatch
1, accumulation 16, 600 steps = 2,457,600 input tokens / 2,448,000 shifted targets.
AdamW lr 1e-4, betas (0.9,0.95), weight decay 0.01 on all parameters, warmup 20,
cosine decay to 1e-5, global gradient clip 1.0. Same shared src.train entry point
as T0; no gradient checkpointing.

Data: identical WikiText-2 raw splits and pinned Salesforce/wikitext mirror
revision b08601e04326c79dfdd32d625aee71d232d685c3 as EXP-01. Nonempty rows
verbatim, empty rows restored to newline, no EOS. Source/text/token digests in
runs/data-*.json. Train 2,409,267 tokens, 9,411 nonoverlapping 256-token chunks,
51 dropped tail tokens; validation 250,139 tokens, 977 chunks, 27 dropped,
249,135 scored targets. Same unchanged runs/exp01-order.json: 9,600 ordered
microbatches, seed 42. EXP-01's S3 failure/mirror substitution and inability
to verify ZIP byte equivalence carry over.

Commands below ran in /home/lenovo_jason/ponderlm2-looped-exp on WSL via the
assistant VM tssh wsl -i /home/hatch/.ssh/id_ed25519 wrapper; the VM exports
TUNNEL_PROXY from the HTTPS proxy host at port 3130.

FENCEbash
git pull --ff-only
HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 .venv/bin/python -m unittest discover -s tests -v
nohup bash scripts/run_exp05.sh > runs/exp05-pipeline.log 2>&1 < /dev/null &
# Pipeline exports HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4, then:
timeout 3650 .venv/bin/python -m src.train --config configs/exp05_t4.json
.venv/bin/python -m src.evaluate_looped --checkpoint runs/EXP-05-T4/checkpoints/step-600
.venv/bin/python scripts/summarize_exp05.py
FENCE

All 12 old + new tests passed before full training (reports/exp05-tests.txt).
Two random-token optimizer steps, batch 1×64, finite losses, backward, clip
and AdamW succeeded. Object identity/count checks establish weight sharing and
70,426,624 parameters. R=1 matches official vanilla FP32 logits and CE with
atol=1e-5, rtol=0. The candidate test asserts the closest FLOPs choice.
An extra causality check confirms future-token perturbations leave earlier
logits unchanged. Its separate prefix-shape comparison initially failed at an
overly strict 1e-5 absolute tolerance: maximum FP32 shape-related difference was
0.00048828125, relative 4.77e-7, across 12 calls. That additional comparison
now uses atol=1e-3, rtol=1e-6; the required R=1 test remains at atol=1e-5.
Initial output is retained in reports/exp05-tests-initial.txt.

Evaluation immediately followed training: full validation, seq 256, batch 1,
same FP32 summed CE/target count as EXP-01; sequential uses the same first
eight validation chunks, truncated to 128 (1,016 targets). Warmup and model
loading are excluded from synchronized timings. Tokens/s counts scored targets.
Parallel ms/target is amortized throughput, not autoregressive request latency.
Memory is PyTorch peak allocated including weights, excluding desktop/other
processes.

## 3. Results

"""
    report += f'T4 validation NLL − T0 = {e[2]["nll"]-e[0]["nll"]:+.9f}; T4 − T1 = {e[2]["nll"]-e[1]["nll"]:+.9f}. '
    report += ('Simple looping beats vanilla T0' if e[2]["nll"]<e[0]["nll"] else 'Simple looping does not beat vanilla T0')
    report += ' at this approximately T1-matched analytical inference budget. '
    report += ('It scores better than T1' if e[2]["nll"]<e[1]["nll"] else 'It scores worse than T1')
    report += ' in this single-seed WikiText run; this does not establish general superiority.\n\n'
    report += '| Measurement | T0 | T1 | T4 |\n|---|---:|---:|---:|\n'
    for label,key in [("Validation NLL","nll"),("Validation PPL","ppl"),
                      ("Sequential NLL 8×128","sequential_nll_8x128"),("Parallel scored tok/s","tokens_s"),
                      ("Sequential scored tok/s","sequential_tokens_s"),
                      ("Analytical cached FLOPs/output","inference_flops_per_token_estimate")]:
        report += '| '+label+' | '+' | '.join(f'{v[key]:.6f}' for v in e)+' |\n'
    for label,values in [
        ("Parallel amortized ms/target",[1000/v["tokens_s"] for v in e]),
        ("Sequential ms/target",[1000/v["sequential_tokens_s"] for v in e]),
        ("Training wall seconds",walls),
        ("Whole-run train input tok/s",[2457600/w for w in walls]),
        ("Peak train allocated MiB",[max(r["peak_memory_bytes"] for r in rs)/2**20 for rs in logs]),
        ("Peak parallel validation MiB",[v["peak_memory_bytes"]/2**20 for v in e])]:
        report += '| '+label+' | '+' | '.join(f'{v:.6f}' for v in values)+' |\n'
    report += '| Sequential implementation | Real KV cache | Real KV cache | Full-prefix recomputation |\n'
    report += '| KV-cache bytes after 127 tokens | 2340864 | 4681728 | Not measured / no cache |\n\n'
    report += f'T4 sequential peak allocated memory: {e[2]["sequential_peak_memory_bytes"]/2**20:.3f} MiB. '
    report += 'T0/T1 are original EXP-01 measurements, not reruns. System load/thermals/kernels can vary. T1 parallel evaluation is fixed-K=3 Jacobi; sequential uses cached thought generation. T4 sequential tok/s includes recomputation and cannot establish a cached serving advantage/disadvantage.\n'
    report += """
Loss curve: mean over up to the last 20 steps at each 100-step boundary,
plus steps 1/20, the EXP-01/EXP-03 table convention. Raw per-step losses remain.

| Step | Input tokens | T0 loss | T1 loss | T4 loss |
|---:|---:|---:|---:|---:|
"""
    curve=[]
    for step in [1,20,100,200,300,400,500,600]:
        values=[sum(r["loss"] for r in rs[max(0,step-20):step])/min(step,20) for rs in logs]
        curve.append(dict(step=step,input_tokens=step*4096,t0=values[0],t1=values[1],t4=values[2]))
        report += f'| {step} | {step*4096} | '+' | '.join(f'{v:.6f}' for v in values)+' |\n'
    (run/"loss-curve.json").write_text(json.dumps(curve,indent=2)+"\n")
    report += f'\nAll 600 losses and preclip norms were finite; max preclip norm {max(r["grad_norm"] for r in logs[2]):.6f}, clipped at 1.0 each step. '
    report += """
No OOM, seq-128 fallback, timeout, restart or step reduction occurred. Local
checkpoints at 200/400/600 include model plus optimizer/RNG state under
runs/EXP-05-T4/checkpoints/, gitignored; checkpoint-index.jsonl records paths.
Training wall includes saves. Shorter training than T1 is expected from one
12-layer schedule versus multiple lengthened Jacobi passes. Matching inference
FLOPs does not match training FLOPs.

## 4. Problems and limitations

Single seed, WikiText-2 only, pretrained 70M tiny-scale CPT, 600 steps, no
downstream benchmark; this is not paper-scale pretraining. Fixed training-token
budget is not a training-compute match. Identical starting tensors do not give
an identical initial function once layers repeat, which can cause large early
losses/gradients. No other loop structure was trained; selection was FLOPs-only.

FLOPs are analytical estimates, not profiler measurements, and omit norms/softmax.
The architecture is within 2.36% of T1 under the specified cached arithmetic,
but our sequential implementation recomputes full prefixes. No T4 cache memory
or cached throughput is claimed. Mixed precision and different attention/GEMM
shapes can cause numerical differences; full validation and sequential subset
also use different contexts. Throughput depends on hardware/load.

No Huginn-style input injection, random state, sandwich norm or learned exit:
this is a simple-loop control, not a full Looped Transformer system. It does
not establish the general superiority of thoughts or loops.

## 5. Next steps

Repeat frozen T0/T1/T4 comparisons with more seeds and larger data/models.
Implement and test exact per-effective-depth loop caches before comparing
serving throughput/memory at matched inference arithmetic. Predefine a
downstream benchmark and separately compare training-compute-matched budgets.
Other block placements should be a new experiment, not selected using this
validation result.
"""
    Path("reports/EXP-05-T4-looped.md").write_text("\n".join(line.rstrip() for line in report.replace("FENCE",chr(96)*3).splitlines())+"\n")


if __name__ == "__main__":
    main()
