"""Validate finished runs and generate reports exclusively from saved measurements."""
import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def logs(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


parser = argparse.ArgumentParser()
parser.add_argument('--only', choices=['T2', 'T3'])
args = parser.parse_args()
baseline = Path('runs/EXP-01-T1')
b = read(baseline/'evaluation.json')
btrain = logs(baseline/'train.jsonl')
bcomplete = read(baseline/'complete.json')
provenance_path = Path('runs/exp0304-provenance.json')
provenance = read(provenance_path)
for name, expected in provenance['protected_sha256'].items():
    actual = hashlib.sha256(Path(name).read_bytes()).hexdigest()
    assert actual == expected, ('protected file changed', name)
provenance['protected_files_verified_unchanged'] = True
provenance_path.write_text(json.dumps(provenance, indent=2)+'\n')
summary = {}
for exp, label, condition, title, filename in [
    ('03','T2','D1','No previous h','EXP-03-T2-no-prev-h.md'),
    ('04','T3','D2','History via h','EXP-04-T3-history-via-h.md'),
]:
    if args.only and label != args.only:
        continue
    dest = Path(f'runs/EXP-{exp}-{label}')
    ev, full = read(dest/'evaluation.json'), read(dest/'evaluation-D0.json')
    rows, complete = logs(dest/'train.jsonl'), read(dest/'complete.json')
    metadata, config = read(dest/'metadata.json'), read(dest/'config.json')
    assert complete['steps'] == 600 and len(rows) == 600
    assert [r['step'] for r in rows] == list(range(1,601))
    assert [r['K'] for r in rows] == [r['K'] for r in btrain]
    assert all(math.isfinite(r[key]) for r in rows for key in ('loss','grad_norm'))
    assert ev['mask'] == condition and full['mask'] == 'D0'
    assert ev['predicted_tokens'] == full['predicted_tokens'] == b['predicted_tokens'] == 249135
    assert config['seq'] == 256 and config['steps'] == 600 and config['grad_accum'] == 16
    checkpoints = logs(dest/'checkpoint-index.jsonl')
    assert [r['step'] for r in checkpoints] == [200,400,600]
    for checkpoint in checkpoints:
        assert (Path(checkpoint['path'])/'model.safetensors').exists()
    counts = dict(sorted(Counter(r['K'] for r in rows).items()))
    peak = max(r['peak_memory_bytes'] for r in rows)/2**20
    delta = ev['nll']-b['nll']
    intervention = read(Path('runs')/f'EXP-02-{condition}'/'evaluation.json')
    summary[label] = dict(primary=ev, d0_diagnostic=full, training=complete,
                          k_counts=counts, peak_training_mib=peak,
                          all_losses_and_gradient_norms_finite=True,
                          k_sequence_identical_to_t1=True)
    comparisons = '\n'.join(
        f"| {name} | {mask} | {e['nll']:.9f} | {e['ppl']:.6f} | {e['sequential_nll_8x128']:.9f} | {e['tokens_s']:.1f} | {e['sequential_tokens_s']:.2f} | {e['peak_memory_bytes']/2**20:.2f} |"
        for name,mask,e in [('T1 baseline','D0',b),(label+' primary',condition,ev),(label+' cross-condition diagnostic','D0',full)])
    curves = []
    for step in (1,20,100,200,300,400,500,600):
        def mean_loss(rs):
            block = rs[max(0,step-20):step]
            return sum(r['loss'] for r in block)/len(block)
        curves.append(f'| {step} | {step*4096} | {mean_loss(btrain):.6f} | {mean_loss(rows):.6f} |')
    restriction = (
        'D1 allows all causal real tokens x_≤i, blocks every previous thought h_<i, and retains the current h_i subject to slot causality. The token-only initialization is causal (D1 with zero thoughts).'
        if condition == 'D1' else
        'D2 blocks direct access to historical real tokens x_<i, keeps current x_i, and allows previous thoughts h_<i. The token-only initialization is diagonal (D2 with zero thoughts), preventing history leakage through initialization. Indirect history through allowed thoughts remains intentional.')
    grad_note = (
        'D1 has a structural degeneracy: token/even queries cannot read their following current h or any previous h. Thus refinement token states do not depend on the input thought states. The graph is never detached, but initialization and early-refinement activation gradients are exactly zero by mask semantics. The final refinement and thought-logit pass receive nonzero gradients. All requested passes still execute; no computation shortcut was introduced.'
        if condition == 'D1' else
        'D2 permits x→h→later-position information paths. All initialization, refinement, and final-pass outputs received finite nonzero gradients in the smoke test; no state is detached. Whole-network independence from historical x would incorrectly forbid these permitted paths.')
    text = f'''# EXP-{exp} — {label}: {title}

## 1. Method

{label} starts from the same pretrained EleutherAI/pythia-70m revision `a39f36b100fe8a5377810d56c3f4789b9c53ac42` as T1, with {condition} applied from the first training forward. {restriction} Definitions come unchanged from `src/masks.py`, including the EXP-02 initialization restriction.

| Query in pair i | Previous x | Current x_i | Previous h | Current h_i | Future |
|---|---|---|---|---|---|---|
| D0 | allow | allow | allow | causal | block |
| D1 | allow | allow | block | causal | block |
| D2 | block | allow | allow | causal | block |

The sequence is `[e(x1),h1,e(x2),h2,...]`; h_i shares x_i's rotary position. Causality means x_i cannot attend to its following h_i; h_i sees itself. Additive masks have shape [1,1,L,L], zero allowed and negative infinity blocked, explicitly passed to the official backbone.

Training uses `src/masked.py`: official backbone + interleave helper + hidden-extraction helper; initialization → sampled K refinements → final thought-logit pass. Autograd spans the whole schedule, with no detach, no intermediate loss, and shifted next-token CE only on thought logits. The official recurrent forward is unsuitable because it heuristically collapses 4-D masks into 2-D padding masks. Vendor is unchanged. Unmasked training keeps the original EXP-01 official forward by default; `--mask D1|D2` or the config selects the new path.

{grad_note}

To match T1 training exactly, CE receives contiguous bf16 logits under autocast, as the official forward does; master weights and optimizer state remain FP32. An explicit FP32-logit conversion initially failed exact CE parity despite identical logits, so the final path reproduces the official CE call. D0 manual logits and loss now match the official forward at atol=1e-6, rtol=0. Evaluation recomputes FP32 CE from returned logits exactly as EXP-01.

Primary comparison: {label}/{condition} versus T1/D0. T1 − T2 estimates the contribution of previous-h information passing when the restriction is present throughout training; T3 tests adaptation to routing history through h while keeping current x_i. Evaluating the same {label} checkpoint under D0 is an extra cross-condition diagnostic, with changed evaluation visibility; it is not the primary comparison.

## 2. What was done

Started with `git pull --ff-only` on main, base `{metadata['git_hash']}`. Final implementation revision is the commit containing this report; exact source SHA256 digests at training launch are saved in `{dest}/metadata.json`. Seed 42. WSL platform: `{provenance['wsl_uname']}`. Python {metadata['python']}, torch {metadata['torch']}, transformers {metadata['versions']['transformers']}, NumPy {metadata['versions']['numpy']}, pyarrow {metadata['versions']['pyarrow']}, tokenizers {metadata['versions']['tokenizers']}; datasets is unused.

GPU: {metadata['gpu']}, {metadata['gpu_total_bytes']:,} bytes; {metadata['parameters']:,} trainable parameters. SDPA; bf16 autocast, FP32 master weights/gradients/Adam state, no gradient checkpointing. One latent thought, scale_embeds=true (sqrt(2.5*d)), interpolation=false.

Frozen config: `configs/exp{exp}_{label.lower()}.json`, copied to `{dest}/config.json`. Seq 256, microbatch 1, gradient accumulation 16, effective 4096 input tokens/step, 600 optimizer steps = 2,457,600 input tokens (2,448,000 shifted targets). AdamW lr 1e-4, betas (0.9,0.95), weight decay 0.01 on all parameters, warmup 20, cosine decay to 1e-5, clip 1.0. An independent Python RNG seeded 42 chooses K from {{2,3}} once per optimizer step, shared across the 16 microbatches. Counts: {counts}; all 600 logged K values match T1 in order.

WikiText-2 raw train/validation are the identical pinned Salesforce/wikitext mirror used in EXP-01 (dataset revision `b08601e04326c79dfdd32d625aee71d232d685c3`), with nonempty text rows verbatim, empty rows restored to newline, no EOS. Original S3-source substitution and inability to verify ZIP byte equivalence carry over from EXP-01. Train: 2,409,267 tokens, 9411 nonoverlapping 256-token chunks, 51 tail tokens dropped. Validation: 250,139 tokens, 977 chunks, 27 dropped, 249,135 scored targets. Same immutable `runs/exp01-order.json` (9600 examples). Exact source, text and token hashes remain in `runs/data-*.json`; data/order and EXP-01 checkpoint hashes are verified unchanged in `runs/exp0304-provenance.json`.

Commands executed from `/home/lenovo_jason/ponderlm2-looped-exp` over the assistant VM's `tssh wsl -i /home/hatch/.ssh/id_ed25519` wrapper with `TUNNEL_PROXY="${{HTTPS_PROXY%:*}}:3130"`:

```bash
git pull --ff-only
HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 .venv/bin/python -m unittest discover -s tests -v
nohup bash scripts/run_exp0304.sh > runs/exp0304-pipeline.log 2>&1 < /dev/null &
# Pipeline exports HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 and runs T2 then T3.
# Immediately after each training, its own-mask evaluation then D0 diagnostic:
timeout 3650 .venv/bin/python -m src.train --config configs/exp{exp}_{label.lower()}.json
.venv/bin/python -m src.evaluate --checkpoint {dest}/checkpoints/step-600 --mode ponder --mask {condition}
.venv/bin/python -m src.evaluate --checkpoint {dest}/checkpoints/step-600 --mode ponder --mask D0
.venv/bin/python scripts/summarize_exp0304.py
```

All 8 tests (old + new) passed before full training; full output: `reports/exp0304-tests.txt`. The final full-suite rerun after all evaluations also passed (8 tests; `reports/exp0304-tests-final.txt`). Two optimizer steps per masked condition, batch 1×64 random tokens, finite loss/gradients, backward + clipping + AdamW successful. Gradient hooks verify graph connectivity in all passes and the expected condition-specific nonzero/zero activation gradients described above. The FP32 loss-effect test exercises the actual training forward, using a single real backbone layer and a perturbation in its final interleaved pass. Later-pair CE is invariant to a blocked h_j (D1) / x_j (D2), atol=1e-6, while the same perturbation changes D0 CE. This isolates direct attention edges; it does not claim D2's whole graph ignores historical x. EXP-02 tests are unchanged. K sampling tests assert 306 twos and 294 threes over 600 steps.

Evaluation uses the adapted EXP-01 `src/evaluate.py`: manual masked parallel forward (initialization + 3 refinements + final thought logits), seq 256, batch 1, full validation, bf16 compute and FP32 CE. Sequential evaluation reuses EXP-02 cached mask-row slicing on the same chunks 0..7, prefixes 128, 1016 targets. The cache-byte counter was corrected during T2 training before its evaluation; evaluation source SHA256 is separately recorded in `runs/exp0304-provenance.json`. D0 manual evaluation retains official implicit causal mask representation for numerical parity, as EXP-02. Warm-up/model loading are excluded from synchronized evaluation timings; training wall includes training and checkpoint saves. Checkpoints at 200/400/600 plus optimizer/RNG state are local under `{dest}/checkpoints/`, gitignored; paths in checkpoint-index.jsonl. Training wall: {complete['wall_s']:.3f} seconds.

## 3. Results

Primary validation ΔNLL ({label} − T1) = {delta:+.9f}; {'lower' if delta < 0 else 'higher'} NLL at this fixed budget. This single-seed outcome is not a general performance conclusion.

For context, EXP-02 inference-only T1/{condition} had NLL {intervention['nll']:.9f} and PPL {intervention['ppl']:.6f}; trained {label}/{condition} now has NLL {ev['nll']:.9f}. This comparison quantifies the outcome after training with the restriction rather than abruptly imposing it on T1. It does not by itself isolate a unique internal information route. Giving this trained checkpoint full D0 visibility changes NLL by {full['nll']-ev['nll']:+.9f}; this is a cross-condition diagnostic, not a replacement primary score.

| Model / purpose | Eval mask | Validation NLL | PPL | Sequential NLL 8×128 | Parallel tok/s | Sequential tok/s | Peak validation MiB |
|---|---|---:|---:|---:|---:|---:|---:|
{comparisons}

| Cost | T1 baseline | {label} |
|---|---:|---:|
| Training wall seconds | {bcomplete['wall_s']:.3f} | {complete['wall_s']:.3f} |
| Whole-run input tok/s | {2457600/bcomplete['wall_s']:.1f} | {2457600/complete['wall_s']:.1f} |
| Peak training allocated MiB | {max(r['peak_memory_bytes'] for r in btrain)/2**20:.2f} | {peak:.2f} |
| Sequential inference FLOPs/token estimate | {b['inference_flops_per_token_estimate']:.0f} | {ev['inference_flops_per_token_estimate']:.0f} |
| KV cache bytes, 127 real tokens | {b['kv_cache_bytes_127_tokens']} | {ev['kv_cache_bytes_127_tokens']} |

Loss curve: mean over the last 20 steps of each 100-step block (initial step/20 also shown), identical input-token budget:

| Step | Input tokens | T1 loss | {label} loss |
|---:|---:|---:|---:|
{chr(10).join(curves)}

K mean = 2.49; paper-style training estimate 3+2K = 7.98× vanilla, same schedule as T1. Actual masked dense SDPA still computes dense tensors; masks do not imply a sparse FLOPs reduction. FLOPs are analytical estimates, not profiler measurements. Sequential inference uses 2 FLOPs/MAC, mean cached context 128 real tokens, transformer dense projections/MLP + causal QK/AV + one vocabulary head; excludes norms, softmax, embedding lookup. Formula matches EXP-01: `2*layers*(8*d*d+4*d*ff) + 2*d*vocab + 4*layers*d*4*128`. This estimate describes cached sequential inference, not multi-pass parallel validation. KV bytes count the actual cached tensors after 127 real tokens (254 positions); no all-bf16 dtype assumption is made. A lightweight hook retains only the last cache during sequential evaluation. Peak memory is PyTorch allocated memory including weights, excluding desktop/other processes.

## 4. Problems and limitations

Completed all 600 steps, with finite logged losses and preclip gradient norms. No OOM, sequence fallback, reduced batch, timeout, or silently omitted condition. EXP-01/02 source behavior and checkpoint files are preserved; no vendor or GPT2_Latent_Reasoning changes. Explicit 4-D masks versus implicit D0 masks can select different bf16 SDPA behavior, leaving a numerical confound. The loss-effect test checks direct edges in an isolated final pass; the multi-layer model may route information through all permitted paths.

Single seed, WikiText-2 only, pretrained 70M tiny-scale CPT, no downstream benchmark. These results do not reproduce paper-scale pretraining or establish broad generalization. The short sequential subset and full parallel validation differ in context/computation and cannot isolate iteration effects. D0 evaluation changes the trained checkpoint's visibility and is a cross-condition diagnostic, not evidence for training under D0. Throughput is hardware/load/kernel dependent, and raw timing differences are not a measured FLOPs reduction.

## 5. Next steps

Repeat matched T1/{label} training with more seeds and a larger corpus/token budget before interpreting the sign or magnitude as a mechanism conclusion. Inspect Jacobi convergence and precision sensitivity on these trained checkpoints; add a fixed downstream benchmark in a separately specified experiment. Keep D0 cross-condition diagnostics separate from the primary own-mask comparison.
'''
    Path('reports', filename).write_text(text)
Path('runs/exp0304-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
