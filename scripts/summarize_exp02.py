import json
import hashlib
from pathlib import Path
rows = [json.loads(Path(f'runs/EXP-02-{c}/evaluation.json').read_text()) for c in ('D0','D1','D2')]
for r in rows:
    r['delta_nll_vs_d0'] = r['nll']-rows[0]['nll']
    r['delta_ppl_vs_d0'] = r['ppl']-rows[0]['ppl']
Path('runs/exp02-summary.json').write_text(json.dumps(rows,indent=2)+'\n')
p=Path('runs/exp02-provenance.json')
v=json.loads(p.read_text())
for name,digest in v['sha256'].items():
    if '/checkpoints/' in name:
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest
v['checkpoint_digest_verified_after']=True
for name in ('src/masks.py','src/diagnose.py','tests/test_masks.py'):
    v['sha256'][name]=hashlib.sha256(Path(name).read_bytes()).hexdigest()
p.write_text(json.dumps(v,indent=2)+'\n')
table='\n'.join(f"| {r['condition']} | {r['nll']:.9f} | {r['ppl']:.6f} | {r['delta_nll_vs_d0']:+.6f} | {r['delta_ppl_vs_d0']:+.6f} | {r['top1_agreement']:.6%} | {r['top1_change_rate']:.6%} | {r['seconds']:.3f} | {r['tokens_s']:.1f} | {r['peak_memory_bytes']/2**20:.2f} |" for r in rows)
seq='\n'.join(f"| {r['condition']} | {r['sequential_nll_8x128']:.9f} | {r['sequential_seconds']:.3f} | {r['sequential_tokens_s']:.2f} |" for r in rows)
report='''# EXP-02 — Diagnostics D0–D2 on the trained T1 checkpoint

## 1. Method

Inference only on `runs/EXP-01-T1/checkpoints/step-600`: no optimizer, retraining, or weight changes. Its SHA256 is recorded and verified unchanged after evaluation in `runs/exp02-provenance.json`. No vendor or EXP-01 training source changes.

Interleaved sequence: `[e(x1),h1,e(x2),h2,...]`; h_i reuses x_i's rotary position ID. Shared definitions in `src/masks.py` return FP32 additive masks `[1,1,L,L]`, L=seq_len*(1+num_thoughts). Rows=query, columns=key, 0 allowed, -inf blocked; cast to input dtype/device at the backbone call.

Exact visibility table for either query in pair i, subject to interleaved causality:

| Condition | x_j, j<i | x_i | h_j, j<i | h_i | Future pairs |
|---|---|---|---|---|---|
| D0 full | allowed | allowed | allowed | causal | blocked |
| D1 no previous h | allowed | allowed | blocked | causal | blocked |
| D2 no previous x | blocked | allowed | allowed | causal | blocked |

Causal means x_i cannot see its following h_i; h_i sees itself. Multiple thoughts within a pair retain ordinary slot causality. D2 blocks direct token-history attention, while allowing indirect history through thoughts, including across backbone layers. Token-only initialization uses num_thoughts=0: D0/D1 causal, D2 diagonal, preventing an initialization bypass.

**Forward path: manual loop**, invoking the unmodified official backbone, interleave helper, and hidden-extraction helper as established in `src/jacobi_check.py`. The official recurrent forward heuristically collapses a 4-D attention mask to 2-D padding, so it cannot reliably apply these interventions. Explicit masks reach `model.gpt_neox(... attention_mask=...)`.

All conditions use initial token-only H^(0), exactly three parallel Jacobi refinements extracting even/token states, then a final interleaved pass selecting odd/thought logits. No adaptive stopping. D0 preserves the official 2-D mask representation and implicit causal SDPA for numerical parity; D1/D2 require explicit 4-D masks. The D0 builder defines the equivalent additive causal pattern for future training. An initial parity test using explicit 4-D D0 showed bf16 kernel-sensitive rounding; restoring the official representation gave bitwise parity. Warm-up manual versus official D0 max logit difference: 0.0.

All three conditions support cached sequential inference: feed x_i then its hidden output h_i, slicing the current query row from the same global mask. D0 retains official implicit cached causality. Sequential inference has no Jacobi schedule, matching EXP-01. The subset is validation chunks 0..7, first 128 tokens, 1016 scored targets.

## 2. What was done

Pulled main with `git pull --ff-only`; base `d22e073bbeb6f6c2dd176ac1679e61e3ce6e37ee`. Final code revision is the commit containing this report. Seed 42. WSL2 Ubuntu; Python 3.12.3, torch 2.11.0+cu130, transformers 4.46.1, NumPy 1.26.4, pyarrow 18.1.0; datasets unused. GPU: NVIDIA GeForce RTX 4060 Laptop, 8,585,216,000 bytes. FP32 weights, bf16 autocast, FP32 CE, SDPA, eval/no-grad.

Same Pythia-70M revision `a39f36b100fe8a5377810d56c3f4789b9c53ac42`, tokenizer and WikiText-2 raw validation data as EXP-01. `runs/data-validation.json` unchanged: 250139 tokens, 977 nonoverlapping 256-token chunks, 27 tail tokens dropped, 249135 within-chunk targets. No EOS or carried context. Uses the EXP-01 pinned Salesforce/wikitext mirror and documented newline reconstruction. Source/data/checkpoint hashes and package versions are in `runs/exp02-provenance.json`. Batch 1; one latent thought per token.

Commands ran in `/home/lenovo_jason/ponderlm2-looped-exp` via the assistant-VM wrapper: export `TUNNEL_PROXY="${HTTPS_PROXY%:*}:3130"`, then `/home/hatch/workspace/bin/tssh wsl -i /home/hatch/.ssh/id_ed25519 '<command>'`.

```bash
git pull --ff-only
HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 .venv/bin/python -m unittest discover -s tests -v
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4
for c in D0 D1 D2; do
  timeout 1800 .venv/bin/python -m src.diagnose --checkpoint runs/EXP-01-T1/checkpoints/step-600 --condition "$c" --split validation > "runs/EXP-02-$c/console.txt" 2>&1 || exit $?
done
```

All four tests pass; full output in `reports/exp02-tests.txt`. Explicit handwritten 4-pair matrices and explicit pair blocks for 8 pairs check every entry. The mask-effect test uses one real backbone transformer layer in FP32: perturbing a blocked previous h (D1) or x (D2) leaves all later query outputs unchanged with atol=1e-6, rtol=0; the same perturbation changes D0 by >1e-5. One layer isolates direct edges: whole-network D2 invariance would incorrectly prohibit allowed x→h→query routes. Finite CE tests use two short synthetic chunks per condition and sequential inference. Existing EXP-01 forward/backward/optimizer tests pass on independent pretrained models, never the trained checkpoint.

An initial measurement pass was superseded after correcting D0 sequential mask representation. Final measurements ran sequentially after tests, without concurrent test GPU work. Synchronization brackets timing; warm-up and model/data loading are excluded. Peak memory measures parallel validation PyTorch allocation including model weights, excluding other processes and desktop. D0 also retains its warm-up official parity logits (about 24.56 MiB), so its higher allocated peak is a diagnostic artifact rather than evidence that full attention requires more memory. Sequential timing is separate. No training FLOPs or optimizer steps apply.

## 3. Results

| Condition | NLL | PPL | Δ NLL | Δ PPL | Top-1 agreement | Change rate | Parallel seconds | Scored tok/s | Peak allocated MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
TABLE

Agreement compares exactly the same 249135 scored positions to D0, excluding each chunk's final unscored token. Per-condition evaluation JSON, console and short result logs, and combined `runs/exp02-summary.json` are committed. D0 prediction JSON is reproducible local scratch emitted by the CLI and omitted from the commit.

| Condition | Sequential NLL 8×128 | Sequential seconds | Scored tok/s |
|---|---:|---:|---:|
SEQUENTIAL

D0 reproduces EXP-01 parallel NLL/PPL exactly: 6.0069609707008125 / 406.24684631659244. Sequential D0 matches its reported six decimals, 5.897509. Throughput differs from EXP-01 because the manual loop avoids the official forward's redundant loss computation and includes prediction collection; this is not a model speed improvement. All full conditions completed without OOM/timeout.

Both interventions worsen full validation loss: D1 adds 0.154211 NLL and changes 50.571377% of predictions; D2 adds 1.261261 NLL and changes 65.361752%. This checkpoint is more sensitive to removing direct history-token access under this intervention.

The loss and prediction changes measure sensitivity of this checkpoint to removal of specified direct attention edges. They do not establish a learned alternative or a general thought benefit. Explicit versus implicit SDPA masks can change bf16 numerical behavior, so changes cannot be attributed exclusively to information removal at arbitrarily small numerical scales.

## 4. Problems and limitations

**Changing the mask at inference time causes distribution shift: the model was trained with full visibility. D0–D2 diagnose dependence in the existing trained model only; they are not evidence for what training-time masking T2/T3 will do.**

Single checkpoint/seed, WikiText-2 only, pretrained 70M tiny-scale CPT, not paper-scale pretraining. No downstream benchmark: EXP-01 added none, and the plan's downstream task set remains unimplemented. No generation-quality, serving or generalization claims. Full parallel validation and the sequential subset use different context/computation; their losses cannot isolate iteration effects. Mixed-precision mask/kernel sensitivity remains a numerical limitation. The mask-effect test proves blocked direct attention, not independence from history routed through permitted thoughts.

No vendor or GPT2_Latent_Reasoning modification, checkpoint change, sequence reduction, or incomplete condition presented as complete. The official tuple-cache deprecation warning is retained in logs.

## 5. Next steps

T2 (trained D1/no previous h) and T3 (trained D2/history via h) should reuse `src/masks.py`, including initialization restrictions, with other T1 training settings fixed. These training-time counterparts are needed to distinguish adaptation from inference distribution shift. They were not launched in EXP-02.
'''
Path('reports/EXP-02-diagnostics-D0-D2.md').write_text(report.replace('TABLE',table).replace('SEQUENTIAL',seq))
print(json.dumps(rows,indent=2))
