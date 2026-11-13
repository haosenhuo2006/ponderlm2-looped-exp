"""Build measured result tables. Run from repository root."""
import json
from pathlib import Path
R=Path("runs")
def read(p): return json.loads(p.read_text())
logs={m:[json.loads(x) for x in (R/f"EXP-01-{m}/train.jsonl").read_text().splitlines()] for m in ["T0","T1"]}
ev={m:read(R/f"EXP-01-{m}/evaluation.json") for m in logs}
done={m:read(R/f"EXP-01-{m}/complete.json") for m in logs}
assert all(len(v)==600 for v in logs.values())
out=["| Measurement | T0 | T1 |","|---|---:|---:|"]
def row(label,values): out.append("| "+label+" | "+" | ".join(values)+" |")
row("Training wall seconds",[f"{done[m]['wall_s']:.3f}" for m in logs])
row("Whole-run train input tok/s",[f"{2457600/done[m]['wall_s']:.1f}" for m in logs])
row("Peak train allocated MiB",[f"{max(x['peak_memory_bytes'] for x in logs[m])/2**20:.2f}" for m in logs])
for key in ["nll","ppl","tokens_s","sequential_nll_8x128","sequential_tokens_s","inference_flops_per_token_estimate","kv_cache_bytes_127_tokens"]:
    row(key,[f"{ev[m][key]:.6f}" for m in logs])
row("Peak validation allocated MiB",[f"{ev[m]['peak_memory_bytes']/2**20:.2f}" for m in logs])
out+=["","Equal-input-token loss curves (mean over up to last 20 steps):","","| Step | Input tokens | T0 loss | T1 loss |","|---:|---:|---:|---:|"]
def avg(m,end): return sum(x["loss"] for x in logs[m][max(0,end-20):end])/min(20,end)
for step in [1,20,100,200,400,600]:
    out.append(f"| {step} | {step*4096} | {avg('T0',step):.6f} | {avg('T1',step):.6f} |")
out+=["","Approximately matched training FLOPs (paper estimate T1 step = 3+2*num_jacobi_iterations T0 steps; excludes output-head and sequence-dependent corrections):","","| T1 step | T0-equivalent step | T0 last-20 loss | T1 last-20 loss |","|---:|---:|---:|---:|"]
for step in [20,40,60]:
    eq=sum(3+2*x["K"] for x in logs["T1"][:step])
    out.append(f"| {step} | {eq} | {avg('T0',eq):.6f} | {avg('T1',step):.6f} |")
ks={k:sum(x["K"]==k for x in logs["T1"]) for k in [2,3]}
ratio=sum(3+2*x["K"] for x in logs["T1"])/600
out+=["",f"num_jacobi_iterations counts: {ks}. Mean={sum(x['K'] for x in logs['T1'])/600:.6f}; estimated total training FLOPs ratio={ratio:.6f}. Measured whole-run wall ratio={done['T1']['wall_s']/done['T0']['wall_s']:.3f} (wall time is not FLOPs). Inference FLOPs ratio={ev['T1']['inference_flops_per_token_estimate']/ev['T0']['inference_flops_per_token_estimate']:.3f}. Combined training wall={sum(x['wall_s'] for x in done.values()):.3f}s.","","Jacobi convergence (mean of eight sequence RMSEs):","","| Refinement k | r_k | d_k to sequential |","|---:|---:|---:|"]
for x in read(R/"EXP-01-T1/jacobi_summary.json"):
    out.append(f"| {x['k']} | {x['r_k']:.9g} | {x['d_k']:.9g} |")
fp32=R/"EXP-01-T1/fp32-diagnostic/jacobi_summary.json"
if fp32.exists():
    out += ["", "Additional FP32 diagnostic (same checkpoint and sequences; training/evaluation unchanged):", "", "| Refinement k | FP32 r_k | FP32 d_k |", "|---:|---:|---:|"]
    for x in read(fp32):
        out.append(f"| {x['k']} | {x['r_k']:.9g} | {x['d_k']:.9g} |")
Path("reports/exp01-result-tables.md").write_text("\n".join(out)+"\n")

p=Path("reports/EXP-01-baseline-reproduction.md")
s=p.read_text()
prefix,rest=s.split("<!--RESULTS-->",1)
_,suffix=rest.split("<!--END_RESULTS-->",1)
p.write_text(prefix+"<!--RESULTS-->\n"+"\n".join(out)+"\n<!--END_RESULTS-->"+suffix)
