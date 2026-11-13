"""Manual diagnostic using official backbone and hidden extraction; no model patch."""
from contextlib import nullcontext
import argparse
import csv
import json
from pathlib import Path
import torch
from .ponder import load_model, embeddings, hidden
from .data import chunks

@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",required=True)
    p.add_argument("--precision",choices=["bf16","fp32"],default="bf16")
    p.add_argument("--output-dir")
    a = p.parse_args()
    model = load_model("ponder",a.checkpoint).eval()
    data = chunks("validation")[:8,:128]
    rows = []
    with (torch.autocast("cuda",dtype=torch.bfloat16) if a.precision == "bf16" else nullcontext()):
        for index,x in enumerate(data):
            x = x.unsqueeze(0).cuda()
            e = embeddings(model,x)
            pos = torch.arange(128,device="cuda").unsqueeze(0)
            # Sequential fixed point: token output is h_i, thought consumes h_i.
            cache = None
            hs = []
            for i in range(128):
                out = hidden(model,e[:,i:i+1],pos[:,i:i+1],cache,True)
                hs.append(out.last_hidden_state)
                out = hidden(model,out.last_hidden_state,pos[:,i:i+1],out.past_key_values,True)
                cache = out.past_key_values
            target = torch.cat(hs,dim=1).float()
            h = hidden(model,e,pos).last_hidden_state
            for k in range(1,9):
                # Exact official interleave and source (token/even) hidden slice.
                inp, pp, mask = model.prepare_gpt_inputs_for_stages(
                    [e,h],1,128,e.shape[-1],e.device,pos,torch.ones_like(pos),None,model.config)
                out = model.gpt_neox(inputs_embeds=inp,position_ids=pp,attention_mask=mask,use_cache=False,return_dict=True)
                new = model.extract_hidden_for_computation(out,slice(0,None,2),model.config)
                rows.append(dict(sequence=index,k=k,r_k=(new.float()-h.float()).square().mean().sqrt().item(),
                                 d_k=(new.float()-target).square().mean().sqrt().item()))
                h = new
    dest = Path(a.output_dir) if a.output_dir else Path(a.checkpoint).parents[1]
    dest.mkdir(parents=True,exist_ok=True)
    with (dest/"jacobi_convergence.csv").open("w") as f:
        w = csv.DictWriter(f,fieldnames=["sequence","k","r_k","d_k"],lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    summary = [dict(k=k,r_k=sum(r["r_k"] for r in rows if r["k"]==k)/8,
                    d_k=sum(r["d_k"] for r in rows if r["k"]==k)/8) for k in range(1,9)]
    (dest/"jacobi_summary.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))
if __name__ == "__main__":
    main()
