"""T6: recompute prefixes at current parameters, truncate, update each round."""
import time
import torch
import torch.nn.functional as F
from .ponder import embeddings, forward
from .diagnose import backbone, interleaved

def round_forward(model, x, k):
    if k not in (1,2,3):
        raise ValueError("round must be 1..3")
    # Prefix has no gradient; detach also guards graph-bearing injected states.
    with torch.no_grad():
        e = embeddings(model,x)
        pos = torch.arange(x.shape[1],device=x.device)[None].expand(x.shape[0],-1)
        out = backbone(model,e,pos,torch.ones_like(pos))
        h = model.extract_hidden_for_computation(out,slice(0,None,1),model.config).detach()
        for _ in range(k-1):
            out = interleaved(model,e,h,"D0")
            h = model.extract_hidden_for_computation(out,slice(0,None,2),model.config).detach()
    # Recreate token embeddings so this round trains the embedding parameters.
    e = embeddings(model,x)
    out = interleaved(model,e,h.detach(),"D0")
    z = model.embed_out(out.last_hidden_state[:,1::2])
    return F.cross_entropy(z[:,:-1].contiguous().float().view(-1,z.shape[-1]),
                           x[:,1:].contiguous().view(-1))

def update_batch(model, batches, k, optimizer, clip, deadline=None):
    if k not in (2,3):
        raise ValueError("K must be 2 or 3")
    def budget():
        if deadline is not None and time.perf_counter()>deadline:
            raise RuntimeError("60-minute run budget exceeded; refusing to shorten training silently")
    ces, norms = [], []
    for j in range(1,k+1):
        optimizer.zero_grad(set_to_none=True)
        total = 0.
        for x in batches:
            budget()
            with torch.autocast("cuda",dtype=torch.bfloat16):
                loss = round_forward(model,x,j)
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite per-round loss")
            (loss/len(batches)).backward()
            total += loss.detach().item()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(),clip,error_if_nonfinite=True)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        ces.append(total/len(batches))
        norms.append(float(norm))
    final = 0.
    with torch.no_grad():
        for x in batches:
            budget()
            loss = forward(model,x,k).loss
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite separate final loss")
            final += loss.item()/len(batches)
    return dict(loss=sum(ces)/k,round_ce=ces,round_grad_norm=norms,
                grad_norm=max(norms),final_prediction_ce=final,optimizer_updates=k)
