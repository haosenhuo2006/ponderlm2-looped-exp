import hashlib
import platform
import importlib.metadata
import argparse
import json
import math
import random
import subprocess
import time
from pathlib import Path
import torch
from .data import chunks, order
from .ponder import load_model, forward
from .masked import forward as masked_forward
from . import looped, looped2, ponder
from .per_round import update_batch

def split_groups(n, k):
    """Partition consecutive microbatch indices, putting larger groups first."""
    if not isinstance(n, int) or not isinstance(k, int) or not 1 <= k <= n:
        raise ValueError("require integer n and k with 1 <= k <= n")
    size, extra = divmod(n, k)
    groups, start = [], 0
    for j in range(k):
        end = start + size + (j < extra)
        groups.append(list(range(start, end)))
        start = end
    return groups


def split_update_batch(model, batches, k, optimizer, clip, deadline=None):
    """T7: final-round T1 objective and full graphs, with K optimizer updates."""
    if k not in (2, 3):
        raise ValueError("K must be 2 or 3")
    groups = split_groups(len(batches), k)
    total, norms = 0., []
    for group in groups:
        optimizer.zero_grad(set_to_none=True)
        for micro in group:
            if deadline is not None and time.perf_counter() > deadline:
                raise RuntimeError("60-minute run budget exceeded; refusing to shorten training silently")
            loss = forward(model, batches[micro], k).loss
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite loss")
            # Keep T1's full-batch denominator, never the group size.
            (loss / len(batches)).backward()
            total += loss.item()
        # Residual difference from T1: clipping occurs per group, not once per data step.
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip, error_if_nonfinite=True)
        norms.append(float(norm))
        optimizer.step()
    return dict(loss=total / len(batches), grad_norm=max(norms), optimizer_updates=k)


def split_supervised_update_batch(model, batches, k, optimizer, clip, deadline=None):
    """T8: T5 mean-round supervision and full graphs, with T7 split updates."""
    if k not in (2, 3):
        raise ValueError("K must be 2 or 3")
    groups = split_groups(len(batches), k)
    total, norms = 0., []
    round_sums, final_sum = [0.] * k, 0.
    for group in groups:
        optimizer.zero_grad(set_to_none=True)
        for micro in group:
            if deadline is not None and time.perf_counter() > deadline:
                raise RuntimeError("60-minute run budget exceeded; refusing to shorten training silently")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = masked_forward(model, batches[micro], k, "D0", intermediate_supervision=True)
                loss = output.loss
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite loss")
            # Full T5 graph, no round detach; retain the full-batch denominator.
            (loss / len(batches)).backward()
            total += loss.detach().item()
            for j, component in enumerate(output.round_losses):
                round_sums[j] += component.detach().item()
            final_sum += output.final_loss.detach().item()
            del output
        # As in T7, clipping is per group rather than once per data step.
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip, error_if_nonfinite=True)
        norms.append(float(norm))
        optimizer.step()
    return dict(loss=total / len(batches), grad_norm=max(norms), optimizer_updates=k,
                round_ce=[v / len(batches) for v in round_sums],
                final_prediction_ce=final_sum / len(batches))


def build_metadata(model):
    metadata = dict(git_hash=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
                    torch=torch.__version__, gpu=torch.cuda.get_device_name(),
                    parameters=sum(p.numel() for p in model.parameters()), start_unix=time.time(),
                    python=platform.python_version(),
                    versions={n:importlib.metadata.version(n) for n in ["transformers","numpy","pyarrow","tokenizers"]},
                    gpu_total_bytes=torch.cuda.get_device_properties(0).total_memory,
                    source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in sorted(Path("src").glob("*.py"))})
    try:
        metadata["model_provenance"] = model.load_provenance
    except AttributeError:
        metadata["model_provenance"] = ponder.provenance(None)
    vendored = ponder.ROOT / ponder.VENDORED_PATH
    metadata["source_sha256"][ponder.VENDORED_PATH] = hashlib.sha256(vendored.read_bytes()).hexdigest()
    return metadata


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--mask", choices=["D1", "D2"])
    a = p.parse_args()
    c = json.loads(Path(a.config).read_text())
    run_budget_s = c.get("run_budget_s", 3600)
    if a.mask is not None:
        c["mask"] = a.mask
    looped2_options = c.get("looped2")
    from_scratch = c.get("from_scratch", False)
    use_looped2 = looped2_options is not None
    track_looped2 = use_looped2 or from_scratch
    if track_looped2 and (c["mode"] != "vanilla" or c.get("loop") is not None
                          or c.get("mask") is not None or c.get("intermediate_supervision")
                          or c.get("split_optimizer_updates") or c.get("per_round_optimizer_update")):
        raise ValueError("looped2 and from_scratch require vanilla mode without loop, mask, "
                         "intermediate supervision or split/per-round updates")
    if c.get("per_round_optimizer_update") and (not c.get("intermediate_supervision") or c["mode"] != "ponder" or c.get("mask") or c.get("loop")):
        raise ValueError("per-round updates require unmasked ponder intermediate supervision")
    if c.get("split_optimizer_updates") and (c["mode"] != "ponder" or c.get("mask") or c.get("loop") or c.get("per_round_optimizer_update")):
        raise ValueError("split updates require unmasked ponder without loop or per-round updates")
    torch.manual_seed(c["seed"])
    rng = random.Random(c["seed"])
    dest = Path(c["run_dir"])
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "config.json").write_text(json.dumps(c, indent=2))
    model = load_model(c["mode"])
    if c.get("loop"):
        looped.configure(model, **c["loop"])
    if track_looped2:
        # D2/D3: register the added norms before reinitializing or making AdamW.
        model_cfg = model.config
        h, ff, vocab = model_cfg.hidden_size, model_cfg.intermediate_size, model_cfg.vocab_size
        emb_scale = math.sqrt(h) if from_scratch else 1.0
        if use_looped2:
            cfg = looped2.resolve_config(model_cfg.num_hidden_layers, looped2_options, emb_scale)
            looped2.apply_config(model, cfg)
            emb_scale = cfg.emb_scale
            widths = (len(cfg.prelude), len(cfg.core), len(cfg.coda))
            l_eff = widths[0] + 4 * widths[1] + widths[2]
            sampler_gen = torch.Generator().manual_seed(c["seed"])
            expected_ratio = looped2.expected_flops_ratio(
                *widths, h, ff, vocab, t=cfg.t, s=cfg.s, sigma=cfg.sigma, cap=cfg.cap)
        else:
            emb_scale = c.get("emb_scale", emb_scale)
            if not math.isfinite(emb_scale):
                raise ValueError("emb_scale must be finite")
            model_cfg.looped2_emb_scale = emb_scale
            l_eff = model_cfg.num_hidden_layers
            widths = (0, l_eff, 0)
            sampler_gen = None
            expected_ratio = 1.0
        if from_scratch:
            looped2.reinitialize_from_scratch(model, l_eff)
        looped2_state = dict(looped2=looped2_options, from_scratch=from_scratch, l_eff=l_eff,
                             emb_scale=emb_scale, cumulative_flops=0, nkr_counts={},
                             sampler_rng=sampler_gen.get_state() if sampler_gen is not None else None)
    data = chunks(seq=c["seq"])
    indices = order(len(data), (max(600,c["steps"]) if c.get("per_round_optimizer_update") else c["steps"]) * c["grad_accum"], c["seed"])
    # A9: retain the config-driven local AdamW and cosine schedule for each arm.
    optimizer = torch.optim.AdamW(model.parameters(), lr=c["lr"], betas=tuple(c["betas"]), weight_decay=c["weight_decay"])
    model.train()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    metadata = build_metadata(model)

    def write_metadata():
        if track_looped2:
            metadata.update(looped2_state, expected_flops_ratio=expected_ratio)
            if sampler_gen is not None:
                metadata["sampler_rng"] = looped2_state["sampler_rng"].tolist()
        (dest / "metadata.json").write_text(json.dumps(metadata, indent=2))

    write_metadata()
    sum_updates = 0
    with (dest / "train.jsonl").open("w", buffering=1) as log:
        for step in range(1, c["steps"]+1):
            if time.perf_counter()-start > run_budget_s:
                raise RuntimeError("60-minute run budget exceeded; refusing to shorten training silently")
            t = time.perf_counter()
            lr = c["lr"] * (step/c["warmup"] if step <= c["warmup"] else
                 0.1+0.9*0.5*(1+math.cos(math.pi*(step-c["warmup"])/(c["steps"]-c["warmup"]))))
            for group in optimizer.param_groups:
                group["lr"] = lr
            k = rng.choice([2,3]) if c["mode"] == "ponder" else 0
            if track_looped2:
                step_nkr = [] if use_looped2 else None
                step_flops = 0
            if c.get("per_round_optimizer_update"):
                batches = [data[indices[(step-1)*c["grad_accum"]+micro]].unsqueeze(0).cuda()
                           for micro in range(c["grad_accum"])]
                result = update_batch(model,batches,k,optimizer,c["clip"],start+run_budget_s)
                loss_sum = result["loss"]*c["grad_accum"]
                norm = result["grad_norm"]
                round_sums = [v*c["grad_accum"] for v in result["round_ce"]]
                final_sum = result["final_prediction_ce"]*c["grad_accum"]
                del batches
            elif c.get("split_optimizer_updates"):
                batches = [data[indices[(step-1)*c["grad_accum"]+micro]].unsqueeze(0).cuda()
                           for micro in range(c["grad_accum"])]
                if c.get("intermediate_supervision"):
                    result = split_supervised_update_batch(model, batches, k, optimizer, c["clip"], start+run_budget_s)
                    round_sums = [v*c["grad_accum"] for v in result["round_ce"]]
                    final_sum = result["final_prediction_ce"]*c["grad_accum"]
                else:
                    result = split_update_batch(model, batches, k, optimizer, c["clip"], start+run_budget_s)
                loss_sum = result["loss"]*c["grad_accum"]
                norm = result["grad_norm"]
                del batches
            else:
                optimizer.zero_grad(set_to_none=True)
                loss_sum = 0.
                round_sums = [0.] * k
                final_sum = 0.
                for micro in range(c["grad_accum"]):
                    x = data[indices[(step-1)*c["grad_accum"]+micro]].unsqueeze(0).cuda()
                    if track_looped2:
                        # D2: draw depth and s0 once per microbatch, in that order.
                        record = {}
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            if use_looped2:
                                loss = looped2.forward(model, x, labels=x, cfg=cfg,
                                                       rng=sampler_gen, record=record).loss
                            else:
                                loss = looped2.plain_forward(model, x, labels=x, emb_scale=emb_scale).loss
                        r, grad_rounds = 1, 1
                        if use_looped2:
                            nkr = record["nkr"]
                            _, grad_rounds, r = nkr
                            step_nkr.append(nkr)
                            key = ",".join(map(str, nkr))
                            counts = looped2_state["nkr_counts"]
                            counts[key] = counts.get(key, 0) + 1
                        step_flops += x.numel() * looped2.train_flops_per_token(
                            r, grad_rounds, *widths, h, ff, vocab)
                    elif c.get("loop"):
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            loss = looped.forward(model, x, labels=x).loss
                    elif c.get("intermediate_supervision"):
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            output = masked_forward(model, x, k, "D0", intermediate_supervision=True)
                            loss = output.loss
                        for j, component in enumerate(output.round_losses):
                            round_sums[j] += component.detach().item()
                        final_sum += output.final_loss.detach().item()
                        del output
                    elif c.get("mask"):
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            loss = masked_forward(model, x, k, c["mask"]).loss
                    else:
                        loss = forward(model, x, k).loss
                    if not torch.isfinite(loss):
                        raise RuntimeError("nonfinite loss")
                    (loss / c["grad_accum"]).backward()
                    loss_sum += loss.item()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), c["clip"], error_if_nonfinite=True)
                optimizer.step()
            torch.cuda.synchronize()
            row = dict(step=step, loss=loss_sum/c["grad_accum"], K=k, lr=lr,
                       tokens=step*c["seq"]*c["grad_accum"], tokens_s=c["seq"]*c["grad_accum"]/(time.perf_counter()-t),
                       grad_norm=float(norm), peak_memory_bytes=torch.cuda.max_memory_allocated(),
                       elapsed_s=time.perf_counter()-start)
            if track_looped2:
                looped2_state["cumulative_flops"] += step_flops
                row.update(nkr=step_nkr, flops=step_flops,
                           cumulative_flops=looped2_state["cumulative_flops"],
                           nkr_counts=looped2_state["nkr_counts"])
                if step == 1:
                    row["emb_scale"] = emb_scale
            if c.get("intermediate_supervision"):
                row.update(round_ce=[v/c["grad_accum"] for v in round_sums],
                           final_prediction_ce=final_sum/c["grad_accum"])
            if c.get("per_round_optimizer_update"):
                row.update(round_grad_norm=result["round_grad_norm"], optimizer_updates=k, cumulative_optimizer_updates=sum_updates+k)
                sum_updates += k
            if c.get("split_optimizer_updates"):
                row.update(optimizer_updates=k, cumulative_optimizer_updates=sum_updates+k)
                sum_updates += k
            log.write(json.dumps(row)+"\n")
            print(json.dumps(row), flush=True)
            if step % 200 == 0 or step == c["steps"]:
                checkpoint = dest / "checkpoints" / ("step-"+str(step))
                model.save_pretrained(checkpoint)
                trainer = dict(optimizer=optimizer.state_dict(), step=step, torch_rng=torch.get_rng_state(),
                               cuda_rng=torch.cuda.get_rng_state(), python_rng=rng.getstate())
                if track_looped2:
                    if sampler_gen is not None:
                        looped2_state["sampler_rng"] = sampler_gen.get_state()
                    trainer.update(looped2_state)
                torch.save(trainer, checkpoint / "trainer.pt")
                if track_looped2:
                    write_metadata()
                with (dest / "checkpoint-index.jsonl").open("a") as index:
                    index.write(json.dumps(dict(step=step,path=str(checkpoint.resolve())))+"\n")
    (dest / "complete.json").write_text(json.dumps(dict(steps=c["steps"],wall_s=time.perf_counter()-start)))
if __name__ == "__main__":
    main()
