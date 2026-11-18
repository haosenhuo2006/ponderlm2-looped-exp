"""Unmodified official model, with explicit experiment flags."""
import hashlib
import importlib.util
import os
from pathlib import Path
import torch
from transformers import GPTNeoXConfig
ROOT = Path(__file__).resolve().parents[1]
# Default to 70M; EXP-14 selects 160M through environment variables.
# src/data.py shares these constants for its tokenizer, so it switches too.
MODEL = os.environ.get("PYTHIA_MODEL", "EleutherAI/pythia-70m")
REVISION = os.environ.get("PYTHIA_REVISION", "a39f36b100fe8a5377810d56c3f4789b9c53ac42")
VENDORED_PATH = "vendor/PonderLM-2/src/llamafactory/model/modeling/modeling_gpt_neox.py"

def provenance(checkpoint=None):
    if checkpoint is None:
        weight_source = f"hub:{MODEL}@{REVISION}"
    else:
        weight_source = f"checkpoint:{Path(str(checkpoint)).resolve()}"
    return dict(
        model=MODEL,
        revision=REVISION,
        weight_source=weight_source,
        env_model_override="PYTHIA_MODEL" in os.environ,
        env_revision_override="PYTHIA_REVISION" in os.environ,
        vendored_path=VENDORED_PATH,
        vendored_sha256=hashlib.sha256((ROOT / VENDORED_PATH).read_bytes()).hexdigest(),
    )

def load_model(mode, checkpoint=None):
    spec = importlib.util.spec_from_file_location("official_gpt_neox", ROOT / VENDORED_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = str(checkpoint) if checkpoint else MODEL
    kw = {} if checkpoint else {"revision": REVISION}
    c = GPTNeoXConfig.from_pretrained(source, **kw)
    for k, v in dict(num_latent_thoughts=int(mode == "ponder"), num_jacobi_iterations=3,
                     random_jacobi_iterations=False, scale_embeds=mode == "ponder",
                     interpolation=False, recurrent_model=mode == "ponder", use_cache=False).items():
        setattr(c, k, v)
    # FP32 master weights / optimizer state, bf16 autocast compute.
    model = module.GPTNeoXForCausalLM.from_pretrained(source, config=c, attn_implementation="sdpa", **kw).cuda()
    model.load_provenance = provenance(checkpoint)
    return model

def forward(model, x, iterations=3):
    model.config.num_jacobi_iterations = iterations
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return model(input_ids=x, labels=x, use_cache=False)

def embeddings(model, x):
    e = model.gpt_neox.embed_in(x)
    if model.config.scale_embeds:
        e = e * torch.sqrt(torch.tensor(2.5 * e.shape[-1], device=e.device, dtype=e.dtype))
    return e

def hidden(model, e, positions, cache=None, use_cache=False):
    return model.gpt_neox(inputs_embeds=e, position_ids=positions,
                          past_key_values=cache, use_cache=use_cache, return_dict=True)
