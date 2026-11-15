"""Plain weight-shared top-block repetition of the official Pythia backbone.

No cache is exposed: sequential evaluation recomputes each real-token prefix.
Only the base model owns parameters; layer_schedule returns existing modules.
"""
from types import SimpleNamespace
import torch
import torch.nn.functional as F

T1_FLOPS = 133300224
CONVENTION = ("2 FLOPs/MAC; cached context mean 128 real tokens; transformer dense "
              "+ causal QK/AV + one vocab head; excludes norms/softmax")


def flops(block_size, repetitions, d=512, ff=2048, vocab=50304, layers=6, context=128):
    calls = layers + block_size * (repetitions - 1)
    return calls * (8*d*d + 4*d*ff + 4*d*context) + 2*d*vocab


def candidates():
    return [dict(block_size=b, layers=list(range(6-b, 6)), repetitions=r,
                 layer_calls=6+b*(r-1), flops_per_token=flops(b, r),
                 ratio_vs_t1=flops(b, r)/T1_FLOPS,
                 relative_error=abs(flops(b, r)/T1_FLOPS-1))
            for b in range(1, 6) for r in range(2, 5)]


def selected():
    return min(candidates(), key=lambda c: (c["relative_error"], c["repetitions"], c["block_size"]))


def configure(model, block_size, repetitions, start=None):
    if not 1 <= block_size < len(model.gpt_neox.layers) or not 1 <= repetitions <= 4:
        raise ValueError("require a proper top block and R in 1..4")
    if model.config.num_latent_thoughts or model.config.scale_embeds:
        raise ValueError("looped control requires vanilla unscaled real tokens")
    if start is None:
        start = len(model.gpt_neox.layers) - block_size
    if type(start) is not int or not 0 <= start <= len(model.gpt_neox.layers) - block_size:
        raise ValueError("require integer loop block start within layer bounds")
    model.config.loop_block_start = start
    model.config.loop_block_size = block_size
    model.config.loop_repetitions = repetitions
    model.config.use_cache = False
    return model


def layer_schedule(model):
    layers = model.gpt_neox.layers
    block_size = model.config.loop_block_size
    start = getattr(model.config, "loop_block_start", len(layers) - block_size)
    return (tuple(layers[:start]) + tuple(layers[start:start+block_size]) *
            model.config.loop_repetitions + tuple(layers[start+block_size:]))


def forward(model, x, labels=None, last_only=False):
    """Same differentiable forward in training, validation and prefix evaluation."""
    backbone = model.gpt_neox
    e = backbone.embed_in(x)
    cache_position = torch.arange(x.shape[1], device=x.device)
    positions = cache_position[None].expand(x.shape[0], -1)
    # Match the official vanilla forward's all-ones real-token attention mask.
    mask = backbone._update_causal_mask(torch.ones_like(x), e, cache_position, None, False)
    h = backbone.emb_dropout(e)
    rotary = backbone.rotary_emb(h, positions)
    for layer in layer_schedule(model):
        h = layer(h, attention_mask=mask, position_ids=positions, head_mask=None,
                  layer_past=None, use_cache=False, output_attentions=False,
                  cache_position=cache_position, position_embeddings=rotary)[0]
    h = backbone.final_layer_norm(h)
    logits = model.embed_out(h[:, -1:] if last_only else h)
    loss = None
    if labels is not None:
        if last_only:
            raise ValueError("last_only does not support sequence labels")
        loss = F.cross_entropy(logits[:, :-1].contiguous().view(-1, logits.shape[-1]),
                               labels[:, 1:].contiguous().view(-1))
    return SimpleNamespace(logits=logits, loss=loss)
