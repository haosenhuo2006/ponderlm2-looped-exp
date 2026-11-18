"""Prelude, recurrent core, and coda for the Huginn-style LM reproduction.

Local adaptations (A2): sandwich normalization uses LayerNorm only in the
core, while Huginn uses sandwich RMSNorm throughout. The vendored GPT-NeoX
core's parallel attention and MLP residuals become serial residuals here.
"""

from dataclasses import dataclass, fields
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def _validate_sampler(t, s, sigma, cap):
    if type(cap) is not int or cap < 1 or type(s) is not int or s < 0:
        raise ValueError("require integer cap >= 1 and s >= 0")
    if not math.isfinite(t + s) or t < 0 or t + s <= 0:
        raise ValueError("require finite t >= 0 and t + s > 0")
    if not math.isfinite(sigma) or sigma < 0:
        raise ValueError("require finite sigma >= 0")


@dataclass(frozen=True)
class Looped2Config:
    prelude: tuple[int, ...]
    core: tuple[int, ...]
    coda: tuple[int, ...]
    emb_scale: float = 1.0
    t: float = 2.0
    s: int = 2
    sigma: float = 0.5
    cap: int = 12
    eval_r: int = 4

    def __post_init__(self):
        for name in ("prelude", "core", "coda"):
            indices = getattr(self, name)
            if not isinstance(indices, (tuple, list)):
                raise ValueError(f"{name} must contain explicit layer indices")
            object.__setattr__(self, name, tuple(indices))
        self.validate_partition(len(self.prelude) + len(self.core) + len(self.coda))
        _validate_sampler(self.t, self.s, self.sigma, self.cap)
        if type(self.eval_r) is not int or self.eval_r < 1:
            raise ValueError("eval_r must be a positive integer")
        if not math.isfinite(self.emb_scale):
            raise ValueError("emb_scale must be finite")

    def validate_partition(self, num_layers):
        indices = self.prelude + self.core + self.coda
        if (not self.core or any(type(i) is not int for i in indices)
                or sorted(indices) != list(range(num_layers))):
            raise ValueError("prelude, nonempty core and coda must partition all layers")

    @classmethod
    def from_widths(cls, num_layers, p, r_, c, **kwargs):
        if (type(num_layers) is not int
                or any(type(width) is not int or width < 0 for width in (p, r_, c))
                or not r_ or p + r_ + c != num_layers):
            raise ValueError("widths must partition all layers with a nonempty core")
        return cls(tuple(range(p)), tuple(range(p, p + r_)),
                   tuple(range(p + r_, num_layers)), **kwargs)


def resolve_config(num_layers, values, default_emb_scale=1.0):
    """Resolve the width shorthand or explicit layer indices from JSON."""
    if not isinstance(values, dict):
        raise ValueError("looped2 must be a config object")
    values = dict(values)
    values.setdefault("emb_scale", default_emb_scale)
    segments = ("prelude", "core", "coda")
    if "widths" in values:
        widths = values.pop("widths")
        if any(name in values for name in segments):
            raise ValueError("use looped2 widths or explicit layer indices")
        if not isinstance(widths, (tuple, list)) or len(widths) != 3:
            raise ValueError("looped2 widths must contain prelude, core and coda widths")
        return Looped2Config.from_widths(num_layers, *widths, **values)
    if not all(name in values for name in segments):
        raise ValueError("looped2 requires widths or all three layer segments")
    cfg = Looped2Config(**values)
    cfg.validate_partition(num_layers)
    return cfg


def apply_config(model, cfg):
    layers = model.gpt_neox.layers
    cfg.validate_partition(len(layers))
    for field in fields(cfg):
        setattr(model.config, f"looped2_{field.name}", getattr(cfg, field.name))
    model.config.use_cache = False
    for index in cfg.core:
        layer = layers[index]
        weight = layer.input_layernorm.weight
        for name in ("looped2_n2", "looped2_n4"):
            if not hasattr(layer, name):
                # D1/D3: register before optimizer creation and checkpoint loading.
                setattr(layer, name, nn.LayerNorm(model.config.hidden_size,
                                                 eps=model.config.layer_norm_eps,
                                                 device=weight.device, dtype=weight.dtype))
    return model


def config_from_checkpoint(config):
    """Return the recurrent config, or None for a vanilla checkpoint."""
    segments = ("prelude", "core", "coda")
    present = [hasattr(config, f"looped2_{name}") for name in segments]
    if not any(present):
        return None
    if not all(present) or not hasattr(config, "looped2_emb_scale"):
        raise ValueError("incomplete looped2 checkpoint config")
    return resolve_config(config.num_hidden_layers, {
        field.name: getattr(config, f"looped2_{field.name}")
        for field in fields(Looped2Config) if hasattr(config, f"looped2_{field.name}")
    })


def sample_s0(shape, device, dtype, generator=None):
    # A5/E2: the pre-truncation std is independent of hidden size and gamma.
    # With a CPU generator, draw on CPU before transferring to the model.
    std = math.sqrt(2 / 5)
    draw_device = device if generator is None else generator.device
    state = torch.empty(shape, device=draw_device, dtype=dtype)
    nn.init.trunc_normal_(state, std=std, a=-3 * std, b=3 * std, generator=generator)
    return state.to(device=device)


def sample_nkr(generator, t=2.0, s=2, sigma=0.5, cap=12):
    _validate_sampler(t, s, sigma, cap)
    # A4: the log-normal mean is t+s; the shifted Poisson mean is t+s+1.
    mu = math.log(t + s) - sigma ** 2 / 2
    normal = torch.randn((), device="cpu", generator=generator)
    rate = (mu + sigma * normal).exp()
    r = min(1 + int(torch.poisson(rate, generator=generator).item()), cap)
    k = min(s, r)
    return r - k, k, r


def expected_rk(t=2.0, s=2, sigma=0.5, cap=12):
    """C3: integrate the capped Poisson survival sums over the log-normal rate."""
    _validate_sampler(t, s, sigma, cap)
    nodes, weights = np.polynomial.hermite.hermgauss(64)
    log_rates = math.log(t + s) - sigma ** 2 / 2 + math.sqrt(2) * sigma * nodes
    rates = np.exp(log_rates)
    # E[min(1+Poisson(rate), j)] = 1 + sum_{i=1}^{j-1} P(Poisson(rate) >= i).
    r_mean, k_mean = 1.0, float(s > 0)
    cdf = np.zeros_like(rates)
    for count in range(cap - 1):
        cdf += np.exp(count * log_rates - rates - math.lgamma(count + 1))
        tail = float(np.dot(weights, np.clip(1 - cdf, 0, 1)) / math.sqrt(math.pi))
        r_mean += tail
        if count + 1 < s:
            k_mean += tail
    return r_mean, k_mean


def layer_flops_per_token(h, ff, vocab, context=128):
    # C1: dense projections and causal attention at the mean context length.
    return 8 * h ** 2 + 4 * h * ff + 4 * h * context, 2 * h * vocab


def train_flops_per_token(r, k, n_prelude, n_core, n_coda, h, ff, vocab, context=128):
    layer, head = layer_flops_per_token(h, ff, vocab, context)
    # C2: gradients from the tracked injections meet at one prelude graph.
    outer = n_prelude + n_coda
    return ((outer + r * n_core) * layer + head
            + 2 * ((outer + k * n_core) * layer + head))


def expected_flops_ratio(n_prelude=3, n_core=6, n_coda=3, h=768, ff=3072,
                         vocab=50304, context=128, *, t=2.0, s=2, sigma=0.5, cap=12):
    r, k = expected_rk(t, s, sigma, cap)
    looped = train_flops_per_token(r, k, n_prelude, n_core, n_coda, h, ff, vocab, context)
    vanilla = train_flops_per_token(1, 1, n_prelude, n_core, n_coda, h, ff, vocab, context)
    return looped / vanilla


def _attention_args(mask, positions, cache_position, rotary):
    return dict(attention_mask=mask, position_ids=positions, head_mask=None,
                layer_past=None, use_cache=False, output_attentions=False,
                cache_position=cache_position, position_embeddings=rotary)


def core_block(layer, x, mask, positions, cache_position, rotary):
    # A2: use the same dropout sites as the vendored layer and serial residuals.
    attn = layer.attention(layer.input_layernorm(x),
                           **_attention_args(mask, positions, cache_position, rotary))[0]
    h = layer.looped2_n2(x + layer.post_attention_dropout(attn))
    mlp = layer.mlp(layer.post_attention_layernorm(h))
    return layer.looped2_n4(h + layer.post_mlp_dropout(mlp))


def _embed(backbone, x, emb_scale):
    e = backbone.embed_in(x) * emb_scale  # A6/E2: gamma belongs to the forward.
    cache_position = torch.arange(x.shape[1], device=x.device)
    positions = cache_position[None].expand(x.shape[0], -1)
    mask = backbone._update_causal_mask(torch.ones_like(x), e, cache_position, None, False)
    e = backbone.emb_dropout(e)
    return e, (mask, positions, cache_position, backbone.rotary_emb(e, positions))


def _output(model, h, labels, last_only=False):
    logits = model.embed_out(h[:, -1:] if last_only else h)
    loss = None
    if labels is not None:
        # A8: supervise only the final, shifted next-token predictions.
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]),
                               labels[:, 1:].reshape(-1))
    return SimpleNamespace(logits=logits, loss=loss)


def plain_forward(model, x, labels=None, emb_scale=1.0):
    backbone = model.gpt_neox
    h, attention = _embed(backbone, x, emb_scale)
    args = _attention_args(*attention)
    for layer in backbone.layers:
        h = layer(h, **args)[0]
    return _output(model, backbone.final_layer_norm(h), labels)


def _depths(cfg, mode, rng, r_override, k_override):
    if mode not in ("train", "eval"):
        raise ValueError("mode must be 'train' or 'eval'")
    if r_override is not None:
        r = r_override
    elif mode == "eval":
        r = cfg.eval_r
    else:
        _, _, r = sample_nkr(rng, cfg.t, cfg.s, cfg.sigma, cfg.cap)
    if type(r) is not int or r < 1:
        raise ValueError("r must be a positive integer")
    if mode == "eval" and k_override is None:
        return 0, 0, r  # D5: eval runs all rounds without training truncation.
    k = min(cfg.s, r) if k_override is None else k_override
    if type(k) is not int or not 0 <= k <= r:
        raise ValueError("k must be an integer in 0..r")
    return r - k, k, r


def forward(model, x, labels=None, last_only=False, *, cfg, mode="train",
            rng=None, s0=None, s0_seed=None, r_override=None, k_override=None, record=None):
    if last_only and labels is not None:
        raise ValueError("last_only does not support sequence labels")
    backbone = model.gpt_neox
    cfg.validate_partition(len(backbone.layers))
    n, k, r = _depths(cfg, mode, rng, r_override, k_override)
    e, attention = _embed(backbone, x, cfg.emb_scale)
    args = _attention_args(*attention)
    for index in cfg.prelude:
        e = backbone.layers[index](e, **args)[0]

    if s0 is None:
        generator = torch.Generator().manual_seed(s0_seed) if s0_seed is not None else rng
        s0 = sample_s0(e.shape, e.device, e.dtype, generator)
    elif s0.shape != e.shape:
        raise ValueError("s0 must have the same shape as the prelude output")

    def apply_core(h, rounds):
        for _ in range(rounds):
            h = h + e  # A1: inject the same prelude output once per round.
            for index in cfg.core:
                h = core_block(backbone.layers[index], h, *attention)
        return h

    h = s0
    if n:
        # A4/D2: preserve e's graph for later injections. Disable the autocast
        # weight cache here so casts made under no_grad cannot affect backprop.
        device_type = e.device.type
        with torch.no_grad(), torch.autocast(device_type,
                                             enabled=torch.is_autocast_enabled(device_type),
                                             cache_enabled=False):
            h = apply_core(h, n)
    h = apply_core(h, r - n)

    # A3: use the very same final norm at the core boundary and after the coda.
    h = backbone.final_layer_norm(h)
    for index in cfg.coda:
        h = backbone.layers[index](h, **args)[0]
    out = _output(model, backbone.final_layer_norm(h), labels, last_only)
    if record is not None:
        record.update(nkr=(n, k, r), r=r)
    return out


@torch.no_grad()
def reinitialize_from_scratch(model, l_eff):
    if not math.isfinite(l_eff) or l_eff <= 0:
        raise ValueError("l_eff must be positive and finite")
    # A7/D3: use nominal effective depth for residual output projections.
    h = model.config.hidden_size
    base_std = math.sqrt(2 / (5 * h))
    output_std = 1 / math.sqrt(5 * h * l_eff)
    seen = set()
    for module in model.modules():
        if not isinstance(module, (nn.Linear, nn.Embedding, nn.LayerNorm)):
            continue
        weight = module.weight
        if weight is not None and id(weight) not in seen:
            seen.add(id(weight))
            if isinstance(module, nn.LayerNorm):
                nn.init.ones_(weight)
            else:
                is_output = hasattr(module, "_is_attention_output") or hasattr(module, "_is_mlp_output")
                std = output_std if is_output else base_std
                nn.init.trunc_normal_(weight, std=std, a=-3 * std, b=3 * std)
                if isinstance(module, nn.Embedding) and module.padding_idx is not None:
                    weight[module.padding_idx].zero_()
        bias = getattr(module, "bias", None)
        if bias is not None and id(bias) not in seen:
            seen.add(id(bias))
            nn.init.zeros_(bias)


def load_looped2_checkpoint(checkpoint, device="cuda"):
    from transformers import GPTNeoXConfig
    from transformers.modeling_utils import load_state_dict
    from .ponder import ROOT, VENDORED_PATH

    checkpoint = Path(checkpoint)
    config = GPTNeoXConfig.from_pretrained(checkpoint)
    cfg = config_from_checkpoint(config)
    if cfg is None:
        raise ValueError(f"checkpoint {checkpoint} has no looped2 layer segments")
    config._attn_implementation = "sdpa"
    spec = importlib.util.spec_from_file_location("looped2_gpt_neox", ROOT / VENDORED_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # D3/E2: the base class cannot restore added norms before registration.
    model = apply_config(module.GPTNeoXForCausalLM(config), cfg)
    weights = checkpoint / "model.safetensors"
    if not weights.is_file():
        weights = checkpoint / "pytorch_model.bin"
    state = load_state_dict(str(weights))
    if model.embed_out.weight is model.gpt_neox.embed_in.weight:
        # save_pretrained can omit the tied head alias from safetensors.
        if "embed_out.weight" not in state and "gpt_neox.embed_in.weight" in state:
            state["embed_out.weight"] = state["gpt_neox.embed_in.weight"]
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()
