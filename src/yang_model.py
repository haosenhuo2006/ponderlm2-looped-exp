"""Yang et al.'s standard and additive looped GPT regression models.

Ported from scripts/models.py and scripts/nano_gpt.py at commit 6b3bdb2.
"""

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


def new_gelu(x):
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class LayerNorm(nn.Module):
    """Layer normalization with an optional bias."""

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        batch_size, seq_len, width = x.shape
        head_shape = (batch_size, seq_len, self.n_head, width // self.n_head)
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(head_shape).transpose(1, 2)
        k = k.view(head_shape).transpose(1, 2)
        v = v.view(head_shape).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, width)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = new_gelu(self.c_fc(x))
        return self.dropout(self.c_proj(x))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPT2Config:
    block_size: int = 1024
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True


class GPT2Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.apply(self._init_weights)
        for name, param in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(param, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(self, inputs_embeds, position_ids=None):
        seq_len = inputs_embeds.shape[1]
        assert seq_len <= self.config.block_size, (
            f"Cannot forward sequence of length {seq_len}, block size is {self.config.block_size}"
        )
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=inputs_embeds.device).unsqueeze(0)
        x = self.transformer.drop(inputs_embeds + self.transformer.wpe(position_ids))
        for block in self.transformer.h:
            x = block(x)
        return self.transformer.ln_f(x)


class TransformerModel(nn.Module):
    """Predict each y from its x token and the preceding examples."""

    def __init__(self, n_dims=20, n_positions=101, n_embd=256, n_layer=12, n_head=8,
                 pred_type="regression"):
        super().__init__()
        if pred_type != "regression":
            raise ValueError(f"Unsupported prediction type {pred_type!r}")
        self.configuration = GPT2Config(
            block_size=2 * n_positions + 1, n_layer=n_layer, n_head=n_head,
            n_embd=n_embd, dropout=0.0, bias=True,
        )
        self._read_in = nn.Linear(n_dims, n_embd)
        self._backbone = GPT2Model(self.configuration)
        self._read_out = nn.Linear(n_embd, 1)

    def _combine(self, xs, ys):
        """Interleave x tokens and zero-padded y tokens into shape [B, 2n, d]."""
        batch_size, n_points, n_dims = xs.shape
        ys_wide = torch.cat((ys.unsqueeze(-1), xs.new_zeros(batch_size, n_points, n_dims - 1)), dim=2)
        return torch.stack((xs, ys_wide), dim=2).view(batch_size, 2 * n_points, n_dims)

    def forward(self, xs, ys):
        embeds = self._read_in(self._combine(xs, ys))
        output = self._backbone(inputs_embeds=embeds)
        return self._read_out(output)[:, 0::2, 0]


class TransformerModelLooped(TransformerModel):
    """Apply a shared backbone with additive input injection at every loop."""

    def __init__(self, n_dims=20, n_positions=101, n_embd=256, n_layer=1, n_head=8,
                 loop_func="z=f(x+z)", pred_type="regression"):
        if loop_func != "z=f(x+z)":
            raise ValueError(f"Unsupported loop function {loop_func!r}")
        super().__init__(n_dims, n_positions, n_embd, n_layer, n_head, pred_type)

    def f(self, output, embeds):
        return self._backbone(inputs_embeds=output + embeds)

    def forward(self, xs, ys, n_loop_start, n_loops):
        embeds = self._read_in(self._combine(xs, ys))
        output = torch.zeros_like(embeds)
        pred_list = []
        for idx in range(n_loops):
            if idx < n_loop_start:
                with torch.no_grad():
                    output = self.f(output, embeds)
            else:
                output = self.f(output, embeds)
                pred_list.append(self._read_out(output)[:, 0::2, 0])
        return pred_list


def build_yang_model(model_type, n_dims=20, n_positions=101, n_embd=256, n_head=8):
    if model_type == "looped":
        return TransformerModelLooped(n_dims, n_positions, n_embd, n_layer=1, n_head=n_head)
    if model_type == "standard":
        return TransformerModel(n_dims, n_positions, n_embd, n_layer=12, n_head=n_head)
    raise ValueError(f"Unknown model type {model_type!r}")
