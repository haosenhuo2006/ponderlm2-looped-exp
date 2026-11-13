"""Shared pair-level causal masks for diagnostics and future T2/T3 training.

Rows=query, columns=key; return FP32 additive [1,1,L,L], 0 allowed,
-inf blocked, L=seq_len*(1+num_thoughts). Causality always applies.
Exact table (query belongs to pair i; key belongs to pair j):
          previous x(j<i)  current x  previous h(j<i) current h  future
D0             allow        allow         allow       causal   block
D1             allow        allow         block       causal   block
D2             block        allow         allow       causal   block
"current h causal": x_i cannot see its following h_i; h_i sees itself
(and earlier thoughts in its own pair for num_thoughts>1).
Token-only initialization uses num_thoughts=0: D2 is diagonal.
Restrictions concern direct attention edges; D2 permits indirect history via h.
"""
import torch


def _build(seq_len, num_thoughts, condition):
    if seq_len < 1 or num_thoughts < 0:
        raise ValueError("seq_len>=1 and num_thoughts>=0 required")
    width = num_thoughts + 1
    ids = torch.arange(seq_len * width)
    q, k = ids[:, None], ids[None, :]
    previous = k // width < q // width
    allowed = k <= q
    if condition == "D1":
        allowed &= ~(previous & (k % width != 0))
    if condition == "D2":
        allowed &= ~(previous & (k % width == 0))
    return torch.zeros(allowed.shape, dtype=torch.float32).masked_fill(
        ~allowed, float("-inf"))[None, None]


def full_mask(seq_len, num_thoughts):
    return _build(seq_len, num_thoughts, "D0")


def no_previous_h_mask(seq_len, num_thoughts):
    return _build(seq_len, num_thoughts, "D1")


def no_previous_x_mask(seq_len, num_thoughts):
    return _build(seq_len, num_thoughts, "D2")


MASKS = {"D0": full_mask, "D1": no_previous_h_mask, "D2": no_previous_x_mask}
