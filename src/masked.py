"""Differentiable masked Jacobi schedule using unmodified official helpers."""
from types import SimpleNamespace
import torch
import torch.nn.functional as F
from .ponder import embeddings
from .diagnose import backbone, interleaved
from .masks import MASKS


def forward(model, x, iterations=3, condition="D1", labels=None):
    """Initialization -> K refinements -> thought logits; never detach states.

    Caller controls autocast, just as the official backbone does. Optional labels
    support ignored targets for tests and follow the official shifted CE contract.
    """
    if iterations not in (2, 3):
        raise ValueError("training/evaluation iterations must be 2 or 3")
    e = embeddings(model, x)
    pos = torch.arange(x.shape[1], device=x.device)[None].expand(x.shape[0], -1)
    mask = torch.ones_like(pos) if condition == "D0" else MASKS[condition](x.shape[1], 0)
    out = backbone(model, e, pos, mask)
    h = model.extract_hidden_for_computation(out, slice(0, None, 1), model.config)
    for _ in range(iterations):
        out = interleaved(model, e, h, condition)
        h = model.extract_hidden_for_computation(out, slice(0, None, 2), model.config)
    out = interleaved(model, e, h, condition)
    logits = model.embed_out(out.last_hidden_state[:, 1::2])
    targets = x if labels is None else labels
    loss = F.cross_entropy(logits[:, :-1].contiguous().view(-1, logits.shape[-1]),
                           targets[:, 1:].contiguous().view(-1))
    return SimpleNamespace(loss=loss, logits=logits)
