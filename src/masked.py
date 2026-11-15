"""Differentiable masked Jacobi schedule using unmodified official helpers."""
from types import SimpleNamespace
import torch
import torch.nn.functional as F
from .ponder import embeddings
from .diagnose import backbone, interleaved
from .masks import MASKS


def forward(model, x, iterations=3, condition="D1", labels=None, intermediate_supervision=False, return_round_logits=False):
    """Initialization -> K refinements -> thought logits; never detach states.

    Caller controls autocast, just as the official backbone does. Optional labels
    support ignored targets for tests and follow the official shifted CE contract.
    """
    if iterations not in (2, 3):
        raise ValueError("training/evaluation iterations must be 2 or 3")
    return _forward(model, x, iterations, condition, labels,
                    intermediate_supervision, return_round_logits)


def forward_eval(model, x, iterations, condition="D0", labels=None,
                 intermediate_supervision=False, return_round_logits=False):
    """R refinements plus a separate final pass; R=0 uses initialization only.

    Caller controls eval mode, gradients and autocast. With R=0 there are no
    intermediate losses, so intermediate supervision falls back to final loss.
    """
    if isinstance(iterations, bool) or not isinstance(iterations, int) or not 0 <= iterations <= 5:
        raise ValueError("evaluation iterations must be an integer from 0 to 5")
    return _forward(model, x, iterations, condition, labels,
                    intermediate_supervision, return_round_logits)


def _forward(model, x, iterations, condition, labels,
             intermediate_supervision, return_round_logits):
    e = embeddings(model, x)
    pos = torch.arange(x.shape[1], device=x.device)[None].expand(x.shape[0], -1)
    mask = torch.ones_like(pos) if condition == "D0" else MASKS[condition](x.shape[1], 0)
    out = backbone(model, e, pos, mask)
    h = model.extract_hidden_for_computation(out, slice(0, None, 1), model.config)
    round_logits, round_losses = [], []
    targets = x if labels is None else labels
    # Round k=1..K is the kth INTERLEAVED REFINEMENT pass, using H^(k-1).
    # Its odd/thought outputs predict x_(i+1); even/token outputs form H^(k).
    # Token-only initialization has no thought slots and is not supervised.
    # The separate final pass below remains T1's prediction, not a K+1 loss.
    for _ in range(iterations):
        out = interleaved(model, e, h, condition)
        if intermediate_supervision or return_round_logits:
            z = model.embed_out(out.last_hidden_state[:, 1::2])
            # Shifted next-token CE as EXP-01, with explicit FP32 logits.
            # This avoids autocast-dependent CE numerical differences while
            # vocab-head matmuls retain bf16 compute.
            ce = F.cross_entropy(z[:, :-1].contiguous().float().view(-1, z.shape[-1]),
                                 targets[:, 1:].contiguous().view(-1))
            round_losses.append(ce)
            if return_round_logits:
                round_logits.append(z)
        h = model.extract_hidden_for_computation(out, slice(0, None, 2), model.config)
    out = interleaved(model, e, h, condition)
    logits = model.embed_out(out.last_hidden_state[:, 1::2])
    targets = x if labels is None else labels
    loss = F.cross_entropy(logits[:, :-1].contiguous().view(-1, logits.shape[-1]),
                           targets[:, 1:].contiguous().view(-1))
    final_loss = loss
    if intermediate_supervision and round_losses:
        loss = torch.stack(round_losses).mean()
    return SimpleNamespace(loss=loss, logits=logits, final_loss=final_loss,
                           round_losses=tuple(round_losses),
                           round_logits=tuple(round_logits))
