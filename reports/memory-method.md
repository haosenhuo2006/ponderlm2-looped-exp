Memory methodology (estimates, not training measurements):

Use measured train-mode, autograd-enabled bf16 forward allocation F(B,S).
There is no backward, optimizer, data pipeline, or training run in this probe.
For ordinary AMP full training with fp32 parameters and gradients and fp32 Adam
moments, budget 16 bytes/parameter total versus the probe's 2-byte weights.
Add 14*parameter_count bytes to the forward result for the difference. This
is conservative relative to bf16 parameter/gradient Adam implementations;
DeepSpeed's precise state layout and temporary buffers are not measured.
Allow a further 25–50% of activation memory for backward temporaries, plus
0.5–1 GiB for runtime/allocator/workspace variation. Desktop usage is separate.

For extrapolation with memory-efficient SDPA/FlashAttention, fit activation
memory against B*S using the observed 256/512 shapes. This is an estimate:
SDPA dispatch, causal-mask allocation and longer sequences can add nonlinear
terms; do not use this linear extrapolation for eager attention. Eager
attention at interleaved length 4096 has a single B=8,H=8 bf16 attention
matrix of 2 GiB, per layer/pass, before softmax/backward intermediates.

For num_latent_thoughts=1 and num_jacobi_iterations=J, official code performs
one S-length initialization and J+1 passes at length 2S. Token-equivalent
transformer activation volume is therefore (2J+3)*B*S: 7 at J=2,
9 at J=3, 11 at J=4. Upstream random mode chooses {2,3,4}, whereas the plan
requires {2,3}; do not silently patch upstream during a feasibility probe.
Scale the transformer activation component, not fixed weights, by up to
11/7 for upstream's worst random draw (a conservative overestimate if
applied to all activation memory including logits, which are emitted once).

Gradient accumulation changes effective batch/token count, not the
per-microbatch activation peak. Original global tokens/update on eight GPUs:
8 GPUs * batch 8 * 2048 * accumulation 2 = 262144. On one GPU at B=1,S=512,
matching that would require accumulation 512; B=2,S=512 needs 256.
This is not a sensible default tiny-scale budget. Choose a smaller explicit
budget before EXP-01; this probe starts no EXP-01 training.
