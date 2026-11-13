| Measurement | T0 | T1 |
|---|---:|---:|
| Training wall seconds | 242.964 | 905.812 |
| Whole-run train input tok/s | 10115.1 | 2713.1 |
| Peak train allocated MiB | 1392.42 | 1685.39 |
| nll | 4.927077 | 6.006961 |
| ppl | 137.975574 | 406.246846 |
| tokens_s | 27115.938154 | 7119.060528 |
| sequential_nll_8x128 | 4.560739 | 5.897509 |
| sequential_tokens_s | 235.192010 | 89.624143 |
| inference_flops_per_token_estimate | 90832896.000000 | 133300224.000000 |
| kv_cache_bytes_127_tokens | 2340864.000000 | 4681728.000000 |
| Peak validation allocated MiB | 517.25 | 621.27 |

Equal-input-token loss curves (mean over up to last 20 steps):

| Step | Input tokens | T0 loss | T1 loss |
|---:|---:|---:|---:|
| 1 | 4096 | 6.083247 | 11.636139 |
| 20 | 81920 | 6.143621 | 10.276195 |
| 100 | 409600 | 6.781144 | 8.470191 |
| 200 | 819200 | 6.325436 | 7.672082 |
| 400 | 1638400 | 5.652151 | 6.124055 |
| 600 | 2457600 | 4.794231 | 5.865784 |

Approximately matched training FLOPs (paper estimate T1 step = 3+2*num_jacobi_iterations T0 steps; excludes output-head and sequence-dependent corrections):

| T1 step | T0-equivalent step | T0 last-20 loss | T1 last-20 loss |
|---:|---:|---:|---:|
| 20 | 150 | 6.572875 | 10.276195 |
| 40 | 310 | 6.022220 | 10.070837 |
| 60 | 470 | 5.234648 | 9.075374 |

num_jacobi_iterations counts: {2: 306, 3: 294}. Mean=2.490000; estimated total training FLOPs ratio=7.980000. Measured whole-run wall ratio=3.728 (wall time is not FLOPs). Inference FLOPs ratio=1.468. Combined training wall=1148.776s.

Jacobi convergence (mean of eight sequence RMSEs):

| Refinement k | r_k | d_k to sequential |
|---:|---:|---:|
| 1 | 1.81736842 | 0.32385242 |
| 2 | 0.311193429 | 0.149037011 |
| 3 | 0.136963565 | 0.138454218 |
| 4 | 0.128487946 | 0.128616437 |
| 5 | 0.130829365 | 0.132344875 |
| 6 | 0.126813856 | 0.141658156 |
| 7 | 0.128263625 | 0.141007504 |
| 8 | 0.122818545 | 0.130986349 |

Additional FP32 diagnostic (same checkpoint and sequences; training/evaluation unchanged):

| Refinement k | FP32 r_k | FP32 d_k |
|---:|---:|---:|
| 1 | 1.80690223 | 0.243136117 |
| 2 | 0.239833223 | 0.0286851602 |
| 3 | 0.0283047131 | 0.00340336718 |
| 4 | 0.00336719358 | 0.000345153287 |
| 5 | 0.000338720829 | 0.000201508919 |
| 6 | 0.000196074 | 0.000195172832 |
| 7 | 0.000184897784 | 0.000192225577 |
| 8 | 0.000183505808 | 0.000199216776 |
