# Official source provenance

- URL: https://github.com/LUMIA-Group/PonderLM-2
- Shallow clone: `vendor/PonderLM-2/`, branch main, HEAD `fa784bece621b989fb008c59b0fd8d282fa9c73c`.
- Assistant VM source read first: same HEAD (fa784be).
- Clone command: `git clone --depth 1 https://github.com/LUMIA-Group/PonderLM-2 vendor/PonderLM-2`.
- Upstream remains unchanged. `.gitignore` excludes `/vendor/PonderLM-2/`;
  upstream source, nested git metadata, weights and environment are not committed.
- Recreate the shallow clone with the command above and verify the recorded hash;
  if main advances, fetch/check out this exact commit.

## Dependencies

Existing environment inspected: `/home/lenovo_jason/GPT2_Latent_Reasoning/.venv/bin/python`.
Available versions below are distribution metadata; core dependencies were also
actually imported (see reports/existing-import-check.txt). In particular,
`import accelerate` raised `ModuleNotFoundError: No module named 'accelerate'`.
The table covers all requirements.txt entries, the README install extras,
and the quickstart integrations. Needed by the package installation does not
mean needed by a direct modeling-file forward. Transitive probe versions are
fully listed in ENV.md. The full CLI is not installed or validated.

| Needed requirement / integration | Existing version / availability | Purpose |
|---|---|---|
| `transformers>=4.41.2,<=4.46.1` | 5.18.0 | Full CLI declared requirement |
| `datasets>=2.16.0,<=3.1.0` | 5.1.0 | Full CLI declared requirement |
| `accelerate>=0.34.0,<=1.0.1` | absent | Full CLI declared requirement |
| `peft>=0.11.1,<=0.12.0` | absent | Full CLI declared requirement |
| `trl>=0.8.6,<=0.9.6` | absent | Full CLI declared requirement |
| `tokenizers>=0.19.0,<0.20.4` | 0.23.2 | Full CLI declared requirement |
| `gradio>=4.0.0,<5.0.0` | absent | Full CLI declared requirement |
| `pandas>=2.0.0` | 3.0.6 | Full CLI declared requirement |
| `scipy` | absent | Full CLI declared requirement |
| `einops` | absent | Full CLI declared requirement |
| `sentencepiece` | absent | Full CLI declared requirement |
| `tiktoken` | absent | Full CLI declared requirement |
| `protobuf` | absent | Full CLI declared requirement |
| `uvicorn` | absent | Full CLI declared requirement |
| `pydantic` | absent | Full CLI declared requirement |
| `fastapi` | absent | Full CLI declared requirement |
| `sse-starlette` | absent | Full CLI declared requirement |
| `matplotlib>=3.7.0` | absent | Full CLI declared requirement |
| `fire` | absent | Full CLI declared requirement |
| `packaging` | 26.3 | Full CLI declared requirement |
| `pyyaml` | 6.0.3 | Full CLI declared requirement |
| `numpy<2.0.0` | 2.5.3 | Full CLI declared requirement |
| `av` | absent | Full CLI declared requirement |
| `tyro<0.9.0` | absent | Full CLI declared requirement |
| `torch` | 2.11.0 | Model + CLI; README tested 2.4/CUDA12.1; setup requires >=1.13.1 |
| `deepspeed` | absent | Quickstart ZeRO-0; setup extra >=0.10.0,<=0.14.4; omitted in probe |
| `wandb` | absent | Quickstart report_to wandb; no version pin; omitted in probe |
| `flash-attn` | absent | Quickstart fa2; README pins 2.7.2.post1; omitted in probe |
| `ponderlm2` | absent | Provides llamafactory-cli via editable install; omitted in probe |
| `nltk` | absent | README metrics extra; not needed by direct forward |
| `jieba` | absent | README metrics extra; not needed by direct forward |
| `rouge-chinese` | absent | README metrics extra; not needed by direct forward |

## Smoke result

PASS: direct import of upstream `GPTNeoXForCausalLM`, Pythia-70M pretrained
weights, bf16 GPU forward, train mode + autograd enabled, batch 1, length 64,
num_latent_thoughts=1, num_jacobi_iterations=2, random_jacobi_iterations=false,
scale_embeds=true, interpolation=false, use_cache=false, SDPA.
Peak allocated: 296641024 bytes (282.9 MiB), isolated cold forward 0.684455 s
(first download-run forward 2.485086 s; potentially overlapping retry).
Finite logits shape [1,64,50304]; no backward or parameter updates.
Weights: 166029852 bytes, first fetch 274.242354 s; config 567 bytes.
HF revision: a39f36b100fe8a5377810d56c3f4789b9c53ac42.
See reports/FEASIBILITY-01-official-route.md for estimates and limitations.
