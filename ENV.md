# Feasibility environment

Path: `/home/lenovo_jason/ponderlm2-looped-exp/.venv`. Created with uv 0.11.7.
Python: 3.12.3 (main, Aug 31 2026, 10:18:26) [GCC 13.3.0].
Runtime verified: torch **2.11.0+cu130**, transformers **4.46.1**, tokenizers **0.20.3**, NumPy **1.26.4**.

The existing GPT2 environment was inspected only. Its transformers 5.18.0,
datasets 5.1.0, tokenizers 0.23.2 and NumPy 2.5.3 violate upstream constraints;
accelerate is not importable. The experiment plan also requires independence.

Commands executed from the experiment repo, exclusively through tssh:
```bash
/home/lenovo_jason/.local/bin/uv venv --python /usr/bin/python3 .venv
/home/lenovo_jason/.local/bin/uv pip install --python .venv/bin/python torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130
/home/lenovo_jason/.local/bin/uv pip install --python .venv/bin/python transformers==4.46.1 numpy==1.26.4
```

The first torch install was interrupted after ~11 minutes of slow downloads;
an offline default-index dry run failed, then the same cu130 command resumed
and completed (remaining preparation 5m08s). Transformers/NumPy preparation
was 2m54s. See `reports/setup-notes.txt` for the exact offline error.
Only model-forward dependencies were installed. No flash-attn, DeepSpeed,
wandb, accelerate, datasets, peft, trl, or LLaMA-Factory CLI was installed.
The upstream modeling file is imported via importlib, bypassing CLI/package
initialization without modifying the file. It uses PyTorch SDPA.

All resolved distributions (also machine-readable in reports/environment-versions.json):

| Distribution | Exact version |
|---|---|
| Jinja2 | 3.1.6 |
| MarkupSafe | 3.0.3 |
| PyYAML | 6.0.3 |
| certifi | 2026.10.23 |
| charset-normalizer | 3.5.2 |
| cuda-bindings | 13.4.4 |
| cuda-pathfinder | 1.8.4 |
| cuda-toolkit | 13.0.2 |
| filelock | 4.2.4 |
| fsspec | 2026.9.0 |
| hf-xet | 1.8.0 |
| huggingface_hub | 0.36.2 |
| idna | 3.20 |
| mpmath | 1.3.0 |
| networkx | 3.7 |
| numpy | 1.26.4 |
| nvidia-cublas | 13.1.0.3 |
| nvidia-cuda-cupti | 13.0.85 |
| nvidia-cuda-nvrtc | 13.0.88 |
| nvidia-cuda-runtime | 13.0.96 |
| nvidia-cudnn-cu13 | 9.19.0.56 |
| nvidia-cufft | 12.0.0.61 |
| nvidia-cufile | 1.15.1.6 |
| nvidia-curand | 10.4.0.35 |
| nvidia-cusolver | 12.0.4.66 |
| nvidia-cusparse | 12.6.3.3 |
| nvidia-cusparselt-cu13 | 0.8.0 |
| nvidia-nccl-cu13 | 2.28.9 |
| nvidia-nvjitlink | 13.0.88 |
| nvidia-nvshmem-cu13 | 3.4.5 |
| nvidia-nvtx | 13.0.85 |
| packaging | 26.3 |
| regex | 2026.11.9 |
| requests | 2.34.2 |
| safetensors | 0.8.0 |
| setuptools | 78.1.0 |
| sympy | 1.14.0 |
| tokenizers | 0.20.3 |
| torch | 2.11.0+cu130 |
| tqdm | 4.70.1 |
| transformers | 4.46.1 |
| triton | 3.6.0 |
| typing_extensions | 4.16.0 |
| urllib3 | 2.8.0 |
