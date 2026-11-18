"""D3: build a zero-step looped2 checkpoint from base Pythia-160M weights.

Same save path as train.py (model.save_pretrained) so the official evaluator
(src.evaluate_looped2 lm) can read it. No training step is run.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
os.environ.update(
    HF_HUB_OFFLINE="1",
    TRANSFORMERS_OFFLINE="1",
    PYTHIA_MODEL="EleutherAI/pythia-160M",
    PYTHIA_REVISION="50f5173d932e8e61f858120bcb800b97af589f46",
)
sys.path.insert(0, str(ROOT))

from src.ponder import load_model
from src import looped2

model = load_model("vanilla")
cfg = looped2.resolve_config(model.config.num_hidden_layers,
                             {"widths": [3, 6, 3]}, 1.0)
looped2.apply_config(model, cfg)
out = ROOT / "runs/reproduction/d3-zero-step"
model.save_pretrained(out)
print(f"D3 zero-step checkpoint saved to {out}")
print(f"params={sum(p.numel() for p in model.parameters())}")
