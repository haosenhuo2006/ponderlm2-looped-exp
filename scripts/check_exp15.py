"""Check EXP-15 config identity and environment defaults without model imports."""
import ast
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    for model in ("T0", "T1"):
        original = json.loads((ROOT / f"configs/exp14_{model.lower()}.json").read_text())
        config = json.loads((ROOT / f"configs/exp15_{model.lower()}.json").read_text())
        expected = dict(original, steps=1200, run_dir=f"runs/EXP-15-{model}")
        assert config == expected, f"{model}: unexpected configuration differences"
    pipeline = ROOT / "scripts/run_exp15.sh"
    assert pipeline.is_file() and os.access(pipeline, os.X_OK), "pipeline missing or not executable"
    assignments = {
        node.targets[0].id: node.value
        for node in ast.parse((ROOT / "src/ponder.py").read_text()).body
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
    }
    for name, env, default in (
        ("MODEL", "PYTHIA_MODEL", "EleutherAI/pythia-70m"),
        ("REVISION", "PYTHIA_REVISION", "a39f36b100fe8a5377810d56c3f4789b9c53ac42"),
    ):
        expected = ast.parse(f'os.environ.get({env!r}, {default!r})', mode="eval").body
        assert ast.dump(assignments[name]) == ast.dump(expected), f"{name}: override/default differs"
    print("EXP-15: configs match EXP-14 except steps/run_dir; pipeline executable; 70M defaults preserved")


if __name__ == "__main__":
    main()
