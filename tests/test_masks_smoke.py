"""EXP-01 forward/backward/optimizer smoke; D1/D2 masks are out of scope."""
import json
import random
import unittest
from pathlib import Path
import torch
from src.ponder import load_model, forward

class TrainingSmoke(unittest.TestCase):
    def test_both_configs(self):
        for name in ["t0", "t1"]:
            c = json.loads(Path("configs/exp01_"+name+".json").read_text())
            torch.manual_seed(c["seed"])
            model = load_model(c["mode"]).train()
            opt = torch.optim.AdamW(model.parameters(), lr=c["lr"])
            for step in range(2):
                x = torch.randint(0, model.config.vocab_size, (1,64), device="cuda")
                opt.zero_grad(set_to_none=True)
                out = forward(model, x, 2+step)
                self.assertTrue(torch.isfinite(out.loss).item())
                self.assertEqual(tuple(out.logits.shape), (1,64,50304))
                out.loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                opt.step()
                print(name, step, out.loss.item(), flush=True)
            del out, model, opt
            torch.cuda.empty_cache()
if __name__ == "__main__":
    unittest.main()
