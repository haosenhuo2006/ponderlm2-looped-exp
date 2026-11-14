"""Small-input inference smoke test; never loads the validation dataset."""
import math
import unittest

import torch

from src.ponder import load_model
from src.probe_h import probe_chunk


class ProbeHSmoke(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA and model weights")
    def test_probe_chunk(self):
        torch.manual_seed(42)
        model = load_model("ponder").eval()
        x = torch.randint(0, model.config.vocab_size, (1, 16), device="cuda")
        for condition in ("D0", "D1"):
            with self.subTest(condition=condition):
                result = probe_chunk(model, x, condition)
                self.assertEqual(set(result["per_k"]), {"0", "1", "2", "3"})
                self.assertEqual(set(result["round_thought"]), {"1", "2", "3"})
                self.assertEqual(set(result["baselines"]), {"shuffled", "token_hidden", "zero"})
                for group in ("per_k", "round_thought"):
                    for metrics in result[group].values():
                        self.assertEqual(set(metrics), {"next_token_acc", "final_agreement"})
                for baseline in result["baselines"].values():
                    self.assertEqual(set(baseline), {"H0", "H3", "round_3_thought"})
                    for metrics in baseline.values():
                        self.assertEqual(set(metrics), {"next_token_acc"})

                def check_metrics(value):
                    if isinstance(value, dict):
                        for child in value.values():
                            check_metrics(child)
                    else:
                        self.assertTrue(math.isfinite(value))
                        self.assertGreaterEqual(value, 0.0)
                        self.assertLessEqual(value, 1.0)

                check_metrics(result)


if __name__ == "__main__":
    unittest.main()
