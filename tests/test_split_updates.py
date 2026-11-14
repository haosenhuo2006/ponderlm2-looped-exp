"""T7 partitions and GPU checks using the same small Pythia model as T6."""
import math
import unittest
from unittest.mock import patch
import torch
from src.ponder import forward, load_model
from src.train import split_groups, split_update_batch


class SplitGroups(unittest.TestCase):
    def test_partition(self):
        self.assertEqual(split_groups(16, 2), [list(range(8)), list(range(8, 16))])
        groups = split_groups(16, 3)
        self.assertEqual([len(group) for group in groups], [6, 5, 5])
        self.assertEqual([i for group in groups for i in group], list(range(16)))
        for k in (0, -1, 17, 2.5):
            with self.assertRaises(ValueError):
                split_groups(16, k)


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA and cached model")
class SplitUpdates(unittest.TestCase):
    def tearDown(self):
        torch.cuda.empty_cache()

    def test_gradient_equivalence(self):
        torch.manual_seed(42)
        model = load_model("ponder").eval()
        batches = [torch.randint(model.config.vocab_size, (1, 16), device="cuda") for _ in range(16)]
        model.zero_grad(set_to_none=True)
        for x in batches:
            (forward(model, x, 2).loss / len(batches)).backward()
        expected = {name: p.grad.detach().clone() for name, p in model.named_parameters() if p.grad is not None}
        model.zero_grad(set_to_none=True)
        # Exercise the production helper, disabling mutations to compare the
        # sum of group gradients at fixed parameters with T1 accumulation.
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        with patch.object(optimizer, "zero_grad"), patch.object(optimizer, "step"), patch(
            "src.train.torch.nn.utils.clip_grad_norm_", return_value=0.
        ):
            split_update_batch(model, batches, 2, optimizer, 1.)
        for name, parameter in model.named_parameters():
            if name in expected:
                torch.testing.assert_close(parameter.grad, expected[name], atol=1e-6, rtol=0)
            else:
                self.assertIsNone(parameter.grad)

    def test_update_count_and_parameter_change(self):
        torch.manual_seed(42)
        model = load_model("ponder").train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        batches = [torch.randint(model.config.vocab_size, (1, 16), device="cuda") for _ in range(16)]
        before = model.embed_out.weight.detach().clone()
        with patch.object(optimizer, "step", wraps=optimizer.step) as step:
            result = split_update_batch(model, batches, 2, optimizer, 1.)
        self.assertEqual(step.call_count, 2)
        self.assertEqual(result["optimizer_updates"], 2)
        self.assertFalse(torch.equal(before, model.embed_out.weight))
        self.assertTrue(all(math.isfinite(result[key]) for key in ("loss", "grad_norm")))


if __name__ == "__main__":
    unittest.main()
