"""T8 full-graph mean supervision with split optimizer updates."""
import math
import random
import unittest
from unittest.mock import patch
import torch
from src.masked import forward as masked_forward
from src.ponder import load_model
from src.train import split_supervised_update_batch


class Schedule(unittest.TestCase):
    def test_600_step_schedule(self):
        rng = random.Random(42)
        ks = [rng.choice([2, 3]) for _ in range(600)]
        self.assertEqual([ks.count(2), ks.count(3), sum(ks)], [306, 294, 1494])


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA and cached model")
class SplitSupervisedUpdates(unittest.TestCase):
    def tearDown(self):
        torch.cuda.empty_cache()

    def test_gradient_equivalence_and_loss_definition(self):
        torch.manual_seed(42)
        model = load_model("ponder").eval()
        batches = [torch.randint(model.config.vocab_size, (1, 16), device="cuda") for _ in range(16)]
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        for k in (2, 3):
            model.zero_grad(set_to_none=True)
            losses, rounds, finals = [], [], []
            for x in batches:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = masked_forward(model, x, k, "D0", intermediate_supervision=True)
                (out.loss / 16).backward()
                losses.append(out.loss.detach().item())
                rounds.append([v.detach().item() for v in out.round_losses])
                finals.append(out.final_loss.detach().item())
            expected = {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}
            model.zero_grad(set_to_none=True)
            # Disable parameter mutation, resets and clipping: compare summed
            # group gradients at identical parameters and autocast conditions.
            with patch.object(optimizer, "zero_grad"), patch.object(optimizer, "step"), patch(
                "src.train.torch.nn.utils.clip_grad_norm_", return_value=0.
            ):
                result = split_supervised_update_batch(model, batches, k, optimizer, 1.)
            for name, parameter in model.named_parameters():
                if name in expected:
                    torch.testing.assert_close(parameter.grad, expected[name], atol=1e-6, rtol=0)
                else:
                    self.assertIsNone(parameter.grad)
            self.assertAlmostEqual(result["loss"], sum(losses) / 16, delta=1e-6)
            self.assertAlmostEqual(result["loss"], sum(result["round_ce"]) / k, delta=1e-6)
            for j in range(k):
                self.assertAlmostEqual(result["round_ce"][j], sum(r[j] for r in rounds) / 16, delta=1e-6)
            self.assertAlmostEqual(result["final_prediction_ce"], sum(finals) / 16, delta=1e-6)

    def test_update_count_and_parameter_change(self):
        torch.manual_seed(42)
        model = load_model("ponder").train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        batches = [torch.randint(model.config.vocab_size, (1, 16), device="cuda") for _ in range(16)]
        for k in (2, 3):
            before = model.embed_out.weight.detach().clone()
            with patch.object(optimizer, "step", wraps=optimizer.step) as step:
                result = split_supervised_update_batch(model, batches, k, optimizer, 1.)
            self.assertEqual(step.call_count, k)
            self.assertEqual(result["optimizer_updates"], k)
            self.assertFalse(torch.equal(before, model.embed_out.weight))
            self.assertEqual(len(result["round_ce"]), k)
            self.assertTrue(all(math.isfinite(v) for v in [
                result["loss"], result["grad_norm"], result["final_prediction_ce"], *result["round_ce"]]))
        for k in (0, 1, 4):
            with self.assertRaises(ValueError):
                split_supervised_update_batch(model, batches, k, optimizer, 1.)


if __name__ == "__main__":
    unittest.main()
