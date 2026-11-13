"""T5 unified supervision, exact round attribution, and schedule checks."""
import random
import unittest
from unittest.mock import patch
import torch
import torch.nn.functional as F
from src.ponder import load_model, forward as official
from src.masked import forward


class IntermediateSupervision(unittest.TestCase):
    def tearDown(self):
        torch.cuda.empty_cache()

    def test_two_optimizer_steps(self):
        torch.manual_seed(42)
        model = load_model("ponder").train()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(.9, .95), weight_decay=.01)
        for k in (2, 3):
            x = torch.randint(model.config.vocab_size, (1, 64), device="cuda")
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = forward(model, x, k, "D0", intermediate_supervision=True)
            self.assertTrue(torch.isfinite(out.loss))
            self.assertEqual(len(out.round_losses), k)
            self.assertTrue(all(torch.isfinite(v) for v in out.round_losses))
            out.loss.backward()  # One backward and ONE unified update.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            opt.step()
            self.assertEqual(next(model.parameters()).dtype, torch.float32)

    def test_mean_and_final_prediction_parity(self):
        torch.manual_seed(42)
        model = load_model("ponder").eval()
        x = torch.randint(model.config.vocab_size, (1, 8), device="cuda")
        for k in (2, 3):
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                out = forward(model, x, k, "D0", intermediate_supervision=True,
                              return_round_logits=True)
                reference = official(model, x, k)
            # Independent FP32 CE recomputation from each exact returned logit
            # tensor, outside autocast. No rerun with different random states.
            ces = [F.cross_entropy(z[:, :-1].contiguous().float().view(-1, z.shape[-1]),
                                   x[:, 1:].contiguous().view(-1)) for z in out.round_logits]
            torch.testing.assert_close(out.loss, torch.stack(ces).mean(), atol=1e-5, rtol=0)
            torch.testing.assert_close(torch.stack(out.round_losses), torch.stack(ces), atol=1e-5, rtol=0)
            torch.testing.assert_close(out.logits, reference.logits, atol=1e-6, rtol=0)

    def test_round_two_perturbation_attribution(self):
        torch.manual_seed(42)
        model = load_model("ponder").eval()
        x = torch.randint(model.config.vocab_size, (1, 8), device="cuda")
        original = model.gpt_neox.forward
        delta = torch.randn(1, 16, model.config.hidden_size, device="cuda") * 10
        def run(perturb):
            calls = 0
            def wrapped(*args, **kwargs):
                nonlocal calls
                out = original(*args, **kwargs)
                calls += 1
                if perturb and calls == 3:  # init=1, refinement 1=2, refinement 2=3
                    out.last_hidden_state = out.last_hidden_state + delta
                return out
            with torch.no_grad(), patch.object(model.gpt_neox, "forward", side_effect=wrapped):
                return forward(model, x, 3, "D0", intermediate_supervision=True)
        a, b = run(False), run(True)  # FP32, deterministic; all 6 layers.
        torch.testing.assert_close(a.round_losses[0], b.round_losses[0], atol=0, rtol=0)
        self.assertGreater(abs((a.round_losses[1]-b.round_losses[1]).item()), 1e-5)
        self.assertGreater(abs((a.loss-b.loss).item()), 1e-5)

    def test_600_step_schedule(self):
        rng = random.Random(42)
        ks = [rng.choice([2, 3]) for _ in range(600)]
        self.assertEqual(set(ks), {2, 3})
        self.assertEqual([ks.count(2), ks.count(3)], [306, 294])
