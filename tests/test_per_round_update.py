"""GPU checks of actual T6 update helper and truncated graph."""
import random
import unittest
from unittest.mock import patch
import torch
from src.ponder import load_model
from src.per_round import round_forward, update_batch

class PerRoundUpdate(unittest.TestCase):
    def tearDown(self):
        torch.cuda.empty_cache()

    def test_updates_and_first_change(self):
        torch.manual_seed(42)
        model = load_model("ponder").train()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        x = torch.randint(model.config.vocab_size, (1,64), device="cuda")
        parameter = model.embed_out.weight
        for k in (2,3):
            changes = []
            original = opt.step
            def step(*args, **kwargs):
                before = parameter.detach().clone()
                result = original(*args, **kwargs)
                changes.append(not torch.equal(before, parameter))
                return result
            with patch.object(opt, "step", side_effect=step) as spy:
                result = update_batch(model, [x]*16, k, opt, 1.)
            self.assertEqual(spy.call_count, k)
            self.assertTrue(changes[0])
            self.assertEqual(len(result["round_ce"]), k)
            self.assertTrue(all(torch.isfinite(torch.tensor(result[n])).all()
                                for n in ("round_ce","round_grad_norm","final_prediction_ce")))

    def test_attribution_and_detach(self):
        torch.manual_seed(42)
        model = load_model("ponder").eval()
        x = torch.randint(model.config.vocab_size,(1,64),device="cuda")
        original = model.gpt_neox.forward
        def run(k, perturb=False, capture=False):
            calls = 0
            states = []
            def wrapped(*args, **kwargs):
                nonlocal calls
                out = original(*args, **kwargs)
                calls += 1
                if calls == 2 and capture:
                    # Force a live prefix graph: helper must explicitly detach.
                    out.last_hidden_state = out.last_hidden_state.detach().requires_grad_()
                    out.last_hidden_state.retain_grad()
                    states.append(out.last_hidden_state)
                if calls == 3 and perturb:
                    out.last_hidden_state = out.last_hidden_state + delta
                return out
            with patch.object(model.gpt_neox,"forward",side_effect=wrapped):
                loss = round_forward(model,x,k)
            return loss, states
        delta = torch.randn(1,128,model.config.hidden_size,device="cuda")*10
        for k in (1,2,3):
            a,_=run(k)
            b,_=run(k,True)
            if k==1:
                torch.testing.assert_close(a,b,atol=0,rtol=0)
            elif k==2:
                self.assertGreater(abs((a-b).item()),1e-5)
        loss, states = run(2,capture=True)
        model.zero_grad(set_to_none=True)
        loss.backward()
        self.assertIsNone(states[0].grad)
        self.assertIsNotNone(model.embed_out.weight.grad)
        self.assertGreater(model.embed_out.weight.grad.norm().item(),0)

    def test_schedule(self):
        rng=random.Random(42)
        ks=[rng.choice([2,3]) for _ in range(600)]
        self.assertEqual([ks.count(2),ks.count(3)],[306,294])
        self.assertEqual(sum(ks),1494)
