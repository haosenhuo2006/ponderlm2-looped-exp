"""Masked optimizer smoke, direct-edge loss effects, and seeded K schedule."""
import random
import unittest
from unittest.mock import patch
import torch
from src.ponder import load_model, forward as official
from src.masked import forward


class MaskedTraining(unittest.TestCase):
    def test_k_schedule(self):
        rng = random.Random(42)
        ks = [rng.choice([2, 3]) for _ in range(600)]
        self.assertEqual(set(ks), {2, 3})
        self.assertEqual(ks.count(2), 306)
        self.assertEqual(ks.count(3), 294)

    def test_optimizer_and_all_pass_gradients(self):
        for condition in ("D1", "D2"):
            torch.manual_seed(42)
            model = load_model("ponder").train()
            opt = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(.9,.95), weight_decay=.01)
            for step in range(2):
                states = []
                def retain(module, args, out):
                    out.last_hidden_state.retain_grad()
                    states.append(out.last_hidden_state)
                hook = model.gpt_neox.register_forward_hook(retain)
                x = torch.randint(0, model.config.vocab_size, (1,64), device="cuda")
                opt.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = forward(model, x, 2+step, condition)
                hook.remove()
                self.assertTrue(torch.isfinite(out.loss))
                out.loss.backward()
                self.assertEqual(len(states), 4+step)
                for index, state in enumerate(states):
                    self.assertIsNotNone(state.grad)
                    self.assertTrue(torch.isfinite(state.grad).all())
                    if condition == "D1" and index < len(states)-2:
                        # No previous h edges: token states are independent of h.
                        self.assertEqual(state.grad.abs().sum().item(), 0)
                    else:
                        self.assertGreater(state.grad.abs().sum().item(), 0)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                opt.step()
                print(condition, step, out.loss.item(), flush=True)
                del states, out
            del model, opt
            torch.cuda.empty_cache()

    def test_training_loss_direct_mask_effect(self):
        # Isolate a direct edge in the FINAL pass. Whole-loop D2 independence is
        # incorrect: permitted x->h->later-query paths carry history deliberately.
        torch.manual_seed(42)
        model = load_model("ponder").eval()
        layers = model.gpt_neox.layers
        model.gpt_neox.layers = torch.nn.ModuleList([layers[0]])
        x = torch.randint(0, model.config.vocab_size, (1,8), device="cuda")
        labels = x.clone()
        labels[:, :2] = -100  # Score only later pairs, never the perturbed pair.
        original = model.prepare_gpt_inputs_for_stages
        delta = torch.randn(1,model.config.hidden_size,device="cuda")*10
        try:
            for condition, key in (("D1",1),("D2",0)):
                def loss(mask, perturb):
                    calls = 0
                    def prepare(*args, **kwargs):
                        nonlocal calls
                        inp, pos, attn = original(*args, **kwargs)
                        calls += 1
                        if calls == 3 and perturb:  # K=2 then final pass
                            inp = inp.clone()
                            inp[:,key] = inp[:,key]+delta
                        return inp, pos, attn
                    with patch.object(model, "prepare_gpt_inputs_for_stages", side_effect=prepare):
                        return forward(model,x,2,mask,labels).loss
                a,b = loss(condition,False),loss(condition,True)
                torch.testing.assert_close(a,b,atol=1e-6,rtol=0)
                a,b = loss("D0",False),loss("D0",True)
                self.assertGreater(abs((a-b).item()),1e-6)
        finally:
            model.gpt_neox.layers = layers
            del model
            torch.cuda.empty_cache()

    def test_manual_d0_training_parity(self):
        torch.manual_seed(42)
        model = load_model("ponder").eval()
        x = torch.randint(0,model.config.vocab_size,(1,8),device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            a,b = forward(model,x,3,"D0"),official(model,x,3)
        torch.testing.assert_close(a.logits,b.logits,atol=1e-6,rtol=0)
        torch.testing.assert_close(a.loss,b.loss,atol=1e-6,rtol=0)
        del a,b,model
        torch.cuda.empty_cache()
