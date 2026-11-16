import json
import unittest
from pathlib import Path
from types import SimpleNamespace
import torch
from src.ponder import load_model
from src import looped


class LoopedTests(unittest.TestCase):
    def test_flops_table(self):
        rows = looped.candidates()
        chosen = looped.selected()
        self.assertEqual((chosen["block_size"], chosen["repetitions"]), (3, 3))
        self.assertEqual(looped.flops(1, 1), 90832896)
        self.assertEqual(looped.flops(3, 3), 130154496)
        self.assertEqual(min(r["relative_error"] for r in rows), chosen["relative_error"])
        self.assertLess(chosen["relative_error"], .1)
        c = json.loads(Path("configs/exp05_t4.json").read_text())
        self.assertEqual(c["loop"], dict(block_size=chosen["block_size"], repetitions=chosen["repetitions"]))
        Path("runs/exp05-flops-table.json").write_text(json.dumps(
            dict(t0_flops=90832896,t1_flops=looped.T1_FLOPS,convention=looped.CONVENTION,
                 candidates=rows,selected=chosen,tie_break="minimum error, then smaller R, then smaller block"),indent=2)+"\n")

    def test_shared_weights_and_r1_parity(self):
        torch.manual_seed(42)
        model = load_model("vanilla").eval()
        parameter_ids = [id(p) for p in model.parameters()]
        x = torch.randint(0, model.config.vocab_size, (1,64), device="cuda")
        looped.configure(model, 3, 1)
        with torch.no_grad():
            expected = model(input_ids=x, labels=x, use_cache=False)
            actual = looped.forward(model,x,labels=x)
        torch.testing.assert_close(actual.logits, expected.logits, atol=1e-5, rtol=0)
        torch.testing.assert_close(actual.loss, expected.loss, atol=1e-5, rtol=0)
        looped.configure(model, 3, 3)
        schedule = looped.layer_schedule(model)
        self.assertEqual(len(schedule), 12)
        for i in range(3):
            for r in range(3):
                self.assertIs(schedule[3+3*r+i], model.gpt_neox.layers[3+i])
        self.assertEqual(parameter_ids, [id(p) for p in model.parameters()])
        self.assertEqual(sum(p.numel() for p in model.parameters()), 70426624)
        del model, expected, actual
        torch.cuda.empty_cache()

    def test_checkpoint_config_schedule(self):
        for start, expected in ((2, [0, 1] + [2, 3] * 4 + [4, 5]),
                                (None, [0, 1, 2, 3] + [4, 5] * 4)):
            with self.subTest(start=start):
                config = SimpleNamespace(num_latent_thoughts=0, scale_embeds=False,
                                         loop_block_size=2, loop_repetitions=4)
                if start is not None:
                    config.loop_block_start = start
                model = SimpleNamespace(config=config,
                                        gpt_neox=SimpleNamespace(layers=list(range(6))))
                looped.configure(model, config.loop_block_size, config.loop_repetitions,
                                 start=getattr(config, "loop_block_start", None))
                self.assertEqual(looped.layer_schedule(model), tuple(expected))

    def test_middle_block_schedule(self):
        torch.manual_seed(42)
        model = load_model("vanilla").eval()
        parameter_ids = [id(p) for p in model.parameters()]
        layers = model.gpt_neox.layers
        looped.configure(model, 2, 4, start=2)
        schedule = looped.layer_schedule(model)
        self.assertEqual(len(schedule), 12)
        for actual, index in zip(schedule, [0, 1] + [2, 3] * 4 + [4, 5]):
            self.assertIs(actual, layers[index])
        self.assertEqual(parameter_ids, [id(p) for p in model.parameters()])
        self.assertEqual(sum(p.numel() for p in model.parameters()), 70426624)
        self.assertEqual(looped.flops(2, 4), 130154496)
        for invalid in (-1, 5, 2.0, True, "2"):
            with self.assertRaises(ValueError):
                looped.configure(model, 2, 4, start=invalid)
        looped.configure(model, 3, 3)
        self.assertEqual(model.config.loop_block_start, 3)
        expected = looped.layer_schedule(model)
        del model.config.loop_block_start
        self.assertEqual(looped.layer_schedule(model), expected)
        del model
        torch.cuda.empty_cache()

    def test_two_optimizer_steps(self):
        torch.manual_seed(42)
        model = looped.configure(load_model("vanilla"),3,3).train()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(.9,.95),weight_decay=.01)
        for step in range(2):
            x = torch.randint(0,model.config.vocab_size,(1,64),device="cuda")
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda",dtype=torch.bfloat16):
                out = looped.forward(model,x,labels=x)
            self.assertTrue(torch.isfinite(out.loss))
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            opt.step()
            print("loop smoke",step,out.loss.item(),flush=True)
        del model,opt,out
        torch.cuda.empty_cache()

    def test_causal_prefix(self):
        torch.manual_seed(42)
        model = looped.configure(load_model("vanilla"),3,3).eval()
        x = torch.randint(0,model.config.vocab_size,(1,16),device="cuda")
        with torch.no_grad():
            full = looped.forward(model,x).logits
            prefix = looped.forward(model,x[:,:8],last_only=True).logits
            changed = x.clone()
            changed[:,8:] = 0
            other = looped.forward(model,changed).logits
        torch.testing.assert_close(full[:,:8],other[:,:8],atol=1e-5,rtol=0)
        # Different GEMM/SDPA shapes accumulate small FP32 roundoff through 12 calls.
        torch.testing.assert_close(full[:,7:8],prefix,atol=1e-3,rtol=1e-6)
        del model,full,prefix,other
        torch.cuda.empty_cache()
