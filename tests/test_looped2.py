"""CPU checks for the looped2 LM reproduction, using fresh tiny weights."""

from dataclasses import replace
import math
import random
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn
from transformers import GPTNeoXConfig

from src import looped2
from src.addition import sample_operands, sample_operands_strict3
from src.train_addition import random_model


def tiny_model(**kwargs):
    return random_model(GPTNeoXConfig(
        hidden_size=64, num_hidden_layers=6, intermediate_size=128,
        num_attention_heads=4, vocab_size=512, max_position_embeddings=128,
        num_latent_thoughts=0, scale_embeds=False, use_cache=False,
        attn_implementation="sdpa", **kwargs))


class Looped2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        threads = torch.get_num_threads()
        cls.addClassCleanup(torch.set_num_threads, threads)
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(42)
        self.cfg = looped2.Looped2Config.from_widths(6, 2, 2, 2)
        self.x = torch.randint(512, (2, 12))

    def configured_model(self, **kwargs):
        return looped2.apply_config(tiny_model(**kwargs), self.cfg).eval()

    @torch.no_grad()
    def test_prelude_injection_matches_reference_each_round(self):
        model = self.configured_model()
        backbone = model.gpt_neox
        e = backbone.emb_dropout(backbone.embed_in(self.x))
        cache = torch.arange(self.x.size(1))
        positions = cache.unsqueeze(0).expand_as(self.x)
        mask = backbone._update_causal_mask(torch.ones_like(self.x), e, cache, None, False)
        args = dict(attention_mask=mask, position_ids=positions, cache_position=cache,
                    position_embeddings=backbone.rotary_emb(e, positions))
        for index in (0, 1):
            e = backbone.layers[index](e, **args)[0]

        def sandwich(h):
            for index in (2, 3):
                layer = backbone.layers[index]
                attn = layer.attention(layer.input_layernorm(h), **args)[0]
                h = layer.looped2_n2(h + layer.post_attention_dropout(attn))
                mlp = layer.mlp(layer.post_attention_layernorm(h))
                h = layer.looped2_n4(h + layer.post_mlp_dropout(mlp))
            return h

        def finish(h):
            h = backbone.final_layer_norm(h)
            for index in (4, 5):
                h = backbone.layers[index](h, **args)[0]
            return model.embed_out(backbone.final_layer_norm(h))

        zero = torch.zeros_like(e)
        one = looped2.forward(model, self.x, cfg=self.cfg, mode="eval", r_override=1, s0=zero).logits
        torch.testing.assert_close(one, finish(sandwich(e)), atol=1e-5, rtol=0)
        self.assertFalse(torch.allclose(one, finish(sandwich(zero))))
        # A1: a second round adds e again, to the previous core output.
        two = looped2.forward(model, self.x, cfg=self.cfg, mode="eval", r_override=2, s0=zero).logits
        torch.testing.assert_close(two, finish(sandwich(sandwich(e) + e)), atol=1e-5, rtol=0)
        self.assertFalse(torch.allclose(two, finish(sandwich(sandwich(e)))))
        # A one-round example with s0=e gives a core input of 2e.
        doubled = looped2.forward(model, self.x, cfg=self.cfg, mode="eval", r_override=1, s0=e).logits
        torch.testing.assert_close(doubled, finish(sandwich(2 * e)), atol=1e-5, rtol=0)
        self.assertFalse(torch.allclose(doubled, one))

    @torch.no_grad()
    def test_fixed_s0_seed_gives_identical_logits(self):
        model = self.configured_model()
        first = looped2.forward(model, self.x, cfg=self.cfg, mode="eval", s0_seed=0)
        second = looped2.forward(model, self.x, cfg=self.cfg, mode="eval", s0_seed=0)
        self.assertTrue(torch.equal(first.logits, second.logits))

    @torch.no_grad()
    def test_different_s0_seeds_change_logits(self):
        model = self.configured_model()
        first = looped2.forward(model, self.x, cfg=self.cfg, mode="eval", s0_seed=0)
        second = looped2.forward(model, self.x, cfg=self.cfg, mode="eval", s0_seed=1)
        self.assertFalse(torch.allclose(first.logits, second.logits))

    @torch.no_grad()
    def test_s0_truncated_scale_is_independent_of_embedding_scale(self):
        sample = looped2.sample_s0((200000,), "cpu", torch.float32, torch.Generator().manual_seed(0))
        self.assertTrue(0.60 <= sample.std().item() <= 0.65)
        self.assertLessEqual(sample.abs().max().item(), 3 * math.sqrt(2 / 5))
        model = self.configured_model()
        samples = []

        def capture(*args):
            state = draw(*args)
            samples.append(state)
            return state

        draw = looped2.sample_s0
        with patch.object(looped2, "sample_s0", side_effect=capture):
            for emb_scale in (1.0, math.sqrt(64)):
                cfg = replace(self.cfg, emb_scale=emb_scale)
                looped2.forward(model, self.x, cfg=cfg, mode="eval", s0_seed=0)
        self.assertTrue(torch.equal(*samples))

    @torch.no_grad()
    def test_eval_depth_can_exceed_training_cap_and_reuses_s0(self):
        model = self.configured_model()
        cfg = replace(self.cfg, cap=4)
        shape = (*self.x.shape, model.config.hidden_size)
        samples = [looped2.sample_s0(shape, "cpu", torch.float32, torch.Generator().manual_seed(5))
                   for _ in range(2)]
        self.assertTrue(torch.equal(*samples))
        for r, s0 in zip((1, 8), samples):
            with self.subTest(r=r):
                record = {}
                seeded = looped2.forward(model, self.x, cfg=cfg, mode="eval", r_override=r,
                                         s0_seed=5, record=record)
                explicit = looped2.forward(model, self.x, cfg=cfg, mode="eval", r_override=r, s0=s0)
                self.assertEqual(record, dict(nkr=(0, 0, r), r=r))
                self.assertEqual(seeded.logits.shape, (*self.x.shape, 512))
                self.assertTrue(torch.equal(seeded.logits, explicit.logits))

    def test_truncated_backprop_reaches_prelude_and_core(self):
        model = self.configured_model().train()
        for autocast in (False, True):
            with self.subTest(autocast=autocast):
                model.zero_grad(set_to_none=True)
                record = {}
                with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
                    out = looped2.forward(model, self.x, labels=self.x, cfg=self.cfg,
                                          r_override=4, k_override=2, s0_seed=0, record=record)
                self.assertEqual(record["nkr"], (2, 2, 4))
                out.loss.backward()
                for layer in (0, 2):
                    grad = model.gpt_neox.layers[layer].attention.query_key_value.weight.grad
                    self.assertIsNotNone(grad)
                    self.assertGreater(grad.norm().item(), 0)

    @torch.no_grad()
    def test_registered_norms_survive_checkpoint_round_trip(self):
        for tied, safetensors in ((False, True), (True, True), (False, False)):
            with self.subTest(tied=tied, safetensors=safetensors), tempfile.TemporaryDirectory() as path:
                model = self.configured_model(tie_word_embeddings=tied)
                layer = model.gpt_neox.layers[2]
                weight = layer.looped2_n2.weight
                weight.add_(0.2)
                layer.looped2_n4.bias.add_(0.1)
                looped2.apply_config(model, self.cfg)
                self.assertIs(layer.looped2_n2.weight, weight)
                self.assertIn(id(weight), {id(p) for p in model.parameters()})
                expected = looped2.forward(model, self.x, cfg=self.cfg, mode="eval", s0_seed=7).logits
                model.save_pretrained(path, safe_serialization=safetensors)
                restored = looped2.load_looped2_checkpoint(path, device="cpu")
                restored_cfg = looped2.config_from_checkpoint(restored.config)
                self.assertEqual(restored_cfg, self.cfg)
                self.assertFalse(restored.training)
                actual = looped2.forward(restored, self.x, cfg=restored_cfg, mode="eval", s0_seed=7).logits
                torch.testing.assert_close(actual, expected, atol=1e-5, rtol=0)

    def test_flops_and_capped_sampler_expectations(self):
        h, ff, vocab = 768, 3072, 50304
        layer = 8 * h ** 2 + 4 * h * ff + 4 * h * 128
        head = 2 * h * vocab
        self.assertEqual(looped2.layer_flops_per_token(h, ff, vocab), (layer, head))
        expected = (6 + 6 * 4) * layer + head + 2 * ((6 + 6 * 2) * layer + head)
        self.assertEqual(looped2.train_flops_per_token(4, 2, 3, 6, 3, h, ff, vocab), expected)
        r, k = looped2.expected_rk()
        self.assertAlmostEqual(r, 4.9324, delta=0.01)
        self.assertAlmostEqual(k, 1.9423, delta=0.01)
        self.assertAlmostEqual(looped2.expected_flops_ratio(3, 6, 3, h, ff, vocab), 1.6721, delta=0.001)
        generator = torch.Generator().manual_seed(42)
        draws = [looped2.sample_nkr(generator) for _ in range(20000)]
        self.assertTrue(all(1 <= r <= 12 and k == min(2, r) and n + k == r for n, k, r in draws))
        self.assertAlmostEqual(sum(r for _, _, r in draws) / len(draws), 4.93, delta=0.06)
        self.assertAlmostEqual(sum(k for _, k, _ in draws) / len(draws), 1.94, delta=0.04)
        model = self.configured_model()
        resumed = torch.Generator()
        resumed.set_state(generator.get_state())
        expected_nkr = looped2.sample_nkr(resumed)
        s0 = looped2.sample_s0((*self.x.shape, 64), "cpu", torch.float32, resumed)
        record = {}
        with torch.no_grad():
            actual = looped2.forward(model, self.x, cfg=self.cfg, rng=generator, record=record)
            expected = looped2.forward(model, self.x, cfg=self.cfg, s0=s0,
                                        r_override=expected_nkr[2], k_override=expected_nkr[1])
        self.assertEqual(record["nkr"], expected_nkr)
        self.assertTrue(torch.equal(actual.logits, expected.logits))
        self.assertTrue(torch.equal(generator.get_state(), resumed.get_state()))

    @torch.no_grad()
    def test_plain_forward_matches_native_unscaled_logits_and_loss(self):
        model = tiny_model().eval()
        expected = model(input_ids=self.x, labels=self.x)
        actual = looped2.plain_forward(model, self.x, labels=self.x, emb_scale=1.0)
        torch.testing.assert_close(actual.logits, expected.logits, atol=1e-5, rtol=0)
        torch.testing.assert_close(actual.loss, expected.loss, atol=1e-5, rtol=0)

    @torch.no_grad()
    def test_scratch_initialization_depth_scale_and_tied_parameter_deduplication(self):
        model = self.configured_model(tie_word_embeddings=True)
        shared = model.gpt_neox.embed_in.weight
        self.assertIs(shared, model.embed_out.weight)
        with patch.object(nn.init, "trunc_normal_", wraps=nn.init.trunc_normal_) as init:
            looped2.reinitialize_from_scratch(model, l_eff=12)
        self.assertEqual(sum(call.args[0] is shared for call in init.call_args_list), 1)
        for module in model.modules():
            bias = getattr(module, "bias", None)
            if isinstance(bias, nn.Parameter):
                self.assertEqual(torch.count_nonzero(bias).item(), 0)
            if isinstance(module, nn.LayerNorm):
                self.assertTrue(torch.equal(module.weight, torch.ones_like(module.weight)))
        layer = model.gpt_neox.layers[2]
        std = 1 / math.sqrt(5 * 64 * 12)
        for projection in (layer.attention.dense, layer.mlp.dense_4h_to_h):
            self.assertAlmostEqual(projection.weight.std().item(), std, delta=0.2 * std)
        previous = shared.clone()
        looped2.reinitialize_from_scratch(model, l_eff=12)
        self.assertFalse(torch.equal(shared, previous))

    def test_strict_three_digit_operands_and_legacy_mixed_sampling(self):
        rng = random.Random(42)
        draws = [sample_operands_strict3(rng) for _ in range(200)]
        self.assertTrue(all(100 <= a <= 999 and 100 <= b <= 999 for a, b in draws))
        mixed = [sample_operands(rng, 3) for _ in range(1000)]
        self.assertTrue(any(min(a, b) < 100 for a, b in mixed))

    def test_partition_and_round_depth_validation(self):
        self.assertEqual((self.cfg.prelude, self.cfg.core, self.cfg.coda), ((0, 1), (2, 3), (4, 5)))
        for widths in ((2, 2, 1), (-1, 3, 4), (3, 0, 3)):
            with self.subTest(widths=widths), self.assertRaises(ValueError):
                looped2.Looped2Config.from_widths(6, *widths)
        with self.assertRaises(ValueError):
            replace(self.cfg, coda=(4, 4))
        with self.assertRaises(ValueError):
            replace(self.cfg, coda=(4, -5))
        with self.assertRaises(ValueError):
            replace(self.cfg, cap=0)
        with self.assertRaises(ValueError):
            looped2.sample_nkr(torch.Generator(), cap=0)
        model = self.configured_model()
        with self.assertRaises(ValueError):
            looped2.apply_config(model, replace(self.cfg, coda=(4,)))
        for depth in (dict(r_override=0), dict(r_override=4, k_override=-1),
                      dict(r_override=4, k_override=5)):
            with self.subTest(depth=depth), self.assertRaises(ValueError):
                looped2.forward(model, self.x, cfg=self.cfg, **depth)


if __name__ == "__main__":
    unittest.main()
