"""Small-input evaluation scan contracts; no training or full-split inference."""
import argparse
import unittest
import torch
import torch.nn.functional as F
from src.masked import forward, forward_eval
from src.ponder import load_model
from scripts.eval_kscan import final_ce_sum, parse_rounds, validation_chunks


class KScanHelpers(unittest.TestCase):
    def test_rounds_parser(self):
        self.assertEqual(parse_rounds("0,1,2,3,4,5"), list(range(6)))
        for value in ("6", "a", "", "-1", "2.5", "True", "1,", "1,1"):
            with self.subTest(value=value), self.assertRaises(argparse.ArgumentTypeError):
                parse_rounds(value)

    def test_invalid_eval_rounds(self):
        # Validation must happen before accessing model or input.
        for r in (-1, 6, 2.5, "3", True):
            with self.subTest(r=r), self.assertRaises(ValueError):
                forward_eval(None, None, r)

    def test_training_rejects_other_rounds(self):
        for r in (0, 1, 4, 5):
            with self.subTest(r=r), self.assertRaisesRegex(
                    ValueError, "^training/evaluation iterations must be 2 or 3$"):
                forward(None, None, r)

    def test_final_ce_sum(self):
        torch.manual_seed(42)
        logits = torch.randn(2, 8, 13).bfloat16()
        x = torch.randint(0, 13, (2, 8))
        total, count = final_ce_sum(logits, x)
        reference = F.cross_entropy(logits[:, :-1].float().reshape(-1, 13),
                                    x[:, 1:].reshape(-1), reduction="sum").item()
        self.assertEqual(count, 14)
        self.assertEqual(total/count, reference/count)

    def test_validation_order(self):
        # Calls the real chunks("validation") twice, suppressing its writes.
        first, second = validation_chunks(), validation_chunks()
        self.assertEqual(tuple(first.shape), (977, 256))
        self.assertEqual(tuple(second.shape), (977, 256))
        self.assertTrue(torch.equal(first, second))


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class KScanForward(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.model = load_model("ponder").eval()
        self.x = torch.randint(0, self.model.config.vocab_size, (1, 16), device="cuda")

    def tearDown(self):
        del self.x, self.model
        torch.cuda.empty_cache()

    @torch.no_grad()
    def test_all_eval_rounds_and_zero_changes_logits(self):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            reference = forward(self.model, self.x, 3, "D0").logits
            for r in range(6):
                with self.subTest(r=r):
                    out = forward_eval(self.model, self.x, r)
                    self.assertEqual(out.logits.shape, reference.shape)
                    self.assertTrue(torch.isfinite(out.logits).all())
                    self.assertTrue(torch.isfinite(out.loss))
                    self.assertTrue(torch.isfinite(out.final_loss))
                    self.assertEqual(out.round_losses, ())
                    self.assertEqual(out.round_logits, ())
                    if r == 0:
                        self.assertGreater((out.logits.float()-reference.float()).abs().max().item(), 0.)
                    del out
        del reference

    @torch.no_grad()
    def test_shared_forward_parity_and_training_losses(self):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for condition in ("D0", "D1", "D2"):
                for r in (2, 3):
                    with self.subTest(condition=condition, r=r):
                        train = forward(self.model, self.x, r, condition)
                        ev = forward_eval(self.model, self.x, r, condition)
                        self.assertTrue(torch.isfinite(train.loss))
                        difference = (train.logits.float()-ev.logits.float()).abs().max().item()
                        self.assertLess(difference, 1e-6)
                        torch.testing.assert_close(train.loss, ev.loss, atol=0, rtol=0)
                        del train, ev

    @torch.no_grad()
    def test_repeatability_after_other_rounds(self):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            first = forward_eval(self.model, self.x, 3).logits
            other = forward_eval(self.model, self.x, 5)
            del other
            second = forward_eval(self.model, self.x, 3).logits
        self.assertTrue(torch.equal(first, second))
        del first, second

    @torch.no_grad()
    def test_labels_and_round_outputs(self):
        labels = self.x.clone()
        labels[:, :2] = -100
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for r in (0, 2, 3, 5):
                with self.subTest(r=r):
                    out = forward_eval(self.model, self.x, r, labels=labels,
                                       intermediate_supervision=True, return_round_logits=True)
                    self.assertEqual(len(out.round_logits), r)
                    self.assertEqual(len(out.round_losses), r)
                    expected = torch.stack(out.round_losses).mean() if r else out.final_loss
                    torch.testing.assert_close(out.loss, expected, atol=0, rtol=0)
                    reference = F.cross_entropy(
                        out.logits[:, :-1].contiguous().view(-1, out.logits.shape[-1]),
                        labels[:, 1:].contiguous().view(-1))
                    torch.testing.assert_close(out.final_loss, reference, atol=0, rtol=0)
                    reference_fp32 = F.cross_entropy(
                        out.logits[:, :-1].contiguous().float().view(-1, out.logits.shape[-1]),
                        labels[:, 1:].contiguous().view(-1))
                    # This bf16 precision gap is not a bug; evaluation uses the script's outer FP32 CE.
                    self.assertLess((out.final_loss - reference_fp32).abs().item(), 0.05)
                    for logits, loss in zip(out.round_logits, out.round_losses):
                        self.assertEqual(logits.shape, out.logits.shape)
                        self.assertTrue(torch.isfinite(loss))
                    del out
        del labels


if __name__ == "__main__":
    unittest.main()
