"""Check addition sampling, actual Pythia token boundaries, and vendored forwards."""
import random
import unittest
from collections import Counter

import torch
from transformers import AutoTokenizer, GPTNeoXConfig

from src import looped
from src.addition import build_test_set, encode_example, format_example, greedy_answer, sample_operands
from src.train_addition import MODEL, REVISION, make_batch, random_model


class AdditionDataTests(unittest.TestCase):
    def test_sample_operands(self):
        # Replay the exact draws to check each sampled digit count independently.
        rng, reference = random.Random(42), random.Random(42)
        counts = [Counter(), Counter()]
        zero_seen = [False, False]
        for _ in range(12000):
            operands = sample_operands(rng, 3)
            for side, value in enumerate(operands):
                digits = reference.randint(1, 3)
                low = 0 if digits == 1 else 10 ** (digits-1)
                expected = reference.randint(low, 10 ** digits-1)
                self.assertEqual(value, expected)
                self.assertEqual(len(str(value)), digits)
                self.assertGreaterEqual(value, low)
                counts[side][digits] += 1
                zero_seen[side] |= value == 0
        self.assertTrue(all(zero_seen))
        for count in counts:
            for digits in (1, 2, 3):
                self.assertLess(abs(count[digits] / 12000 - 1/3), .025)
        self.assertEqual(format_example(123, 456), "123+456=579")

    def test_test_set(self):
        examples = build_test_set()
        self.assertEqual(examples, build_test_set())
        self.assertNotEqual(examples, build_test_set(seed=1235))
        self.assertEqual(len(examples), 3000)
        self.assertEqual(Counter(len(str(e["a"])) for e in examples),
                         {digits: 500 for digits in range(1, 7)})
        for i, example in enumerate(examples):
            digits = i // 500 + 1
            self.assertEqual(len(str(example["a"])), digits)
            self.assertEqual(len(str(example["b"])), digits)
            self.assertEqual(example["answer"], example["a"]+example["b"])


class AdditionTokenizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)

    def test_encode_example(self):
        tok = self.tok
        prompt = tok.encode("123+456=", add_special_tokens=False)
        answer = tok.encode("579", add_special_tokens=False)
        print("123+456=579 token split:", tok.tokenize("123+456=579"),
              "prompt ids:", prompt, "answer ids:", answer, "EOS:", tok.eos_token_id,
              flush=True)
        # Observed from the actual tokenizer.json at the pinned revision.
        self.assertEqual(tok.tokenize("123+456=579"), ["123", "+", "456", "=", "5", "79"])
        self.assertEqual(prompt, [10683, 12, 25133, 30])
        self.assertEqual(answer, [22, 2787])
        self.assertEqual(tok.eos_token_id, 0)
        ids, labels = encode_example(tok, 123, 456)
        self.assertEqual(ids, prompt + answer + [tok.eos_token_id])
        self.assertEqual(labels, [-100] * len(prompt) + answer + [tok.eos_token_id])
        self.assertEqual(tok.decode(ids[:-1]), "123+456=579")
        for a, b in ((0, 0), (999, 999), (123456, 654321)):
            ids, labels = encode_example(tok, a, b)
            n = len(tok.encode(f"{a}+{b}=", add_special_tokens=False))
            self.assertEqual(labels[:n], [-100]*n)
            self.assertEqual(labels[n:], ids[n:])
            self.assertEqual(ids[-1], tok.eos_token_id)
            self.assertEqual(tok.decode(ids[n:-1]), str(a+b))

    def test_right_padding(self):
        x, labels = make_batch(self.tok, random.Random(42), 32, 3, "cpu")
        reference = random.Random(42)
        pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id
        for row in range(32):
            ids, targets = encode_example(self.tok, *sample_operands(reference, 3))
            self.assertEqual(x[row, :len(ids)].tolist(), ids)
            self.assertEqual(labels[row, :len(ids)].tolist(), targets)
            self.assertTrue(torch.all(x[row, len(ids):] == pad))
            self.assertTrue(torch.all(labels[row, len(ids):] == -100))

    def test_greedy_answer(self):
        model = tiny_model(len(self.tok))
        self.assertIsInstance(greedy_answer(model, self.tok, 123, 456, max_new_tokens=2), str)
        self.assertFalse(model.training)


    def test_evaluate_schema(self):
        from src.evaluate_addition import evaluate

        results = evaluate(tiny_model(len(self.tok)), self.tok,
                           build_test_set(seed=1234, per_length=2))
        self.assertEqual(set(results["per_length"]), {str(i) for i in range(1, 7)})
        self.assertEqual(results["n_total"], 12)
        for key in ("in_distribution_acc", "ood_acc", "overall_acc"):
            self.assertGreaterEqual(results[key], 0.0)
            self.assertLessEqual(results[key], 1.0)
        for row in results["per_length"].values():
            self.assertEqual(row["n"], 2)
            self.assertGreaterEqual(row["acc"], 0.0)
            self.assertLessEqual(row["acc"], 1.0)


def tiny_model(vocab_size=128):
    torch.manual_seed(42)
    config = GPTNeoXConfig(vocab_size=vocab_size, hidden_size=32, intermediate_size=64,
                          num_hidden_layers=6, num_attention_heads=4,
                          max_position_embeddings=128, rotary_pct=.5,
                          attention_dropout=0., hidden_dropout=0.)
    return looped.configure(random_model(config), 2, 1, start=2)


class AdditionModelTests(unittest.TestCase):
    def test_r1_parity(self):
        model = tiny_model().eval()
        x = torch.randint(0, model.config.vocab_size, (2, 16))
        labels = x.clone()
        labels[:, :5] = -100
        labels[0, -3:] = -100
        with torch.no_grad():
            expected = model(input_ids=x, labels=labels, use_cache=False)
            actual = looped.forward(model, x, labels=labels)
        torch.testing.assert_close(actual.logits, expected.logits, atol=1e-5, rtol=0)
        torch.testing.assert_close(actual.loss, expected.loss, atol=1e-5, rtol=0)


if __name__ == "__main__":
    unittest.main()
