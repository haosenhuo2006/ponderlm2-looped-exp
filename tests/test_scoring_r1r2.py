"""Frozen R1 / UTF-8 byte R2 definitions, checked in both MC modules."""
import unittest
import torch
import torch.nn.functional as F
from src import downstream, evalsets26


class ScoringTests(unittest.TestCase):
    def test_context_and_correctness_flip(self):
        for module in (downstream, evalsets26):
            for context in (0., 10., 10000.):
                scores = module.score_options([2., 2.5], [1, 4],
                                              [context+2., context+2.5], [1, 4])
                self.assertEqual(scores['r1'], 0)
            scores = module.score_options([2., 2.5], [1, 4], [12., 12.5], [1, 4])
            self.assertEqual(scores, dict(r1=0, r2=1, r3=1, raw=0))
            r1_correct, r3_correct = scores['r1'] == 0, scores['r3'] == 0
            self.assertTrue(r1_correct)
            self.assertFalse(r3_correct)

    def test_no_long_answer_bias_and_ties(self):
        for module in (downstream, evalsets26):
            # Each byte count equals its token count: all R2 scores tie at 1.
            self.assertEqual(module.score_options([1., 5., 6., 7.], [1, 5, 6, 7]),
                             dict(r1=0, r2=0))
            self.assertEqual(module.score_options([2., 2.], [4, 4]), dict(r1=0, r2=0))

    def test_utf8_bytes(self):
        for module in (downstream, evalsets26):
            self.assertEqual(module.score_options([4., 8.], [4, 16]), dict(r1=0, r2=1))
            answers = ['cafe', 'café']
            # Equal A divided by 5 beats equal A divided by 4.
            self.assertEqual(module.score_options([4., 4.], [len(a.encode('utf-8')) for a in answers])['r2'], 1)
            answer = 'éé'  # 2 characters, 4 UTF-8 bytes, one fake token.
            self.assertEqual((len(answer), len(answer.encode('utf-8'))), (2, 4))
            self.assertEqual(module.score_options([3., 2.], [4, 2])['r2'], 0)
            self.assertEqual(module.score_options([3., 2.], [2, 2])['r2'], 1)
            self.assertEqual(module.score_options([3., 2.], [1, 2])['r2'], 1)

    def test_boundary_and_total_decomposition(self):
        class Tokenizer:
            def __call__(self, text, **kwargs):
                return dict(input_ids=[0, 1, 2, 3, 4],
                            offset_mapping=[(0, 1), (1, 3), (3, 7), (7, 8), (8, 9)])
        enc = Tokenizer()('fake text', return_offsets_mapping=True)
        prefix = 'abcde'
        ids = enc['input_ids']
        start = next(i for i, (_, end) in enumerate(enc['offset_mapping']) if end > len(prefix))
        self.assertEqual(start, 2)
        self.assertLess(enc['offset_mapping'][start][0], len(prefix))
        self.assertEqual(evalsets26.answer_token_count(enc, len(prefix)), len(ids)-start)
        logits = torch.zeros(1, len(ids), 6, dtype=torch.bfloat16)
        logits[0, 0, 5] = 100  # Huge prompt loss.
        logits[0, 1, 2] = 3
        logits[0, 2, 3] = 2
        logits[0, 3, 4] = 1
        losses = F.cross_entropy(logits[0, :-1].float(), torch.tensor(ids[1:]), reduction='none')
        expected = sum(F.cross_entropy(logits[0, i-1:i].float(), torch.tensor([ids[i]])).item()
                       for i in range(start, len(ids)))
        for module in (downstream, evalsets26):
            actual = module.answer_span_nll(logits, ids, start)
            self.assertAlmostEqual(actual, expected, delta=1e-5)
            self.assertAlmostEqual(losses.sum().item(), losses[:start-1].sum().item()+actual, delta=1e-5)
            changed = logits.clone()
            changed[0, 0] = 0
            self.assertEqual(module.answer_span_nll(changed, ids, start), actual)
            with self.assertRaises(ValueError):
                module.answer_span_nll(logits, ids, 0)


if __name__ == '__main__':
    unittest.main()
