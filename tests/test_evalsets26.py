import importlib
import json
import math
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from src import evalsets26 as ev


class Evalsets26Tests(unittest.TestCase):
    def test_mc_argmax_negative_normalized_loss(self):
        scores = ev.score_options([2., 2.5], [1, 4], [12., 12.5], [1, 4])
        self.assertEqual(scores, dict(r1=0, r2=1, r3=1, raw=0))
        self.assertEqual(ev.score_options([2., 2.], [1, 1])["r1"], 0)

    def test_prompt_fixed(self):
        with patch.object(ev, 'config', return_value={'prompt': '{question} Answer: {answer}'}):
            self.assertEqual(ev.prompt('Why blue?', 'scattering'), 'Why blue? Answer: scattering')

    def test_answer_boundary_offsets(self):
        class Tokenizer:
            def __call__(self, text, **kwargs):
                return dict(input_ids=[1, 2, 3, 4], offset_mapping=[(0, 2), (2, 4), (4, 7), (7, 9)])
        enc = Tokenizer()('a b hello', add_special_tokens=False, return_offsets_mapping=True)
        self.assertEqual(ev.answer_token_count(enc, 4), 2)
        self.assertEqual(ev.answer_token_count(enc, 5), 2)
        self.assertEqual(ev.answer_token_count(enc, 9), 0)

    def test_empty_answer_rejected_in_evaluate(self):
        # Exercise the actual MC loop, before any CUDA tensor is constructed.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ev.save(root / 'data-provenance.json', {'tasks': {'sciq': {'status': 'ready'}}})
            ev.save(root / 'data/sciq.json', [{'question': 'Q', 'options': [''], 'gold': 0}])
            cfg = {'seed': 42, 'checkpoints': {'fake': {'mode': 'vanilla', 'path': 'fake'}}}
            tokenizer = lambda *a, **kw: {'input_ids': [1], 'offset_mapping': [(0, 1)]}
            model = SimpleNamespace(eval=lambda: SimpleNamespace())
            with patch.object(ev, 'RUN', root), patch.object(ev, 'config', return_value=cfg), \
                 patch.object(ev, 'prompt', return_value='Q Answer: '), \
                 patch.object(ev.AutoTokenizer, 'from_pretrained', return_value=tokenizer), \
                 patch.object(ev, 'load_model', return_value=model), \
                 patch.object(ev.torch.cuda, 'synchronize'), \
                 patch.object(ev.torch.cuda, 'reset_peak_memory_stats'), \
                 patch.object(ev.torch, 'manual_seed'), \
                 patch.object(ev.torch, 'use_deterministic_algorithms'):
                with self.assertRaises(AssertionError):
                    ev.evaluate('fake', 'sciq', limit=1)

    def test_sciq_mapping_support_excluded(self):
        row = dict(question='Who?', correct_answer='darwin', distractor1='Linnaeus',
                   distractor2='shaw', distractor3='Scopes', support='SECRET SUPPORT')
        built = ev.build_sciq_options(row)
        self.assertEqual(built['options'], ['darwin', 'Linnaeus', 'shaw', 'Scopes'])
        self.assertEqual(built['gold'], 0)
        self.assertEqual(built['support'], row['support'])
        with patch.object(ev, 'config', return_value={'prompt': '{question} Answer: {answer}'}):
            self.assertNotIn(row['support'], ev.prompt(built['question'], built['options'][0]))

    def test_hellaswag_mapping_validation(self):
        row = dict(ctx='context', endings=['a', 'b', 'c', 'd'], label='3')
        self.assertEqual(ev.map_hellaswag(row)['gold'], 3)
        self.assertEqual(ev.map_hellaswag(row)['options'], row['endings'])
        with self.assertRaises(AssertionError):
            ev.map_hellaswag(dict(row, endings=['a', 'b', 'c']))

    def test_piqa_mapping(self):
        row = dict(goal='goal', sol1='first', sol2='second', label=1)
        self.assertEqual(ev.map_piqa(row), dict(question='goal', options=['first', 'second'], gold=1))

    def test_pile_chunk_stream(self):
        class Tokenizer:
            def __call__(self, text, **kwargs):
                return {'input_ids': [int(word) for word in text.split()]}
        def row(start, end, source):
            return dict(text=' '.join(map(str, range(start, end))), meta={'pile_set_name': source})
        stats = {}
        rows = [row(0, 100, 'a'), row(100, 300, 'b'), row(300, 550, 'a')]
        chunks = list(ev.chunk_stream(rows, Tokenizer(), blocks=2, stats=stats))
        self.assertEqual(chunks, [list(range(256)), list(range(256, 512))])
        self.assertEqual(stats, dict(docs_consumed=3, tokens_consumed=512,
                                    leftover_tokens=38, pile_set_counts={'a': 2, 'b': 1}))
        empty = dict(text='', meta={'pile_set_name': 'empty'})
        chunks = list(ev.chunk_stream([empty] + rows, Tokenizer(), blocks=1, stats=stats))
        self.assertEqual(chunks, [list(range(256))])
        self.assertEqual(stats['docs_consumed'], 2)
        self.assertEqual(stats['leftover_tokens'], 44)
        self.assertNotIn('empty', stats['pile_set_counts'])
        stream = ev.chunk_stream([row(0, 255, 'a')], Tokenizer(), blocks=1)
        with self.assertRaisesRegex(ValueError, 'insufficient: 0/1'):
            next(stream)

    def test_protocol_hash_config_change(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'src').mkdir()
            (root / 'src/evalsets26.py').write_text('source')
            cfg = root / 'config.json'
            cfg.write_text('{}')
            with patch.object(ev, 'ROOT', root), patch.object(ev, 'CONFIG', cfg):
                original = ev.protocol_hash()
                self.assertEqual(original, ev.protocol_hash())
                cfg.write_text('{"seed":42}')
                self.assertNotEqual(original, ev.protocol_hash())

    def test_result_filename(self):
        with patch.object(ev, 'RUN', Path('/fake/run')):
            self.assertEqual(ev.result_path('T0_600', 'sciq'), Path('/fake/run/results/sciq-T0_600.json'))
            self.assertEqual(ev.result_path('T0_600', 'sciq', 20), Path('/fake/run/smoke/sciq-T0_600.json'))

    def test_pile_nll_ppl(self):
        metrics = ev.pile_metrics([{'chunk_nll': 1.0}, {'chunk_nll': 3.0}])
        self.assertEqual(metrics['mean_nll'], 2.0)
        self.assertEqual(metrics['ppl'], math.exp(2.0))
        self.assertEqual(metrics['predicted_tokens'], 510)
        self.assertEqual(metrics['total_nll_sum'], 1020.0)

    @unittest.skipUnless(torch.cuda.is_available(), 'GPU model integration')
    def test_actual_checkpoint_entrypoints(self):
        # Run from repository root. Reload ponder for each checkpoint's exact
        # MODEL/REVISION, and patch evalsets26's imported bindings accordingly.
        from src import ponder, looped
        cfg = json.loads(Path('configs/exp26.json').read_text())
        try:
            with patch.dict(os.environ):
                for name, spec in cfg['checkpoints'].items():
                    os.environ['PYTHIA_MODEL'] = spec['model']
                    os.environ['PYTHIA_REVISION'] = spec['revision']
                    importlib.reload(ponder)
                    self.assertEqual((ponder.MODEL, ponder.REVISION), (spec['model'], spec['revision']))
                    with patch.multiple(ev, load_model=ponder.load_model, forward=ponder.forward,
                                        MODEL=ponder.MODEL, REVISION=ponder.REVISION):
                        model = ponder.load_model('vanilla' if spec['mode'] == 'looped' else spec['mode'], spec['path']).eval()
                        if spec['mode'] == 'looped':
                            looped.configure(model, model.config.loop_block_size, model.config.loop_repetitions,
                                             start=getattr(model.config, 'loop_block_start', None))
                        x = torch.tensor([[10, 20, 30, 40, 50]], device='cuda')
                        with torch.no_grad():
                            out = ev.model_forward(model, x, spec)
                            with torch.autocast('cuda', dtype=torch.bfloat16):
                                expected = torch.nn.functional.cross_entropy(
                                    out.logits[:, :-1].contiguous().reshape(-1, out.logits.shape[-1]), x[:, 1:].reshape(-1))
                        self.assertEqual(tuple(out.logits.shape[:2]), (1, 5))
                        torch.testing.assert_close(out.loss.float(), expected, atol=1e-5, rtol=1e-5)
                        print('evalsets26 real entrypoint', name, float(out.loss), flush=True)
                        del model, out, expected
                        torch.cuda.empty_cache()
        finally:
            importlib.reload(ponder)


if __name__ == '__main__':
    unittest.main()
