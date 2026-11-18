"""Tests for scripts/summarize_exp26.py Pile per-set breakdown (no GPU)."""
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"scripts/{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


summary = load("summarize_exp26")


class SummarizeExp26Tests(unittest.TestCase):
    def test_pile_set_breakdown_means(self):
        records = [
            {'index': 0, 'chunk_nll': 1.0},
            {'index': 1, 'chunk_nll': 3.0},  # OpenWebText2
            {'index': 2, 'chunk_nll': 5.0},  # Books3
            {'index': 3, 'chunk_nll': 7.0},  # Wikipedia (en)
        ]
        mapping = {0: 'OpenWebText2', 1: 'OpenWebText2', 2: 'Books3', 3: 'Wikipedia (en)'}
        out = summary.pile_set_breakdown(records, mapping)
        self.assertEqual(out['per_set']['OpenWebText2'], {'blocks': 2, 'mean_nll': 2.0})
        self.assertEqual(out['per_set']['Books3'], {'blocks': 1, 'mean_nll': 5.0})
        self.assertEqual(out['per_set']['Wikipedia (en)'], {'blocks': 1, 'mean_nll': 7.0})
        # 去维基：(1 + 3 + 5) / 3 = 3.0
        self.assertEqual(out['no_wiki']['blocks'], 3)
        self.assertAlmostEqual(out['no_wiki']['mean_nll'], 3.0)
        self.assertAlmostEqual(out['no_wiki']['ppl'], math.exp(3.0))
        # 集按块数降序
        self.assertEqual(list(out['per_set']),
                         ['OpenWebText2', 'Books3', 'Wikipedia (en)'])

    def test_pile_set_breakdown_boundary_attribution(self):
        # 块 4 的首 token 落在 Books3 文档里，前 4 块是 OpenWebText2：
        # 边界块按首 token 归属，不按块内多数 token 归属。
        records = [{'index': i, 'chunk_nll': float(i)} for i in range(5)]
        mapping = {0: 'OpenWebText2', 1: 'OpenWebText2', 2: 'OpenWebText2',
                   3: 'OpenWebText2', 4: 'Books3'}
        out = summary.pile_set_breakdown(records, mapping)
        self.assertEqual(out['per_set']['Books3'], {'blocks': 1, 'mean_nll': 4.0})
        self.assertEqual(out['per_set']['OpenWebText2']['blocks'], 4)
        self.assertAlmostEqual(out['per_set']['OpenWebText2']['mean_nll'], 1.5)
        # 去维基版不受影响（本例无维基块）
        self.assertEqual(out['no_wiki']['blocks'], 5)
        self.assertAlmostEqual(out['no_wiki']['mean_nll'], 2.0)

    def test_pile_set_breakdown_all_wiki_has_no_no_wiki(self):
        records = [{'index': 0, 'chunk_nll': 2.0}]
        out = summary.pile_set_breakdown(records, {0: 'Wikipedia (en)'})
        self.assertEqual(out['per_set']['Wikipedia (en)'], {'blocks': 1, 'mean_nll': 2.0})
        self.assertIsNone(out['no_wiki'])

    def test_pile_set_breakdown_unmapped_block_loud(self):
        with self.assertRaises(KeyError):
            summary.pile_set_breakdown([{'index': 9, 'chunk_nll': 1.0}], {0: 'Books3'})

    def test_load_pile_block_sets_missing_loud(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                summary.load_pile_block_sets(Path(directory) / 'missing.json')

    def test_load_pile_block_sets_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'pile-block-sets.json'
            path.write_text(json.dumps({
                'block_sets': [{'index': 0, 'pile_set_name': 'Books3'},
                               {'index': 1, 'pile_set_name': 'Wikipedia (en)'}],
                'block_counts': {'Books3': 1, 'Wikipedia (en)': 1},
                'pile_set_counts': {'Books3': 1, 'Wikipedia (en)': 1},
            }))
            out = summary.load_pile_block_sets(path)
            self.assertEqual(out['by_index'], {0: 'Books3', 1: 'Wikipedia (en)'})
            self.assertEqual(out['block_counts'], {'Books3': 1, 'Wikipedia (en)': 1})
            self.assertEqual(out['doc_counts'], {'Books3': 1, 'Wikipedia (en)': 1})


if __name__ == '__main__':
    unittest.main()
