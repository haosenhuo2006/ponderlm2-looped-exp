"""Unit tests for evaluation configuration guards."""
from types import SimpleNamespace
import unittest

from src.evaluate import detect_loop_config
from src.evaluate_addition import resolve_loop
from src.evalsets26 import MASK_SEMANTICS, validate_spec_mask


class DetectLoopConfigTests(unittest.TestCase):
    def test_vanilla_config(self):
        self.assertIsNone(detect_loop_config(SimpleNamespace()))

    def test_complete_loop_config(self):
        config = SimpleNamespace(loop_block_size=3, loop_repetitions=3, loop_block_start=0)
        self.assertEqual(detect_loop_config(config), dict(block_size=3, repetitions=3, start=0))

    def test_loop_start_can_be_omitted(self):
        config = SimpleNamespace(loop_block_size=3, loop_repetitions=3)
        self.assertEqual(detect_loop_config(config), dict(block_size=3, repetitions=3, start=None))

    def test_missing_required_loop_field(self):
        for present, missing in (("loop_block_size", "loop_repetitions"),
                                 ("loop_repetitions", "loop_block_size")):
            with self.subTest(missing=missing):
                config = SimpleNamespace(**{present: 3})
                with self.assertRaises(ValueError) as error:
                    detect_loop_config(config, checkpoint="runs/example/checkpoint")
                self.assertIn(missing, str(error.exception))
                self.assertIn("runs/example/checkpoint", str(error.exception))

    def test_looped_mode_requires_checkpoint_loop_fields(self):
        with self.assertRaises(ValueError) as error:
            detect_loop_config(SimpleNamespace(), mode="looped", checkpoint="runs/vanilla/checkpoint")
        self.assertIn("--mode looped", str(error.exception))
        self.assertIn("runs/vanilla/checkpoint", str(error.exception))


class ResolveLoopTests(unittest.TestCase):
    def setUp(self):
        self.checkpoint_loop = dict(block_size=3, repetitions=3, start=1)
        self.config_loop = dict(block_size=2, repetitions=4, start=0)

    def test_checkpoint_takes_precedence(self):
        for external in (None, self.config_loop):
            with self.subTest(external=external):
                self.assertEqual(resolve_loop(self.checkpoint_loop, external, override=False),
                                 (self.checkpoint_loop, "checkpoint", None))

    def test_explicit_override_warns_with_both_configurations(self):
        loop, source, warning = resolve_loop(self.checkpoint_loop, self.config_loop, override=True)
        self.assertEqual(loop, self.config_loop)
        self.assertEqual(source, "external-config-override")
        self.assertIn("WARNING", warning)
        self.assertIn(str(self.checkpoint_loop), warning)
        self.assertIn(str(self.config_loop), warning)
        self.assertGreaterEqual(len(warning.splitlines()), 3)

    def test_override_requires_external_loop(self):
        for external in (None, {}):
            with self.subTest(external=external):
                with self.assertRaisesRegex(ValueError, "--override-loop"):
                    resolve_loop(self.checkpoint_loop, external, override=True)

    def test_config_fallback_without_checkpoint_loop(self):
        self.assertEqual(resolve_loop(None, self.config_loop, override=False),
                         (self.config_loop, "config-fallback", None))

    def test_default_vanilla_without_loop_settings(self):
        for external in (None, {}):
            with self.subTest(external=external):
                self.assertEqual(resolve_loop(None, external, override=False),
                                 (dict(block_size=2, repetitions=1, start=2), "default-vanilla", None))

    def test_incomplete_checkpoint_loop_is_rejected(self):
        for field in self.checkpoint_loop:
            partial = self.checkpoint_loop.copy()
            del partial[field]
            for override in (False, True):
                with self.subTest(missing=field, override=override):
                    with self.assertRaisesRegex(ValueError, "missing loop fields"):
                        resolve_loop(partial, self.config_loop, override=override)


class ValidateSpecMaskTests(unittest.TestCase):
    def test_missing_or_none_mask_is_allowed(self):
        validate_spec_mask({})
        validate_spec_mask({"mask": None})

    def test_non_none_masks_are_rejected(self):
        for value in ("D0", "D1", "D2", "other", "", False, 0, []):
            with self.subTest(mask=value):
                with self.assertRaises(ValueError) as error:
                    validate_spec_mask({"mask": value})
                self.assertIn(MASK_SEMANTICS, str(error.exception))


if __name__ == "__main__":
    unittest.main()
