"""Environment selection only: no weights, network access, or CUDA operations."""
import importlib
import os
import unittest
from unittest.mock import patch


class ModelOverride(unittest.TestCase):
    def test_override_and_defaults(self):
        import src.ponder as ponder

        # Also restore any overrides supplied by the caller after this test.
        try:
            with patch.dict(os.environ):
                os.environ.pop("PYTHIA_MODEL", None)
                os.environ.pop("PYTHIA_REVISION", None)
                with patch.dict(os.environ, {
                    "PYTHIA_MODEL": "EleutherAI/pythia-160M",
                    "PYTHIA_REVISION": "50f5173d932e8e61f858120bcb800b97af589f46",
                }):
                    importlib.reload(ponder)
                    self.assertEqual(ponder.MODEL, "EleutherAI/pythia-160M")
                    self.assertEqual(ponder.REVISION, "50f5173d932e8e61f858120bcb800b97af589f46")
                importlib.reload(ponder)
                self.assertEqual(ponder.MODEL, "EleutherAI/pythia-70m")
                self.assertEqual(ponder.REVISION, "a39f36b100fe8a5377810d56c3f4789b9c53ac42")
        finally:
            importlib.reload(ponder)


if __name__ == "__main__":
    unittest.main()
