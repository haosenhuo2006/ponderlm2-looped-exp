"""Model provenance and training metadata with mocked loading and CUDA calls."""
import hashlib
import importlib
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from src import ponder


VENDORED_PATH = "vendor/PonderLM-2/src/llamafactory/model/modeling/modeling_gpt_neox.py"


class LoadProvenance(unittest.TestCase):
    def test_override_and_defaults(self):
        try:
            with patch.dict(os.environ):
                os.environ.pop("PYTHIA_MODEL", None)
                os.environ.pop("PYTHIA_REVISION", None)
                with patch.dict(os.environ, {
                    "PYTHIA_MODEL": "EleutherAI/pythia-160M",
                    "PYTHIA_REVISION": "50f5173d932e8e61f858120bcb800b97af589f46",
                }):
                    importlib.reload(ponder)
                    result = ponder.provenance()
                    self.assertEqual(result["model"], "EleutherAI/pythia-160M")
                    self.assertEqual(result["revision"], "50f5173d932e8e61f858120bcb800b97af589f46")
                    self.assertIs(result["env_model_override"], True)
                    self.assertIs(result["env_revision_override"], True)
                importlib.reload(ponder)
                result = ponder.provenance()
                self.assertEqual(result["model"], "EleutherAI/pythia-70m")
                self.assertEqual(result["revision"], "a39f36b100fe8a5377810d56c3f4789b9c53ac42")
                self.assertIs(result["env_model_override"], False)
                self.assertIs(result["env_revision_override"], False)
        finally:
            importlib.reload(ponder)

    def test_weight_source(self):
        self.assertEqual(ponder.provenance()["weight_source"], f"hub:{ponder.MODEL}@{ponder.REVISION}")
        checkpoint = Path("checkpoints/step-200")
        self.assertEqual(ponder.provenance(checkpoint)["weight_source"], f"checkpoint:{checkpoint.resolve()}")

    def test_vendored_hash(self):
        result = ponder.provenance()
        self.assertEqual(result["vendored_path"], VENDORED_PATH)
        expected_hash = hashlib.sha256((ponder.ROOT / VENDORED_PATH).read_bytes()).hexdigest()
        self.assertEqual(result["vendored_sha256"], expected_hash)

    def test_load_model_keeps_provenance_out_of_config(self):
        config = ponder.GPTNeoXConfig()
        model = SimpleNamespace(config=config)
        model.cuda = Mock(return_value=model)
        checkpoint = Path("checkpoints/step-200")
        with (
            patch.object(ponder, "importlib") as importer,
            patch.object(ponder.GPTNeoXConfig, "from_pretrained", return_value=config),
        ):
            module = importer.util.module_from_spec.return_value
            module.GPTNeoXForCausalLM.from_pretrained.return_value = model
            loaded_model = ponder.load_model("ponder", checkpoint=checkpoint)

        self.assertIs(loaded_model, model)
        self.assertEqual(loaded_model.load_provenance, ponder.provenance(checkpoint))
        self.assertNotIn("load_provenance", loaded_model.config.to_dict())

    def test_build_metadata(self):
        from src import train

        parameters = Mock(return_value=[])
        recorded = {"model": "example/model", "vendored_sha256": "load-time hash"}
        model = SimpleNamespace(parameters=parameters, load_provenance=recorded)
        model_without_provenance = SimpleNamespace(parameters=parameters)
        with (
            patch.object(train.subprocess, "check_output", return_value="test-head\n"),
            patch.object(train.importlib.metadata, "version", return_value="test-version"),
            patch.object(train.torch.cuda, "get_device_name", return_value="test-gpu"),
            patch.object(train.torch.cuda, "get_device_properties", return_value=SimpleNamespace(total_memory=1024)),
            patch.object(ponder, "provenance", wraps=ponder.provenance) as get_provenance,
        ):
            metadata = train.build_metadata(model)
            get_provenance.assert_not_called()
            fallback = train.build_metadata(model_without_provenance)
            get_provenance.assert_called_once_with(None)

        self.assertEqual(metadata["model_provenance"], recorded)
        self.assertEqual(fallback["model_provenance"], ponder.provenance(None))
        self.assertEqual(metadata["torch"], train.torch.__version__)
        self.assertEqual(metadata["parameters"], 0)
        vendored_hash = hashlib.sha256((ponder.ROOT / VENDORED_PATH).read_bytes()).hexdigest()
        self.assertEqual(metadata["source_sha256"][VENDORED_PATH], vendored_hash)
        source = Path("src/train.py")
        self.assertEqual(metadata["source_sha256"][str(source)], hashlib.sha256(source.read_bytes()).hexdigest())


if __name__ == "__main__":
    unittest.main()
