import unittest
import torch
from src.downstream import select_answer, prompt, word_correct, last_word_tokens

class DownstreamTests(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "GPU model integration")
    def test_actual_checkpoint_entrypoints(self):
        from src.downstream import config, model_forward
        from src.ponder import load_model
        from src import looped
        for name, spec in config()["checkpoints"].items():
            model = load_model("vanilla" if spec["mode"] == "looped" else spec["mode"], spec["path"]).eval()
            if spec["mode"] == "looped":
                looped.configure(model, model.config.loop_block_size, model.config.loop_repetitions, start=getattr(model.config, "loop_block_start", None))
            x = torch.tensor([[10,20,30,40,50]], device="cuda")
            with torch.no_grad():
                out = model_forward(model, x, spec)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    expected = torch.nn.functional.cross_entropy(out.logits[:,:-1].contiguous().reshape(-1,out.logits.shape[-1]),x[:,1:].reshape(-1))
            self.assertEqual(tuple(out.logits.shape[:2]), (1,5))
            torch.testing.assert_close(out.loss.float(), expected, atol=1e-5, rtol=1e-5)
            print("downstream real entrypoint",name,float(out.loss),flush=True)
            del model,out,expected
            torch.cuda.empty_cache()

    def test_mc_argmax_negative_normalized_loss(self):
        # Raw total chooses A; normalized score must choose B.
        self.assertEqual(select_answer([4., 6., 12.], [1, 3, 2]), 1)
        self.assertEqual(select_answer([2., 2.], [1, 1]), 0)

    def test_prompt_fixed(self):
        expected = "Why blue? Answer: scattering"
        self.assertEqual(prompt("Why blue?", "scattering"), expected)
        self.assertEqual(prompt("Why blue?", "scattering"), prompt("Why blue?", "scattering"))

    def test_shifted_last_word_top1(self):
        ids = [1,2,3,4]
        logits = torch.zeros(1,4,6)
        logits[0,1,3] = 9
        logits[0,2,4] = 9
        self.assertTrue(word_correct(logits, ids, 2))
        self.assertTrue(word_correct(logits, ids, 3))
        logits[0,2,0] = 10
        self.assertFalse(word_correct(logits, ids, 2))

    def test_word_boundary_offsets(self):
        class Tokenizer:
            def __call__(self, text, **kwargs):
                return dict(input_ids=[1,2,3,4], offset_mapping=[(0,2),(2,4),(4,7),(7,9)])
        ids, start, word = last_word_tokens(Tokenizer(), "a b hello  ")
        self.assertEqual((start, word, ids[start:]), (2, "hello", [3,4]))

if __name__ == "__main__":
    unittest.main()
