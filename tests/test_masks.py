"""Mask patterns, real attention effects, finite losses, and official D0 parity."""
import unittest
import torch
import torch.nn.functional as F
from src.masks import MASKS
from src.ponder import load_model, embeddings, forward
from src.diagnose import backbone, parallel_logits, sequential


class MaskShape(unittest.TestCase):
    def test_patterns(self):
        # Hand-written four-pair matrices; eight-pair expected is assembled from
        # these same explicit current/previous pair blocks, not builder logic.
        matrices = {
            "D0": ["10000000","11000000","11100000","11110000",
                   "11111000","11111100","11111110","11111111"],
            "D1": ["10000000","11000000","10100000","10110000",
                   "10101000","10101100","10101010","10101011"],
            "D2": ["10000000","11000000","01100000","01110000",
                   "01011000","01011100","01010110","01010111"],
        }
        blocks = {"D0": [[1,1],[1,1]], "D1": [[1,0],[1,0]],
                  "D2": [[0,1],[0,1]]}
        for c, builder in MASKS.items():
            for n in (4,8):
                expected = torch.tensor([[int(v) for v in row] for row in matrices[c]])
                if n == 8:
                    expected = torch.cat([
                        torch.cat([expected, torch.zeros(8,8,dtype=torch.long)],1),
                        torch.cat([torch.tensor(blocks[c]).repeat(4,4), expected],1)],0)
                m = builder(n,1)
                self.assertEqual(m.shape,(1,1,2*n,2*n))
                self.assertEqual(m.dtype,torch.float32)
                self.assertTrue(torch.equal(m[0,0] == 0, expected.bool()))
                self.assertTrue(torch.isneginf(m[0,0][~expected.bool()]).all())
                init = builder(n,0)[0,0] == 0
                self.assertTrue(torch.equal(init, torch.eye(n).bool() if c=="D2"
                                           else torch.ones(n,n).tril().bool()))
                # Multiple thoughts retain causality within the current pair.
                self.assertEqual(builder(n,2).shape,(1,1,3*n,3*n))


class ActualForward(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(42)
        cls.model = load_model("ponder").eval()

    @classmethod
    def tearDownClass(cls):
        del cls.model
        torch.cuda.empty_cache()

    @torch.no_grad()
    def test_attention_effect(self):
        model = self.model
        original_layers = model.gpt_neox.layers
        # One actual backbone layer isolates DIRECT attention edges. With multiple
        # layers D2 deliberately allows x_j -> h_j -> later query, so asserting
        # whole-network invariance to x_j would test the wrong semantics.
        model.gpt_neox.layers = torch.nn.ModuleList([original_layers[0]])
        try:
            e = torch.randn(1,16,model.config.hidden_size,device="cuda")
            pos = torch.arange(8,device="cuda").repeat_interleave(2)[None]
            for condition, key in (("D1",1),("D2",0)):
                changed = e.clone()
                changed[:,key] = torch.randn_like(changed[:,key])*10
                mask = MASKS[condition](8,1)
                a = backbone(model,e,pos,mask).last_hidden_state
                b = backbone(model,changed,pos,mask).last_hidden_state
                # FP32, tolerance 1e-6 for possible kernel rounding.
                torch.testing.assert_close(a[:,2:],b[:,2:],atol=1e-6,rtol=0)
                a = backbone(model,e,pos,MASKS["D0"](8,1)).last_hidden_state
                b = backbone(model,changed,pos,MASKS["D0"](8,1)).last_hidden_state
                self.assertGreater((a[:,2:]-b[:,2:]).abs().max().item(),1e-5)
        finally:
            model.gpt_neox.layers = original_layers

    @torch.no_grad()
    def test_finite_losses_and_parity(self):
        model = self.model
        x = torch.randint(0,model.config.vocab_size,(1,8),device="cuda")
        with torch.autocast("cuda",dtype=torch.bfloat16):
            manual = parallel_logits(model,x,"D0")
            torch.testing.assert_close(manual,forward(model,x,3).logits,atol=1e-6,rtol=0)
            for c in MASKS:
                for _ in range(2):
                    loss = F.cross_entropy(parallel_logits(model,x,c)[:,:-1].float().reshape(
                        -1,model.config.vocab_size),x[:,1:].reshape(-1))
                    self.assertTrue(torch.isfinite(loss))
                loss, n = sequential(model,x,c)
                self.assertTrue(torch.isfinite(torch.tensor(loss/n)))


if __name__ == "__main__":
    unittest.main()
