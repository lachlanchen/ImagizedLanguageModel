import copy
import unittest

import numpy as np
import torch

from ilm.visual_lm.binary_plan_codec_learning import physical_partition,codec_forward,reconstruction_terms
from ilm.visual_lm.binary_plan_glyph_density import BinaryPlanCodec


class BinaryCodecLearningTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(202611194)
        self.codec=BinaryPlanCodec(64,channels=(8,8,8,8))
        self.ink=torch.randint(0,2,(4,1,32,32),dtype=torch.uint8);self.ink[0]=0

    def test_partition_preserves_aliases_empty_and_order(self):
        masks=torch.randint(0,2,(100,1,32,32),dtype=torch.uint8);masks[0]=0;masks[1]=masks[2]
        packed=torch.from_numpy(np.packbits(masks.numpy().reshape(100,1024),axis=1,bitorder='big'))
        held,hashes=physical_partition(packed);reverse,other=physical_partition(packed.flip(0))
        self.assertFalse(bool(held[0]));self.assertEqual(bool(held[1]),bool(held[2]));self.assertEqual(hashes[1],hashes[2])
        self.assertTrue(torch.equal(held,reverse.flip(0)));self.assertEqual(hashes,other[::-1])
        self.assertGreater(int(held.sum()),0);self.assertGreater(int((~held).sum()),0)

    def test_modes_share_all_weights_and_differ_only_in_global_values(self):
        before=copy.deepcopy(self.codec.state_dict())
        a=codec_forward(self.codec,self.ink,'continuous');b=codec_forward(self.codec,self.ink,'binary')
        self.assertTrue(bool((a['plan_values'].abs()<1).all()))
        self.assertTrue(bool(((b['plan_values']==-1)|(b['plan_values']==1)).all()))
        self.assertTrue(torch.equal(a['plan_values']>=0,b['plan_bits'].bool()))
        torch.testing.assert_close(a['image_logits'],self.codec._draw(a['plan_values']),atol=0,rtol=0)
        torch.testing.assert_close(b['image_logits'],self.codec.decode_logits(b['plan_bits']),atol=0,rtol=0)
        self.assertTrue(all(torch.equal(v,before[n]) for n,v in self.codec.state_dict().items()))

    def test_complete_image_gradients_match_microbatch_accumulation(self):
        for mode in ['continuous','binary']:
            a=copy.deepcopy(self.codec);b=copy.deepcopy(self.codec)
            reconstruction_terms(codec_forward(a,self.ink,mode)['image_logits'],self.ink)['loss'].mean().backward()
            for start in [0,2]:
                out=codec_forward(b,self.ink[start:start+2],mode)
                (reconstruction_terms(out['image_logits'],self.ink[start:start+2])['loss'].sum()/4).backward()
            for p,q in zip(a.parameters(),b.parameters()):
                self.assertIsNotNone(p.grad);self.assertTrue(bool(torch.isfinite(p.grad).all()))
                torch.testing.assert_close(p.grad,q.grad,atol=3e-6,rtol=3e-5)

    def test_actual_optimizer_update_and_exact_reset_in_both_modes(self):
        for mode in ['continuous','binary']:
            model=copy.deepcopy(self.codec);before=copy.deepcopy(model.state_dict())
            optimizer=torch.optim.AdamW(model.parameters(),lr=.0002,weight_decay=.01)
            loss=reconstruction_terms(codec_forward(model,self.ink,mode)['image_logits'],self.ink)['loss'].mean()
            loss.backward();optimizer.step()
            self.assertFalse(torch.equal(before['expand.weight'],model.expand.weight))
            model.load_state_dict(before);self.assertTrue(all(torch.equal(v,before[n]) for n,v in model.state_dict().items()))


if __name__=='__main__':unittest.main()
