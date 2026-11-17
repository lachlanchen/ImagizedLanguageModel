import copy
import unittest

import torch

from ilm.visual_lm.binary_plan_glyph_density import BinaryPlanCodec,BinaryPlanGlyphDensity,xor_correction
from ilm.visual_lm.binary_plan_codec_learning import codec_forward,reconstruction_terms
from ilm.visual_lm.spatial_binary_plan_codec import SpatialBinaryPlanCodec,sites_to_plan,plan_to_sites


class SpatialBinaryCodecTest(unittest.TestCase):
    def test_every_physical_plan_coordinate_roundtrips(self):
        values=torch.eye(64).reshape(64,4,4,4)
        self.assertTrue(torch.equal(values,plan_to_sites(sites_to_plan(values))))
        numbered=torch.arange(64).reshape(1,4,4,4)
        self.assertEqual(sites_to_plan(numbered)[0,:4].tolist(),[0,16,32,48])

    def test_original_convolution_initialization_and_fixed_capacity(self):
        torch.manual_seed(202611195);rng=torch.get_rng_state();global_codec=BinaryPlanCodec()
        torch.set_rng_state(rng);spatial=SpatialBinaryPlanCodec();before=global_codec.state_dict();after=spatial.state_dict()
        shared=[n for n in before if n.startswith('decoder.') or (n.startswith('encoder.') and int(n.split('.')[1])<8)]
        self.assertTrue(all(torch.equal(before[n],after[n]) for n in shared))
        self.assertEqual(sum(v.numel() for n,v in before.items() if n in shared),1766209)
        self.assertEqual(sum(p.numel() for p in spatial.parameters()),1767365)
        self.assertEqual(spatial.code_bits,64)

    def test_real_updates_and_reset_in_both_modes(self):
        torch.manual_seed(202611198);images=torch.randint(0,2,(3,1,32,32),dtype=torch.uint8);images[0]=0
        for mode in ['continuous','binary']:
            model=SpatialBinaryPlanCodec(channels=(8,8,8,8));before=copy.deepcopy(model.state_dict())
            out=codec_forward(model,images,mode)
            if mode=='binary':torch.testing.assert_close(out['image_logits'],model.decode_logits(out['plan_bits']),atol=0,rtol=0)
            loss=reconstruction_terms(out['image_logits'],images)['loss'].mean();loss.backward()
            self.assertTrue(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters()))
            self.assertGreater(float(model.plan_projection.weight.grad.norm()),0)
            optimizer=torch.optim.AdamW(model.parameters(),lr=.0002);optimizer.step()
            self.assertFalse(torch.equal(before['expand.weight'],model.expand.weight))
            model.load_state_dict(before);self.assertTrue(all(torch.equal(v,before[n]) for n,v in model.state_dict().items()))

    def test_joint_writer_keeps_target_free_generation(self):
        torch.manual_seed(202611199);codec=SpatialBinaryPlanCodec(channels=(8,8,8,8))
        model=BinaryPlanGlyphDensity(codec,7,width=16,layers=1,heads=4).eval();states=torch.randn(2,7,requires_grad=True)
        images=torch.randint(0,2,(2,1,32,32),dtype=torch.uint8);score=model.training_log_bound(states,images)
        (-score['log_bound'].mean()/1024).backward();self.assertGreater(float(states.grad.norm()),0)
        self.assertTrue(torch.equal(images,xor_correction(score['base'],score['residual'])))
        def forbidden(_):raise RuntimeError('Target encoder entered inference')
        codec.encode_plan=forbidden;generated,trace=model.generate(states.detach())
        self.assertTrue(torch.equal(generated,xor_correction(trace['base'],trace['residual'])))


if __name__=='__main__':unittest.main()
