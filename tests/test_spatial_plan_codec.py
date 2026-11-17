import copy
import unittest

import torch

from ilm.visual_lm.binary_plan_glyph_density import BinaryPlanGlyphDensity,pack_plan,unpack_plan,xor_correction
from ilm.visual_lm.binary_plan_codec_learning import codec_forward,reconstruction_terms
from ilm.visual_lm.spatial_binary_plan_codec import SpatialBinaryPlanCodec
from ilm.visual_lm.spatial_plan_codec import SpatialPlanCodec,RasterFeatures
from ilm.visual_lm.spatial_plan_metrics import plan_codec_metrics


class SpatialPlanCapacityTest(unittest.TestCase):
    def test_exact_old_64_bit_case_and_shared_initial_cnn(self):
        torch.manual_seed(202611195);rng=torch.get_rng_state();old=SpatialBinaryPlanCodec()
        torch.set_rng_state(rng);same=SpatialPlanCodec(4)
        self.assertTrue(all(torch.equal(v,same.state_dict()[n]) for n,v in old.state_dict().items()))
        torch.set_rng_state(rng);wide=SpatialPlanCodec(16)
        shared=[n for n in old.state_dict() if n.startswith('decoder.') or (n.startswith('encoder.') and int(n.split('.')[1])<8)]
        self.assertTrue(all(torch.equal(old.state_dict()[n],wide.state_dict()[n]) for n in shared))
        self.assertEqual(sum(p.numel() for p in wide.parameters()),1770449);self.assertEqual(wide.code_bits,256)

    def test_all_256_feature_coordinates_and_plan_cost(self):
        field=torch.eye(256).reshape(256,4,4,16).permute(0,3,1,2)
        bits=RasterFeatures(16)(field).to(torch.uint8)
        self.assertTrue(torch.equal(bits,torch.eye(256,dtype=torch.uint8)))
        self.assertTrue(torch.equal(bits,unpack_plan(pack_plan(bits))))
        image=torch.zeros(2,1,32,32,dtype=torch.uint8);image[1,0,1,1]=1
        part=dict(held=torch.tensor([False,True]),group=torch.arange(2),group_mask_sha256=['a','b'])
        result=plan_codec_metrics(image,image,part,torch.ones(2,dtype=torch.long),'binary',256)
        self.assertEqual(result['all']['mean_plan_and_correction_events'],33.)

    def test_real_capacity_updates_and_exact_reset(self):
        torch.manual_seed(202611200);images=torch.randint(0,2,(3,1,32,32),dtype=torch.uint8);images[0]=0
        for mode in ['continuous','binary']:
            model=SpatialPlanCodec(16,channels=(8,8,8,8));before=copy.deepcopy(model.state_dict())
            out=codec_forward(model,images,mode)
            if mode=='binary':torch.testing.assert_close(out['image_logits'],model.decode_logits(out['plan_bits']),atol=0,rtol=0)
            loss=reconstruction_terms(out['image_logits'],images)['loss'].mean();loss.backward()
            self.assertTrue(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters()))
            self.assertGreater(float(model.plan_projection.weight.grad.norm()),0)
            optimizer=torch.optim.AdamW(model.parameters(),lr=.0002);optimizer.step()
            self.assertFalse(torch.equal(before['expand.weight'],model.expand.weight))
            model.load_state_dict(before);self.assertTrue(all(torch.equal(v,before[n]) for n,v in model.state_dict().items()))

    def test_joint_256_bit_generation_excludes_target_encoder(self):
        torch.manual_seed(202611201);codec=SpatialPlanCodec(16,channels=(8,8,8,8));model=BinaryPlanGlyphDensity(codec,7,width=16,layers=1,heads=4).eval()
        states=torch.randn(2,7,requires_grad=True);images=torch.randint(0,2,(2,1,32,32),dtype=torch.uint8)
        score=model.training_log_bound(states,images);(-score['log_bound'].mean()/1024).backward();self.assertGreater(float(states.grad.norm()),0)
        def forbidden(_):raise RuntimeError('Target image entered generation')
        codec.encode_plan=forbidden;generated,trace=model.generate(states.detach())
        self.assertEqual(trace['plan_patterns'].shape,(2,32));self.assertEqual(trace['plan_bits'].shape,(2,256))
        self.assertTrue(torch.equal(generated,xor_correction(trace['base'],trace['residual'])))


if __name__=='__main__':unittest.main()
