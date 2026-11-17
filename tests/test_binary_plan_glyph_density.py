import copy
import unittest

import torch

from ilm.visual_lm.binary_plan_glyph_density import BinaryPlanCodec,BinaryPlanGlyphDensity,pack_plan,unpack_plan,xor_correction


class BinaryPlanGlyphTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(202611193)
        self.codec=BinaryPlanCodec(64,channels=(8,8,8,8))
        self.ink=torch.randint(0,2,(2,1,32,32),dtype=torch.uint8);self.ink[0]=0

    def test_all_code_bit_positions_roundtrip(self):
        bits=torch.eye(64,dtype=torch.uint8)
        self.assertTrue(torch.equal(bits,unpack_plan(pack_plan(bits))))
        self.assertTrue(torch.equal(1-bits,unpack_plan(pack_plan(1-bits))))
        with self.assertRaises(ValueError):pack_plan(torch.full((1,64),2))

    def test_lossless_correction_for_every_pixel_and_arbitrary_plan(self):
        pixels=torch.eye(1024,dtype=torch.uint8).reshape(1024,1,32,32)
        base=torch.randint(0,2,pixels.shape,dtype=torch.uint8)
        residual=xor_correction(pixels,base)
        self.assertTrue(torch.equal(pixels,xor_correction(base,residual)))
        for target in [self.ink,1-self.ink]:
            b=self.codec.decode_plan(torch.randint(0,2,(2,64),dtype=torch.uint8))
            self.assertTrue(torch.equal(target,xor_correction(b,xor_correction(target,b))))

    def test_hard_forward_and_real_codec_update_reset(self):
        before=copy.deepcopy(self.codec.state_dict());out=self.codec(self.ink)
        self.assertTrue(bool(((out['plan_values']==-1)|(out['plan_values']==1)).all()))
        torch.testing.assert_close(out['image_logits'],self.codec.decode_logits(out['plan_bits']),atol=0,rtol=0)
        loss=torch.nn.functional.binary_cross_entropy_with_logits(out['image_logits'],self.ink.float());loss.backward()
        self.assertTrue(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in self.codec.parameters()))
        self.assertGreater(float(self.codec.encoder[-1].weight.grad.norm()),0)
        optimizer=torch.optim.AdamW(self.codec.parameters(),lr=.001);optimizer.step()
        self.assertFalse(torch.equal(before['encoder.9.weight'],self.codec.state_dict()['encoder.9.weight']))
        self.codec.load_state_dict(before);self.assertTrue(all(torch.equal(v,before[k]) for k,v in self.codec.state_dict().items()))

    def test_plan_prior_causal_and_cached_replay(self):
        model=BinaryPlanGlyphDensity(self.codec,7,width=16,layers=1,heads=4).eval()
        states=torch.randn(2,7);condition,_,_=model._metadata(states,None,None)
        patterns=pack_plan(torch.randint(0,2,(2,64),dtype=torch.uint8));changed=patterns.clone();changed[:,4:]=255-changed[:,4:]
        logits=model.plan(condition,patterns);other=model.plan(condition,changed)
        torch.testing.assert_close(logits[:,:5],other[:,:5],atol=0,rtol=0)
        features=model.plan.condition_features(condition);cache=None
        for position in range(8):
            value,cache=model.plan.step(features,None if position==0 else patterns[:,position-1],position,cache)
            torch.testing.assert_close(value,logits[:,position],atol=2e-6,rtol=2e-5)

    def test_joint_training_bound_gradients_and_no_target_encoder_in_generation(self):
        model=BinaryPlanGlyphDensity(self.codec,7,width=16,layers=1,heads=4).eval();states=torch.randn(2,7,requires_grad=True)
        scored=model.training_log_bound(states,self.ink)
        self.assertTrue(bool(torch.isfinite(scored['log_bound']).all()))
        self.assertTrue(torch.equal(self.ink,xor_correction(scored['base'],scored['residual'])))
        (-scored['log_bound'].mean()/1024).backward()
        self.assertGreater(float(states.grad.norm()),0)
        self.assertTrue(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in [*model.plan.parameters(),*model.correction.parameters()]))
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in model.codec.parameters()))
        def forbidden(_):raise RuntimeError('Current target encoder entered generation')
        model.codec.encode_plan=forbidden
        generated,trace=model.generate(states.detach())
        self.assertTrue(torch.equal(generated,xor_correction(trace['base'],trace['residual'])))
        self.assertTrue(torch.equal(trace['plan_bits'],unpack_plan(trace['plan_patterns'])))
        self.assertTrue(bool(((generated==0)|(generated==1)).all()))

    def test_stochastic_generation_replays_the_actual_drawn_plan(self):
        model=BinaryPlanGlyphDensity(self.codec,7,width=16,layers=1,heads=4).eval();states=torch.randn(2,7)
        first,a=model.generate(states,stochastic=True,generator=torch.Generator().manual_seed(59))
        second,b=model.generate(states,stochastic=True,generator=torch.Generator().manual_seed(59))
        self.assertTrue(torch.equal(first,second));self.assertTrue(torch.equal(a['plan_bits'],b['plan_bits']))
        self.assertTrue(torch.equal(first,xor_correction(a['base'],a['residual'])))


if __name__=='__main__':unittest.main()
