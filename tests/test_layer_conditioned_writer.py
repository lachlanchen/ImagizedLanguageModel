import copy
import unittest

import torch

from ilm.visual_lm.compact_visual_span_model import VisualSpanWriter
from ilm.visual_lm.layer_conditioned_writer import LayerConditionedWriter


class LayerConditionedWriterTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(202611186)
        self.old=VisualSpanWriter(16,24,4,width=32).double().eval()
        self.noise=torch.randn(3,4,16,dtype=torch.double)
        # The production writer explicitly casts context to FP32. Keep source
        # parameters FP32 for its numerical path rather than changing that API.
        self.old.float()
        self.noise=self.noise.float();self.time=torch.tensor([.1,.4,.9])
        self.context=torch.randn(3,24);self.valid=torch.tensor([[1,1,1,1],[1,1,0,0],[1,0,0,0]],dtype=torch.bool)

    def test_identity_and_common_source_execution(self):
        before=torch.get_rng_state().clone()
        control=LayerConditionedWriter(copy.deepcopy(self.old),modulated=False)
        candidate=LayerConditionedWriter(copy.deepcopy(self.old),modulated=True)
        self.assertTrue(torch.equal(before,torch.get_rng_state()))
        with torch.no_grad():
            expected=self.old.velocity(self.noise,self.time,self.context,self.valid)
            a=control.velocity(self.noise,self.time,self.context,self.valid)
            b=candidate.velocity(self.noise,self.time,self.context,self.valid)
        self.assertTrue(torch.equal(a,b))
        torch.testing.assert_close(a,expected,rtol=0,atol=0)
        self.assertEqual(candidate.added_parameters,len(self.old.flow.layers)*(32*64+64))

    def test_real_conditioning_gradient_update_and_reset(self):
        candidate=LayerConditionedWriter(copy.deepcopy(self.old),modulated=True)
        initial=copy.deepcopy(candidate.state_dict())
        optimizer=torch.optim.AdamW(candidate.parameters(),lr=1e-3)
        target=torch.randn_like(self.noise)
        loss=(candidate.velocity(self.noise,self.time,self.context,self.valid)[self.valid]-target[self.valid]).square().mean()
        loss.backward()
        for mod in candidate.modulations:
            self.assertTrue(bool(torch.isfinite(mod.weight.grad).all()))
            self.assertGreater(float(mod.weight.grad.norm()),0)
        optimizer.step()
        self.assertFalse(torch.equal(candidate.modulations[0].weight,initial['modulations.0.weight']))
        candidate.load_state_dict(initial)
        for n,v in candidate.state_dict().items():self.assertTrue(torch.equal(v,initial[n]))

    def test_masked_future_cells_do_not_change_valid_velocity(self):
        candidate=LayerConditionedWriter(copy.deepcopy(self.old),modulated=True)
        with torch.no_grad():
            for mod in candidate.modulations:mod.weight.normal_(0,.02)
            a=candidate.velocity(self.noise,self.time,self.context,self.valid)
            changed=self.noise.clone();changed[~self.valid]=torch.randn_like(changed[~self.valid])*100
            b=candidate.velocity(changed,self.time,self.context,self.valid)
        self.assertTrue(torch.equal(a[self.valid],b[self.valid]))

    def test_no_empty_span_or_shape_reinterpretation(self):
        candidate=LayerConditionedWriter(copy.deepcopy(self.old),modulated=True)
        with self.assertRaises(ValueError):candidate.velocity(self.noise,self.time,self.context,torch.zeros_like(self.valid))


if __name__=='__main__':unittest.main()
