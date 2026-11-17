import copy
import unittest

import numpy as np
import torch

from ilm.visual_lm.global_form_supervision import GlobalFormSupervisor,build_form_index,form_labels
from ilm.visual_lm.occupancy_glyph_density import OccupancyGlyphDensity,encode_occupancy


class GlobalFormSupervisionTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(202611192)
        self.density=OccupancyGlyphDensity(7,size=4,width=16,layers=2,heads=4).eval()
        self.helper=GlobalFormSupervisor(16,5)
        self.states=torch.randn(4,7)
        self.images=torch.randint(0,2,(4,1,4,4),dtype=torch.uint8);self.images[0]=0
        self.labels=torch.tensor([0,1,3,2])

    def test_exact_physical_labels_aliases_order_and_unknowns(self):
        ink=torch.zeros(4,1,32,32,dtype=torch.uint8);ink[1,0,0,0]=1;ink[2]=ink[1];ink[3,0,31,31]=1
        packed=torch.from_numpy(np.packbits(ink.numpy().reshape(4,1024),axis=1,bitorder='big'))
        index=build_form_index(packed)
        self.assertEqual(index,build_form_index(packed.flip(0)))
        labels=form_labels(ink,index);self.assertEqual(len(index),3);self.assertEqual(int(labels[1]),int(labels[2]))
        unknown=ink[:1].clone();unknown[0,0,5,5]=1
        with self.assertRaises(ValueError):form_labels(unknown,index)

    def test_helper_reaches_shared_condition_and_no_physical_prefix(self):
        state=self.states.clone().requires_grad_()
        ce,logits=self.helper.losses(self.density,state,self.labels)
        ce.mean().backward()
        self.assertGreater(float(state.grad.norm()),0)
        for module in [self.density.condition,self.density.glyph_position,self.density.span_length]:
            self.assertTrue(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in module.parameters()))
            self.assertGreater(sum(float(p.grad.norm()) for p in module.parameters()),0)
        for module in [self.density.blocks,self.density.bits,self.density.root_output,self.density.mask_output]:
            self.assertTrue(all(p.grad is None for p in module.parameters()))
        original=logits.detach().clone()
        self.density(self.states,encode_occupancy(self.images))
        self.density(self.states,encode_occupancy(1-self.images))
        self.assertTrue(torch.equal(original,self.helper.logits(self.density,self.states)))

    def test_combined_complete_gradient_matches_microbatch_accumulation(self):
        first=copy.deepcopy(self.density);second=copy.deepcopy(self.density)
        a=copy.deepcopy(self.helper);b=copy.deepcopy(self.helper)
        seq=encode_occupancy(self.images)
        ce,_=a.losses(first,self.states,self.labels)
        loss=(first.image_nll(self.states,self.images)/1024+ce/a.loss_scale).mean();loss.backward()
        for start in [0,2]:
            states=self.states[start:start+2];ce,_=b.losses(second,states,self.labels[start:start+2])
            term=(second.image_nll(states,self.images[start:start+2])/1024+ce/b.loss_scale).sum()/4
            term.backward()
        for x,y in zip([*first.parameters(),*a.parameters()],[*second.parameters(),*b.parameters()]):
            self.assertIsNotNone(x.grad);self.assertIsNotNone(y.grad)
            torch.testing.assert_close(x.grad,y.grad,atol=2e-6,rtol=2e-5)

    def test_helper_construction_preserves_writer_rng_and_inference(self):
        before=copy.deepcopy(self.density.state_dict());rng=torch.get_rng_state().clone()
        a,trace=self.density.generate(self.states)
        helper=GlobalFormSupervisor(16,5)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        def forbidden(_):raise RuntimeError('Training-only helper entered generation')
        helper.forward=forbidden
        b,other=self.density.generate(self.states)
        self.assertTrue(torch.equal(a,b));self.assertEqual(trace,other)
        self.assertTrue(all(torch.equal(v,before[n]) for n,v in self.density.state_dict().items()))
        self.assertFalse(any('classifier' in n for n in self.density.state_dict()))

    def test_actual_update_and_exact_reset_without_a_lookup(self):
        before=copy.deepcopy(self.density.state_dict());helper_before=copy.deepcopy(self.helper.state_dict())
        optimizer=torch.optim.AdamW([*self.density.parameters(),*self.helper.parameters()],lr=1e-3)
        ce,_=self.helper.losses(self.density,self.states,self.labels)
        loss=(self.density.image_nll(self.states,self.images)/1024+ce/self.helper.loss_scale).mean()
        loss.backward();optimizer.step()
        self.assertFalse(torch.equal(self.density.condition[1].weight,before['condition.1.weight']))
        self.density.load_state_dict(before);self.helper.load_state_dict(helper_before)
        self.assertTrue(all(torch.equal(v,before[n]) for n,v in self.density.state_dict().items()))
        self.assertTrue(all(torch.equal(v,helper_before[n]) for n,v in self.helper.state_dict().items()))


if __name__=='__main__':unittest.main()
