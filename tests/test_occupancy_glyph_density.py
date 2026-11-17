import copy
import unittest

import torch

from ilm.visual_lm.occupancy_glyph_density import OccupancyGlyphDensity,OccupancySequence,encode_occupancy,decode_occupancy


class OccupancyDensityTest(unittest.TestCase):
    def setUp(self):torch.manual_seed(202611189)

    def test_exhaustive_four_by_four_lossless_support(self):
        # Enumerate every physical bitmap, rather than a character catalogue.
        bits=((torch.arange(65536)[:,None]>>torch.arange(16))&1).reshape(-1,1,4,4).to(torch.uint8)
        for mode in ['tree','flat']:
            sequence=encode_occupancy(bits,mode)
            self.assertTrue(torch.equal(decode_occupancy(sequence,4,mode),bits))

    def test_production_size_sparse_and_dense_roundtrip(self):
        images=torch.randint(0,2,(9,1,32,32),dtype=torch.uint8)
        images[0]=0;images[1]=1;images[2]=0;images[2,0,31,31]=1
        for mode in ['tree','flat']:
            sequence=encode_occupancy(images,mode)
            self.assertTrue(torch.equal(decode_occupancy(sequence,32,mode),images))

    def test_illegal_occupancy_and_future_descriptors_are_rejected(self):
        original=encode_occupancy(torch.ones(1,1,8,8,dtype=torch.uint8))
        for name in ['truncated','wrong_position','empty_child','wrong_root','allow_empty']:
            broken=copy.deepcopy(original)
            if name=='truncated':broken.valid[0,-1]=False
            if name=='wrong_position':broken.columns[0,-1]=0
            if name=='empty_child':broken.actions[0,-1]=0
            if name=='wrong_root':broken.actions[0,0]=0
            if name=='allow_empty':broken.allow_zero[0,-1]=True
            with self.assertRaises(ValueError):decode_occupancy(broken,8)

    def test_actual_model_normalizes_over_all_two_by_two_images(self):
        images=((torch.arange(16)[:,None]>>torch.arange(4))&1).reshape(16,1,2,2).to(torch.uint8)
        state=torch.randn(1,7).expand(16,-1)
        for mode in ['tree','flat']:
            model=OccupancyGlyphDensity(7,mode=mode,size=2,width=16,layers=2,heads=4).double().float().eval()
            logp=model.log_prob(state,encode_occupancy(images,mode))
            self.assertAlmostEqual(float(logp.detach().exp().sum()),1.,places=6)

    def test_forward_and_incremental_replay_and_future_exclusion(self):
        for mode in ['tree','flat']:
            model=OccupancyGlyphDensity(7,mode=mode,size=4,width=16,layers=2,heads=4).eval()
            images=torch.ones((2,1,4,4),dtype=torch.uint8);sequence=encode_occupancy(images,mode);states=torch.randn(2,7)
            logits=model(states,sequence);condition=model.condition_features(states);cache=None;parts=[]
            for j in range(sequence.actions.shape[1]):
                value,cache=model.step(condition,None if j==0 else sequence.actions[:,j-1],sequence.depths[:,j],sequence.rows[:,j],sequence.columns[:,j],sequence.allow_zero[:,j],j,cache)
                parts.append(value)
            actual=torch.stack(parts,1)
            torch.testing.assert_close(logits[torch.isfinite(logits)],actual[torch.isfinite(actual)],atol=2e-6,rtol=2e-6)
            changed=copy.deepcopy(sequence);changed.actions[:,-1]=1
            self.assertTrue(torch.equal(logits,model(states,changed)))

    def test_generation_support_and_exact_stochastic_replay(self):
        for mode in ['tree','flat']:
            model=OccupancyGlyphDensity(7,mode=mode,size=4,width=16,layers=2,heads=4).eval()
            states=torch.randn(4,7)
            with torch.no_grad():model.root_output.weight.zero_();model.root_output.bias.copy_(torch.tensor([-100.,100.]))
            a,trace=model.generate(states,greedy=False,generator=torch.Generator().manual_seed(61))
            b,other=model.generate(states,greedy=False,generator=torch.Generator().manual_seed(61))
            self.assertTrue(torch.equal(a,b));self.assertEqual(trace,other);self.assertTrue(bool(a.flatten(1).any(1).all()))
            for item in trace:
                for active,meta,action in zip(item['active'],item['meta'],item['action']):
                    if active and meta[0]>0 and mode=='tree':self.assertGreater(action,0)
            with torch.no_grad():model.root_output.bias.copy_(torch.tensor([100.,-100.]))
            blank,trace=model.generate(states)
            self.assertFalse(bool(blank.any()));self.assertEqual(len(trace),1)

    def test_real_density_gradient_update_and_exact_reset(self):
        model=OccupancyGlyphDensity(7,size=4,width=16,layers=2,heads=4)
        states=torch.randn(4,7,requires_grad=True);images=torch.randint(0,2,(4,1,4,4),dtype=torch.uint8);images[0]=0
        initial=copy.deepcopy(model.state_dict());optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3)
        loss=model.image_nll(states,images).mean();loss.backward()
        self.assertGreater(float(states.grad.norm()),0)
        self.assertTrue(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters()))
        optimizer.step();self.assertFalse(torch.equal(model.mask_output.weight,initial['mask_output.weight']))
        model.load_state_dict(initial)
        self.assertTrue(all(torch.equal(v,initial[n]) for n,v in model.state_dict().items()))

    def test_future_topology_not_provided_to_generation(self):
        # Legal next descriptors are reconstructed from the trace prefix by
        # the same validator, including its nonempty branch constraint.
        model=OccupancyGlyphDensity(7,size=4,width=16,layers=2,heads=4).eval()
        with torch.no_grad():model.root_output.weight.zero_();model.root_output.bias.copy_(torch.tensor([-100.,100.]))
        image,trace=model.generate(torch.randn(1,7))
        sequence=encode_occupancy(image)
        self.assertEqual([v['action'][0] for v in trace],sequence.actions[0,sequence.valid[0]].tolist())
        self.assertEqual([v['meta'][0] for v in trace],list(zip(sequence.depths[0].tolist(),sequence.rows[0].tolist(),sequence.columns[0].tolist())))


if __name__=='__main__':unittest.main()
