import copy
import unittest

import torch

from ilm.visual_lm.binary_plan_glyph_density import BinaryPlanCodec
from ilm.visual_lm.binary_plan_codec_learning import codec_forward,reconstruction_terms
from ilm.visual_lm.form_balanced_codec_support import balanced_codec_support


class FormBalancedCodecTest(unittest.TestCase):
    def setUp(self):
        self.partition=dict(group=torch.tensor([0,1,1,1,2,3]),held=torch.tensor([False,False,False,False,False,True]),
            train_index=torch.arange(5),group_mask_sha256=['empty','one','two','held'])

    def test_exact_uniform_mass_and_no_held_source(self):
        s=balanced_codec_support(self.partition,202611196)
        self.assertEqual(set(s['order'].tolist()),{0,1,4});self.assertEqual(s['training_cells'],5);self.assertEqual(s['training_forms'],3)
        self.assertTrue(torch.equal(s['order'],balanced_codec_support(self.partition,202611196)['order']))
        mass=torch.zeros(4,dtype=torch.float64);mass.scatter_add_(0,self.partition['group'],s['complete_source_weights'])
        torch.testing.assert_close(mass[:3],torch.full((3,),5/3,dtype=torch.float64),atol=0,rtol=1e-15)
        self.assertEqual(float(mass[3]),0)

    def test_complete_source_weighted_gradients_equal_uniform_forms(self):
        torch.manual_seed(202611197)
        images=torch.randint(0,2,(6,1,32,32),dtype=torch.uint8);images[0]=0;images[2]=images[1];images[3]=images[1]
        support=balanced_codec_support(self.partition,202611196);weights=support['complete_source_weights'][:5]
        for mode in ['continuous','binary']:
            a=BinaryPlanCodec(64,channels=(8,8,8,8));b=copy.deepcopy(a)
            losses=reconstruction_terms(codec_forward(a,images[:5],mode)['image_logits'],images[:5])['loss']
            (losses.double()*weights/5).sum().backward()
            idx=support['representatives'];other=reconstruction_terms(codec_forward(b,images[idx],mode)['image_logits'],images[idx])['loss']
            other.mean().backward()
            for p,q in zip(a.parameters(),b.parameters()):torch.testing.assert_close(p.grad,q.grad,atol=3e-6,rtol=3e-5)


if __name__=='__main__':unittest.main()
