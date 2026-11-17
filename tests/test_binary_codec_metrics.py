import unittest

import torch

from ilm.visual_lm.binary_codec_metrics import codec_metrics


class BinaryCodecMetricsTest(unittest.TestCase):
    def setUp(self):
        self.target=torch.zeros(4,1,32,32,dtype=torch.uint8)
        self.target[1:3,0,0,0]=1;self.target[3,0,0,:2]=1
        self.output=self.target.clone();self.output[2]=0;self.output[3,0,0,1]=0
        self.partition=dict(held=torch.tensor([False,False,False,True]),group=torch.tensor([0,1,1,2]),group_mask_sha256=['empty','one','two'])

    def test_macro_alias_weight_empty_and_physical_event_cost(self):
        r=codec_metrics(self.output,self.target,self.partition,torch.tensor([1,1,2,3]),'binary')
        self.assertAlmostEqual(r['train']['macro_form_f1'],.75)
        self.assertAlmostEqual(r['train']['macro_form_exact_rate'],.75)
        self.assertAlmostEqual(r['train']['pixels']['f1'],2/3)
        self.assertEqual(r['train']['pixels']['false_blank'],1)
        self.assertEqual(r['train']['pixels']['source_blank'],1)
        self.assertAlmostEqual(r['held']['macro_form_f1'],2/3)
        self.assertEqual(r['all']['mean_plan_and_correction_events'],9.75)

    def test_continuous_vectors_have_no_claimed_eight_event_cost(self):
        r=codec_metrics(self.output,self.target,self.partition,torch.ones(4,dtype=torch.long),'continuous')
        self.assertNotIn('mean_plan_and_correction_events',r['all'])
        with self.assertRaises(ValueError):codec_metrics(self.output,self.target,self.partition,torch.zeros(4,dtype=torch.long),'binary')


if __name__=='__main__':unittest.main()
