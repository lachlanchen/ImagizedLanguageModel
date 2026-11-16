"""Storage contracts; these checks do not establish full-model execution."""
import copy
import unittest
import torch
from torch import nn
from ilm.visual_lm.trainable_context_checkpoint import payload,restore


class CheckpointTest(unittest.TestCase):
    def model(self):
        m=nn.Module();m.context=nn.Linear(4,4);m.encoder=nn.Linear(4,2);m.writer=nn.Linear(2,4);return m

    def test_restores_every_learned_context_tensor_and_heads(self):
        m=self.model();identity={'train_context':True}
        saved=copy.deepcopy(payload(m,identity,19))
        with torch.no_grad():
            for p in m.parameters():p.add_(7)
        restore(m,saved,identity)
        for name in ('context','encoder','writer'):
            expected=saved['context'] if name=='context' else saved['image_heads'][name]
            for k,v in getattr(m,name).state_dict().items():self.assertTrue(torch.equal(v,expected[k]))

    def test_refuses_missing_learned_context_and_different_identity(self):
        m=self.model();identity={'train_context':True};saved=payload(m,identity,1);saved['context']=None
        with self.assertRaises(ValueError):restore(m,saved,identity)
        with self.assertRaises(ValueError):restore(m,saved,{'train_context':False})

    def test_frozen_control_uses_no_duplicated_context(self):
        m=self.model();identity={'train_context':False};saved=payload(m,identity,0)
        self.assertIsNone(saved['context']);restore(m,saved,identity)


if __name__=='__main__':unittest.main()
