import numpy as np
import torch

from ilm.visual_lm.discourse_writer import DiscourseWriter
from scripts.write_visual_discourse import encode_pages
from scripts.prepare_visual_discourse import normalize


def test_causality_inside_and_across_patches_and_copy_likelihood():
    torch.manual_seed(198)
    model=DiscourseWriter(32,2,16).eval()
    bank=model.encode(torch.randn(13,256))
    ids=torch.tensor([[3,9,4,9,9,5,10,2,6,3,7]])
    changed=ids.clone();changed[:,5:]=8
    at=torch.arange(6)[None]
    before=model.distributions(bank,ids,10,at)
    after=model.distributions(bank,changed,10,at)
    for x,y in zip(before,after):torch.testing.assert_close(x,y)
    logp=model.log_prob(bank,ids,10)
    for j in range(ids.size(1)):
        visible=ids[:,:j+1].clone();visible[:,-1]=0
        p=model.next_mass(bank,visible,10)
        torch.testing.assert_close(p.sum(-1),torch.ones(1))
        x=ids[0,j]
        torch.testing.assert_close(p[0,x].clamp_min(1e-30).log(),logp[0,j],atol=1e-5,rtol=1e-5)
    # An unseen physical shape can be copied only after it enters the page.
    p=model.next_mass(bank,torch.tensor([[10,0]]),10)
    assert p[0,10]>0 and p[0,11]==0 and p[0,0]==0


def test_new_bitmap_roundtrip_and_reserved_blank():
    ink=np.zeros((3,128),dtype=np.uint8)
    glyph=np.full(128,9,dtype=np.uint8)
    bank,pages=encode_pages(ink,[np.array([glyph,ink[2],glyph])])
    assert pages==[[3,2,3]]
    assert np.array_equal(bank[pages[0]],np.array([glyph,ink[2],glyph]))
    assert normalize('Alice greeted Bob before _.',['Alice','Bob'])==normalize('Tim greeted Jo before _.',['Jo','Tim'])
