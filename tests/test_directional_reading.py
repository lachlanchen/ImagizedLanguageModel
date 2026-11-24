import torch
from torch.testing import assert_close
from ilm.visual_lm.directional_reading import ContextualSpanReader, DirectionalCorrection, side_means
from ilm.visual_lm.fact_channel import OpticalFactReader
from ilm.visual_lm.fact_ranking import SpanRanker, candidates


def test_left_right_means_include_only_real_cells_and_exchange():
    x=torch.tensor([[[2.],[6.],[9.],[12.],[3.]]],requires_grad=True)
    valid=torch.tensor([[True,True,False,True,False]])
    left,right=side_means(x,valid,2)
    assert_close(left.flatten(),torch.tensor([0.,2.,4.,6.,12.]))
    assert_close(right.flatten(),torch.tensor([6.,12.,12.,0.,0.]))
    reverse_left,reverse_right=side_means(x.flip(1),valid.flip(1),2)
    assert_close(left,reverse_right.flip(1))
    assert_close(right,reverse_left.flip(1))
    left.sum().backward()
    assert x.grad[0,2].item()==0


def test_zero_head_preserves_full_reranked_parent_and_gets_gradients():
    torch.manual_seed(22)
    base=OpticalFactReader()
    features=torch.randn(18,256)
    q=torch.tensor([[3,5,2,0],[2,6,3,4]])
    page=torch.tensor([[2,4,5,0,7,6],[9,7,2,6,0,3]])
    scores=base(features,q,page)
    ranker=SpanRanker("whole")
    torch.nn.init.normal_(ranker.out.weight)
    c,b,s,n=candidates(base,features,q,page)
    ix=ranker(c,b,n).argmax(1,keepdim=True)
    for mode in ("unsigned","signed"):
        model=ContextualSpanReader(base,DirectionalCorrection(mode))
        new,_=model.lattice(features,q,page)
        assert_close(new,scores)
        a,z=model.select(features,q,page,ranker)
        assert_close(a,s.gather(1,ix).flatten())
        assert_close(z,n.gather(1,ix).flatten())
        loss=base.loss(new,torch.tensor([[0],[0]])).mean()
        loss.backward()
        assert model.correction.combine[-1].weight.grad.abs().max()>0
        assert all(v.grad is None for v in base.parameters())
