from itertools import product

import torch
from torch.nn import functional as F

from ilm.visual_lm.correlated_native_density import (
    ConditionalBitmapMixture,bitmap_log_prob,expert_mode_candidates,strip_blocks,
)


def test_normalization_and_single_expert_bernoulli_equality():
    torch.manual_seed(62)
    logits=torch.randn(1,3,4);gate=torch.randn(1,3)
    outcomes=torch.tensor(list(product((0.,1.),repeat=4)))
    probabilities=bitmap_log_prob(logits.expand(16,-1,-1),gate.expand(16,-1),outcomes).exp()
    torch.testing.assert_close(probabilities.sum(),torch.tensor(1.),atol=1e-6,rtol=0)
    target=outcomes[:1]
    actual=bitmap_log_prob(logits[:,:1],gate[:,:1],target)
    expected=-F.binary_cross_entropy_with_logits(logits[:,0],target,reduction="none").sum(-1)
    torch.testing.assert_close(actual,expected)


def test_shared_latent_preserves_whole_forms_instead_of_pixel_average():
    logits=torch.tensor([[[12.,12.,-12.,-12.],[-12.,-12.,12.,12.]]])
    gate=torch.zeros(1,2)
    image,score=expert_mode_candidates(logits,gate)
    assert image.tolist()[0] in ([1.,1.,0.,0.],[0.,0.,1.,1.])
    hybrid=torch.zeros(1,4)
    assert float(score)>float(bitmap_log_prob(logits,gate,hybrid))+10
    torch.testing.assert_close(score,bitmap_log_prob(logits,gate,image))


def test_native_projection_layout_gradients_and_mode_scores():
    head=ConditionalBitmapMixture(7,pixels=2048,components=2)
    native=torch.randn(7,1,32,64)*.02
    head.initialize_native_projection(native,noise=.01)
    states=torch.randn(2,3,7,requires_grad=True)
    logits,gate=head(states);targets=torch.randint(0,2,(2,3,2048)).float()
    loss=-bitmap_log_prob(logits,gate,targets).mean()/2048;loss.backward()
    for gradient in (states.grad,head.projection.weight.grad,head.gate.weight.grad):
        assert torch.isfinite(gradient).all() and gradient.abs().sum()>0
    source=torch.arange(32*128).reshape(1,1,32,128).float()
    blocks=strip_blocks(source*2-1)
    torch.testing.assert_close(blocks[0,0],source[0,0,:,:64].flatten())
    torch.testing.assert_close(blocks[0,1],source[0,0,:,64:].flatten())
