import torch

from ilm.visual_lm.decoder_endpoint_loss import decoder_endpoint_loss
from ilm.visual_lm.semantic_glyph_loop import SemanticGlyphBridge


def test_endpoint_gradient_reaches_velocity_only_at_eligible_nonterminal_times():
    torch.set_num_threads(2); torch.manual_seed(17)
    decoder=SemanticGlyphBridge().decoder.eval().requires_grad_(False)
    versions=[p._version for p in decoder.parameters()]
    velocity=torch.randn(3,1024,requires_grad=True)
    mixed=torch.randn(3,1024);times=torch.tensor([.2,.8,1.])
    target=torch.zeros(3,1,32,32);target[:,:,10:23,13:17]=1
    loss,count=decoder_endpoint_loss(velocity,mixed,times,torch.zeros(1024),
        torch.ones(1024),decoder,target)
    assert count==2 and torch.isfinite(loss)
    loss.backward()
    assert velocity.grad[0].abs().sum()==0 and velocity.grad[1].abs().sum()>0
    assert velocity.grad[2].abs().sum()==0
    assert all(p.grad is None and p._version==v for p,v in zip(decoder.parameters(),versions))


def test_no_late_samples_has_a_zero_connected_loss():
    decoder=SemanticGlyphBridge().decoder.eval().requires_grad_(False)
    velocity=torch.randn(2,1024,requires_grad=True)
    loss,count=decoder_endpoint_loss(velocity,torch.zeros_like(velocity),torch.tensor([.1,.2]),
        torch.zeros(1024),torch.ones(1024),decoder,torch.zeros(2,1,32,32))
    loss.backward()
    assert count==0 and loss.item()==0 and torch.equal(velocity.grad,torch.zeros_like(velocity))
