"""Numerical solver/gradient checks, not language or physical qualification."""
import torch

from ilm.visual_lm.context_motor_flow import ContextMotorFlow
from ilm.visual_lm.rollout_endpoint import heun_endpoint, balanced_ink_bce


def network():
    torch.set_num_threads(2)
    torch.manual_seed(271)
    model = ContextMotorFlow()
    # Nonzero output ensures the earlier blocks participate in the derivative.
    with torch.no_grad():
        model.output.weight.normal_(std=.002)
    return model


def test_deployed_forward_and_recomputed_gradient():
    model = network()
    h = torch.randn(1,4096)
    noise = torch.randn(1,1024)
    a = heun_endpoint(model, h, noise, recompute=False)
    a.square().mean().backward()
    gradients = {k:p.grad.clone() for k,p in model.named_parameters()}
    model.zero_grad(set_to_none=True)
    b = heun_endpoint(model, h, noise, recompute=True)
    b.square().mean().backward()
    assert torch.equal(a,b)
    for name,p in model.named_parameters():
        assert torch.equal(p.grad,gradients[name]),name
    generator = torch.Generator().manual_seed(781)
    deployed,initial = model.sample(h,generator=generator)
    assert torch.equal(deployed,heun_endpoint(model,h,initial,recompute=True))


def test_actual_parameter_finite_difference_through_full_path():
    model = network()
    h,noise = torch.randn(1,4096),torch.randn(1,1024)
    loss = heun_endpoint(model,h,noise).square().mean()
    loss.backward()
    derivative = model.output.bias.grad[0].item()
    eps = .002
    with torch.no_grad():
        model.output.bias[0] += eps
        hi = heun_endpoint(model,h,noise,recompute=False).square().mean().item()
        model.output.bias[0] -= 2*eps
        lo = heun_endpoint(model,h,noise,recompute=False).square().mean().item()
    assert abs(derivative-(hi-lo)/(2*eps)) < max(2e-5,abs(derivative)*.03)


def test_frozen_decoder_transmits_endpoint_gradient():
    model = network()
    decoder = torch.nn.Sequential(torch.nn.Flatten(1),torch.nn.Linear(1024,1024),
                                  torch.nn.Unflatten(1,(1,32,32))).requires_grad_(False)
    h,noise = torch.randn(1,4096),torch.randn(1,1024)
    target = torch.randint(0,2,(1,1,32,32)).float()
    endpoint = heun_endpoint(model,h,noise)
    loss = balanced_ink_bce(decoder(endpoint),target)
    loss.backward()
    assert torch.isfinite(loss) and model.context[1].weight.grad.abs().sum()>0
    assert all(p.grad is None for p in decoder.parameters())
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
