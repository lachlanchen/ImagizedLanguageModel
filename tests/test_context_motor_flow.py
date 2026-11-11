"""Numerical/runtime contracts only; these tests make no language-quality claim."""
import torch

from ilm.visual_lm.context_motor_flow import ContextMotorFlow, ContextMotorBridge
from ilm.visual_lm.semantic_glyph_loop import SemanticGlyphBridge


def test_registered_integrator_and_explicit_rng():
    class LinearTimeField(ContextMotorFlow):
        def condition(self, state):
            return state.new_zeros(len(state), 1536)

        def velocity(self, x, t, condition):
            return t[:, None].expand_as(x)

    torch.set_num_threads(2)
    field = LinearTimeField()
    states = torch.zeros(2, 4096)
    result, noise = field.sample(states, generator=torch.Generator().manual_seed(49))
    # Heun integrates time exactly on the first31 intervals; the final Euler
    # interval omits half of one squared step. No global RNG is consulted.
    expected = noise + .5 - .5/32**2
    assert torch.allclose(result, expected, atol=2e-6, rtol=0)
    torch.manual_seed(987)
    repeated, same_noise = field.sample(states, generator=torch.Generator().manual_seed(49))
    assert torch.equal(noise, same_noise) and torch.equal(result, repeated)


def test_feedback_is_unchanged_and_old_point_head_is_absent():
    torch.set_num_threads(2)
    base = SemanticGlyphBridge().eval()
    base.semantic_mean.normal_(); base.semantic_scale.uniform_(.1, 2)
    bridge = ContextMotorBridge(base, ContextMotorFlow(), torch.zeros(1024), torch.ones(1024)).eval()
    pixels = 1-2*torch.randint(0, 2, (2, 1, 32, 32)).float()
    with torch.no_grad():
        assert torch.equal(bridge.feedback_embedding(pixels), base.feedback_embedding(pixels))
    assert not any(k.startswith(('readout.', 'decoder_input.')) for k in bridge.state_dict())
    assert bridge.encoder is base.encoder and bridge.decoder is base.decoder and bridge.stop is base.stop


def test_actual_network_receives_flow_gradient():
    torch.set_num_threads(2); torch.manual_seed(7)
    flow = ContextMotorFlow()
    y, epsilon = torch.randn(2, 1024), torch.randn(2, 1024)
    t, h = torch.tensor([.2, .8]), torch.randn(2, 4096)
    x = (1-t[:, None])*epsilon+t[:, None]*y
    loss = (flow(x, t, h)-(y-epsilon)).square().mean()
    loss.backward()
    assert torch.isfinite(loss) and flow.output.weight.grad.abs().sum() > 0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in flow.parameters())
