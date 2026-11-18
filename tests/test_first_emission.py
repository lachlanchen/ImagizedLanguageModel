import pytest
import torch
from safetensors.torch import save_file

from ilm.visual_lm.first_emission import FirstEmission
from scripts.train_first_emission import first_loss


def test_checkpoint_binding_and_identical_initial_state(tmp_path):
    net = FirstEmission()
    x, initial, history = torch.randn(3, 7, 128)
    expected = (x + history).tanh()
    assert torch.equal(net(x, initial, history), expected)
    net(x, initial, history).sum().backward()
    assert net.update[-1].weight.grad.abs().sum() > 0
    path = tmp_path/'residual.safetensors'
    save_file(net.state_dict(), str(path), metadata=dict(format='ilm-first-emission-v1',interaction_sha256='parent'))
    assert torch.equal(FirstEmission.restore(path,'parent','cpu')(x, initial, history), expected)
    with pytest.raises(ValueError):
        FirstEmission.restore(path,'another','cpu')


def test_only_different_signatures_contribute_pair_gradient():
    own = torch.tensor([-2.,-3.],requires_grad=True)
    rival = torch.tensor([-1.,-3.],requires_grad=True)
    different = torch.tensor([True, False])
    first_loss(own,rival,different).backward()
    assert own.grad[0]<-.5 and own.grad[1]==-.5
    assert rival.grad[0]>0 and rival.grad[1]==0
    assert first_loss(own,rival,~torch.ones_like(different))==first_loss(own)


def test_mixed_reference_positions_match_scalar_histories():
    from scripts.exchange_conditional_ink_sources import reference_steps
    y = torch.arange(15).view(3,5)
    history = torch.arange(36).view(3,6,2)
    batch, step, end = torch.tensor([[0,1,2,0],[1,0,4,2],[3,1,5,3]])
    observed = reference_steps(y,history,batch,step,end)
    gold = [(y[b,t],history[b,t],history[b,t+1],t+1==k) for b,t,k in zip(batch,step,end)]
    for actual, expected in zip(observed,zip(*gold)):
        torch.testing.assert_close(actual,torch.stack(expected))
    with pytest.raises(ValueError):
        reference_steps(y,history,batch,step[-1],end)
