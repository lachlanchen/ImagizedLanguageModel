import torch
import pytest
from safetensors.torch import save_file

from ilm.visual_lm.sentence_choice import SentenceBoundary
from scripts.train_sentence_choice import objective


def test_identical_start_and_checkpoint_parent(tmp_path):
    net = SentenceBoundary(width=128)
    x, h = torch.randn(5,128), torch.randn(5,128)
    assert torch.equal(net(x, h), x)
    loss = net(x,h).square().sum()
    loss.backward()
    assert net.delta[-1].weight.grad.abs().sum()>0
    path=tmp_path/'update.safetensors'
    save_file(net.state_dict(),str(path),metadata=dict(format='ilm-sentence-boundary-v1',interaction_sha256='matched'))
    restored=SentenceBoundary.restore(path,'matched','cpu')
    assert torch.equal(restored(x,h),net(x,h))
    with pytest.raises(ValueError):
        SentenceBoundary.restore(path,'other','cpu')


def test_assignment_gradient_depends_on_complete_phrases_and_lengths():
    own=torch.tensor([[-2.,-3.],[-5.,-7.]],requires_grad=True)
    foil=torch.tensor([[-3.,-4.],[-6.,-2.]],requires_grad=True)
    lengths=torch.tensor([1,2])
    loss=objective([own,foil],lengths)
    torch.testing.assert_close(loss,torch.tensor((2.5+3.+torch.nn.functional.softplus(torch.tensor(-1.)).item()+
                                                  torch.nn.functional.softplus(torch.tensor(1.)).item())/2))
    loss.backward()
    assert (own.grad<0).all() and (foil.grad>0).all()
    torch.testing.assert_close(objective([own],lengths),torch.tensor(2.75))
