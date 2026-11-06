import pytest
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from ilm.visual_lm.adaptive_image_continuation import SuffixAdapterLinear, suffix_adapter_scope
from ilm.visual_lm.grounded_image_actor import ImageAdapterLinear


def test_prefill_ignores_changed_adapter_and_scope_restores_after_error():
    torch.manual_seed(81)
    module = SuffixAdapterLinear(ImageAdapterLinear(torch.nn.Linear(7, 5), 2, 4))
    value = torch.randn(3, 7)
    base = module.base(value)
    with torch.no_grad():
        module.up.fill_(.3)
    assert torch.equal(module(value), base)
    with pytest.raises(RuntimeError):
        with suffix_adapter_scope():
            assert not torch.equal(module(value), base)
            raise RuntimeError("scope must restore even after failed suffix")
    assert torch.equal(module(value), base)


def test_backward_recomputation_uses_suffix_adapter_gradients():
    torch.manual_seed(82)
    module = SuffixAdapterLinear(ImageAdapterLinear(torch.nn.Linear(7, 5), 2, 4))
    with torch.no_grad():
        module.up.normal_()
    value = torch.randn(3, 7, requires_grad=True)
    def scoped(x):
        with suffix_adapter_scope():
            return module(x)
    result = checkpoint(scoped, value, use_reentrant=False)
    got = torch.autograd.grad(result.square().sum(), (value, module.down, module.up))
    expected = module.base(value)+F.linear(F.linear(value, module.down), module.up)*module.scale
    reference = torch.autograd.grad(expected.square().sum(), (value, module.down, module.up))
    for actual, wanted in zip(got, reference):
        assert torch.equal(actual, wanted)
        assert torch.isfinite(actual).all() and actual.abs().max() > 0
    assert module.base.weight.grad is None and not module.base.weight.requires_grad
    assert torch.equal(module(value), module.base(value))
