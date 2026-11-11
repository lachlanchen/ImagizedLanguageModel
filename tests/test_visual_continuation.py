"""Numerical/cache tests on real native decoder layers, separate from GPU feasibility."""
import copy
import pytest
import torch
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel

from ilm.visual_lm.visual_continuation import PromptMemory,ReadOnlyPromptCache,VisualContinuationDynamics


def fixture():
    torch.manual_seed(73)
    config=Qwen3VLTextConfig(vocab_size=0,hidden_size=32,intermediate_size=64,
        num_hidden_layers=3,num_attention_heads=4,num_key_value_heads=2,head_dim=8,
        rope_scaling={'rope_type':'default','mrope_section':[2,1,1]})
    config._attn_implementation='eager'
    base=Qwen3VLTextModel(config).eval();base.embed_tokens=None
    prompt=torch.randn(1,5,32)
    with torch.no_grad():out=base(inputs_embeds=prompt,use_cache=True)
    memory=PromptMemory(tuple(x.keys.detach() for x in out.past_key_values.layers),
                        tuple(x.values.detach() for x in out.past_key_values.layers),5)
    dynamics=VisualContinuationDynamics(base,torch.randn(32),rank=2,alpha=4.)
    return dynamics,memory,prompt,out


def test_stateless_cache_does_not_append_on_recomputation():
    d,m,_,_=fixture();cache=ReadOnlyPromptCache(m)
    k=torch.randn(1,2,2,8,requires_grad=True);v=torch.randn_like(k)
    first=cache.update(k,v,0);second=cache.update(k,v,0)
    assert torch.equal(first[0],second[0]) and m.length==5
    assert first[0].shape[-2]==7
    first[0].sum().backward();assert torch.equal(k.grad,torch.ones_like(k))


def test_zero_adapters_match_native_and_future_inputs_are_causal():
    d,m,_,prefix=fixture();x=d.embeddings(torch.randn(1,3,32))
    with torch.no_grad():
        actual=d(x,m,checkpoint_layers=False)
        pos=torch.arange(5,9)[None,None].expand(4,1,-1)
        expected=d.context(inputs_embeds=x,position_ids=pos,past_key_values=prefix.past_key_values,use_cache=True).last_hidden_state
        torch.testing.assert_close(actual,expected,atol=2e-6,rtol=2e-6)
        changed=x.clone();changed[:,-1]+=3
        other=d(changed,m,checkpoint_layers=False)
        assert torch.equal(actual[:,:-1],other[:,:-1])
        assert not torch.equal(actual[:,-1],other[:,-1])


def test_checkpoint_gradients_match_plain_and_frozen_source_has_no_gradients():
    d,m,_,_=fixture();other=copy.deepcopy(d)
    with torch.no_grad():
        for adapter in d.adapters:adapter.up.normal_(0,.01)
        other.load_state_dict(d.state_dict())
    feedback=torch.randn(1,4,32);target=torch.randn(1,5,32)
    (d(d.embeddings(feedback),m,checkpoint_layers=True)*target).sum().backward()
    (other(other.embeddings(feedback),m,checkpoint_layers=False)*target).sum().backward()
    for (name,a),(other_name,b) in zip(d.named_parameters(),other.named_parameters()):
        assert name==other_name
        if a.requires_grad:
            assert a.grad is not None and torch.isfinite(a.grad).all() and a.grad.abs().sum()>0
            torch.testing.assert_close(a.grad,b.grad,atol=1e-5,rtol=1e-5)
        else:assert a.grad is None and b.grad is None
    assert m.length==5


def test_incremental_history_matches_parallel_after_adapters_change():
    d,m,prompt,prefix=fixture()
    with torch.no_grad():
        for adapter in d.adapters:adapter.up.normal_(0,.01)
        x=d.embeddings(torch.randn(1,4,32));parallel=d(x,m,checkpoint_layers=False)
        history=None;values=[]
        for i in range(x.shape[1]):
            value,history=d(x[:,i:i+1],m,checkpoint_layers=False,previous=history,return_history=True)
            values.append(value)
        torch.testing.assert_close(torch.cat(values,1),parallel,atol=2e-6,rtol=2e-6)
        repeated=d.context(inputs_embeds=prompt,use_cache=True)
        assert torch.equal(repeated.last_hidden_state,prefix.last_hidden_state)
        for k,v,layer in zip(m.keys,m.values,repeated.past_key_values.layers):
            assert torch.equal(k,layer.keys) and torch.equal(v,layer.values)


def test_rejects_inference_capture_in_grad_mode():
    d,m,_,_=fixture()
    with pytest.raises(ValueError):d(d.query[None,None],m,return_history=True)
