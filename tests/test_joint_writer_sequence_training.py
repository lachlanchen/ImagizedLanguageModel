"""Check full-sequence reductions and immutable streaming cache boundaries."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from ilm.visual_lm.glyph_joint_density import GlyphJointDensity
from ilm.visual_lm import joint_writer_training as training


def test_complete_sequence_microbatches_exclude_terminal_and_preserve_reduction():
    torch.set_num_threads(2);torch.manual_seed(41)
    head=GlyphJointDensity(torch.zeros(4096),torch.ones(4096))
    reference=copy.deepcopy(head)
    states=torch.randn(66,4096);states[-1].fill_(float('nan'))
    target=torch.randint(0,256,(65,128),dtype=torch.uint8)
    expected=-reference(states[:-1],target.long()).log_softmax(-1).gather(-1,target.long()[...,None]).mean()
    expected.backward()
    optimizer=torch.optim.AdamW(head.parameters(),lr=2e-4,weight_decay=.01,foreach=False)
    class Guard:
        def guard(self):pass
    event=training.update(head,optimizer,states,target,1,41298,Guard())
    assert event['glyphs']==65 and abs(event['nll']-float(expected.detach()))<1e-6
    # Clipping scales every gradient equally; compare directions independently.
    actual=torch.cat([p.grad.flatten() for p in head.parameters()])
    gold=torch.cat([p.grad.flatten() for p in reference.parameters()])
    assert torch.nn.functional.cosine_similarity(actual,gold,dim=0)>.999999
    assert all(int(s['step'])==1 for s in optimizer.state.values())


def test_target_positions_are_complete_deterministic_and_do_not_include_terminal():
    for n in [1,2,7,65,513]:
        chosen=training.selected_positions(n)
        assert chosen[0]==0 and len(chosen)==1+min(8,n-1)
        assert len(set(chosen))==len(chosen) and max(chosen)<n
    with pytest.raises(ValueError):training.selected_positions(0)


def test_streaming_cache_authenticates_development_and_rejects_tampering(tmp_path,monkeypatch):
    monkeypatch.setattr(training,'ROOT',tmp_path);monkeypatch.setattr(training,'RUN',tmp_path)
    directory=tmp_path/'cache/qualification/example';directory.mkdir(parents=True)
    np.save(directory/'states.npy',np.zeros((3,4096),np.float32),allow_pickle=False)
    np.save(directory/'patterns.npy',np.zeros((2,128),np.uint8),allow_pickle=False)
    record=dict(glyphs=2,states_sha256=training.sha(directory/'states.npy'),
                patterns_sha256=training.sha(directory/'patterns.npy'))
    (directory/'record.json').write_text(json.dumps(record))
    entry=dict(directory=str(directory.relative_to(tmp_path)),record_sha256=training.sha(directory/'record.json'))
    (tmp_path/'cache/report.json').write_text(json.dumps(dict(records=[],qualification=[entry])))
    states,patterns,_=training.CachedSequences().get(directory)
    assert states.shape==(3,4096) and patterns.shape==(2,128)
    pixels=np.load(directory/'patterns.npy');pixels[0,0]=1
    np.save(directory/'patterns.npy',pixels,allow_pickle=False)
    with pytest.raises(ValueError,match='Cached array changed'):
        training.CachedSequences().get(directory)


def test_unfinished_prior_stage_cannot_reset_resource_budget(tmp_path,monkeypatch):
    monkeypatch.setattr(training,'RUN',tmp_path)
    p=tmp_path/'controller_journal.jsonl';p.write_text(json.dumps(dict(event='started',action='cache',child_pid=-1))+'\n')
    with pytest.raises(ValueError,match='unterminated'):
        training.previous_seconds('cache')


def test_registered_parameter_draw_is_independent_of_later_normalization():
    torch.manual_seed(73)
    first=GlyphJointDensity(torch.zeros(4096),torch.ones(4096))
    torch.manual_seed(73)
    second=GlyphJointDensity(torch.full((4096,),3.),torch.full((4096,),.2))
    assert all(torch.equal(a,b) for a,b in zip(first.parameters(),second.parameters()))
    assert not torch.equal(first.condition_mean,second.condition_mean)
