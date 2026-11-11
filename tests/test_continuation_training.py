"""Training schedule and state recovery invariants; no language-quality claim."""
import copy
import random
import numpy as np
import pytest
import torch
from torch import nn

from ilm.visual_lm.continuation_training import (layout_schedule,learning_rate,rng_state,
    set_rng_state,trainable_state,restore_trainable,same_nested)


def test_complete_three_epoch_coverage_and_both_presentations():
    a=layout_schedule([None]*17,3,71);b=layout_schedule([None]*17,3,71)
    assert np.array_equal(a,b)
    for epoch in range(3):assert sorted(a[epoch*17:(epoch+1)*17,0])==list(range(17))
    for i in range(17):assert sorted(a[a[:,0]==i,1])==[0,1,1]


def test_absolute_learning_rate_survives_resume_position():
    assert learning_rate(1,1000)==pytest.approx(.0001/500)
    assert learning_rate(500,1000)==pytest.approx(.0001)
    assert learning_rate(1000,1000)==pytest.approx(.00001)
    with pytest.raises(ValueError):learning_rate(0,1000)


def test_saved_weights_optimizer_and_rng_replay_exactly(tmp_path):
    torch.manual_seed(7);random.seed(7);np.random.seed(7)
    model=nn.Sequential(nn.Linear(3,5),nn.GELU(),nn.Linear(5,2))
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
    def step():
        x=torch.randn(4,3);y=torch.randn(4,2)+random.random()+np.random.normal()
        optimizer.zero_grad();loss=(model(x)-y).square().mean();loss.backward();optimizer.step()
        return float(loss.detach())
    step();step()
    path=tmp_path/'state.pt';torch.save(dict(weights=trainable_state(model),optimizer=optimizer.state_dict(),rng=rng_state()),path)
    expected_losses=[step(),step()];weights=trainable_state(model);state=copy.deepcopy(optimizer.state_dict());expected_rng=rng_state()
    saved=torch.load(path,weights_only=True);restore_trainable(model,saved['weights']);optimizer.load_state_dict(saved['optimizer']);set_rng_state(saved['rng'])
    assert [step(),step()]==expected_losses
    same_nested(trainable_state(model),weights);same_nested(optimizer.state_dict(),state);same_nested(rng_state(),expected_rng)
    assert all(int(v['step'])==4 for v in optimizer.state.values())


def test_corrupt_parameter_map_is_rejected():
    model=nn.Linear(3,2);state=trainable_state(model);state.pop('bias')
    with pytest.raises(RuntimeError):restore_trainable(model,state)


def test_checkpoint_retention_preserves_unrecognized_files(tmp_path,monkeypatch):
    from scripts import train_visual_continuation as trainer
    monkeypatch.setattr(trainer,'rel',lambda p:str(p.resolve()))
    folder=tmp_path/'fit';(folder/'checkpoints').mkdir(parents=True)
    (tmp_path/'preparation.json').write_text('{}')
    model=nn.Linear(3,2);optimizer=torch.optim.AdamW(model.parameters())
    r=dict(source_sha256={},seed=7,runtime={},total_steps=10,
        initial_bridge_sha256='unused-for-retention-test',snapshot_steps=[0,10])
    orphan=folder/'checkpoints/step_999999_unknown.pt';orphan.write_bytes(b'preserve unknown evidence')
    records=[trainer.checkpoint(folder,model,optimizer,i,r,tmp_path,float(i),1) for i in range(4)]
    from pathlib import Path
    assert Path(records[0]['file']).exists()  # Immutable initial snapshot.
    assert not Path(records[1]['file']).exists()  # Known superseded pointer only.
    assert Path(records[2]['file']).exists() and Path(records[3]['file']).exists()
    assert orphan.read_bytes()==b'preserve unknown evidence'


def test_event_serialization_preserves_source_kind_and_event_identity():
    import json
    from scripts.train_visual_continuation import training_event
    result=dict(kind='book',step=1,target_glyphs=420,loss=.5)
    for kind in ('scheduled','discarded_resume_verification'):
        event=training_event(result,kind,1)
        assert json.loads(json.dumps(event))==dict(result,event_kind=kind,attempt=1)
    with pytest.raises(RuntimeError):training_event(dict(result,event_kind='corrupt'),'scheduled',1)
