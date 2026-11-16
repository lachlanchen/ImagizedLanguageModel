#!/usr/bin/env python3
"""Preserve the raw execution receipt privately and export model measurements."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
PUBLIC=ROOT/'references/evidence/trainable_visual_context_preflight_result_20261116.json'
RAW=ROOT/'artifacts/trainable_visual_context_preflight_v1/result_original.json'


def export():
    if not RAW.exists():
        original=PUBLIC.read_bytes()
        assert 'project_processes' in json.loads(original)['resources']['admission']
        with RAW.open('xb') as stream:stream.write(original)
    original=RAW.read_bytes();d=json.loads(original)
    assert d['preflight_passed'] and len(d['updates'])==6 and d['all_trial_updates_reset']
    updates=[]
    for row in d['updates']:
        grads=row['context_gradients'];value={k:v for k,v in row.items() if k!='context_gradients'}
        value['context_gradient_counts']={key:sum(v[key] for v in grads.values()) for key in ('present','finite','nonzero')}
        assert value['changed_context_tensors']==(309 if row['train_context'] else 0)
        assert value['context_gradient_counts']['nonzero']==(309 if row['train_context'] else 0)
        assert value['all_parameters_restored'] and value['maximum_sampled_adamw_error']==0
        updates.append(value)
    model=dict(d['model']);model['codec']=dict(model['codec']);model['codec'].pop('path')
    model['recorded_pre_widen_image_parameters']=model.pop('trainable_image_parameters')
    control=next(r for r in updates if not r['train_context'])
    model['trainable_image_parameters']=control['moment_bytes']//8-model['teacher_projection_parameters']
    model['trainable_parameters_including_projection']=next(r for r in updates if r['train_context'])['moment_bytes']//8
    r=d['resources'];resources={k:v for k,v in r.items() if k!='admission'}
    resources['entry']=r['admission']['entry']
    resources.update(minimum_entry_available_gib=48,minimum_runtime_available_gib=24,maximum_allocated_vram_gib=22,
                     foreign_processes_changed=False,physical_gpu=0,cpu_threads=2)
    public={**d,'model':model,'updates':updates,'resources':resources,
            'private_original_receipt_sha256':hashlib.sha256(original).hexdigest(),
            'export_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'export_omits_runtime_identity_and_corrects_pre_widen_count':True}
    content=json.dumps(public,ensure_ascii=False,indent=2)+'\n'
    assert '/home/' not in content and 'project_processes' not in content
    PUBLIC.write_text(content)
    print(json.dumps({'preflight_passed':True,'resources':resources,'model':model},indent=2))


if __name__=='__main__':export()
