#!/usr/bin/env python3
"""Freeze prior publication and export completed measurements with original pixels."""
import fcntl
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image

import build_ranking_completed_article as base
from prepare_glyph_score_gap_figure import owners

ROOT,HERE=base.ROOT,base.HERE
STAGE=ROOT/'artifacts/visual_context_evidence_publication_v1/article_attempt01'
DATA=HERE/'evidence/visual_context_completed_figure_data.json'


def protected_sources():
    old=base.model_sources()
    registry=ROOT/'references/evidence/trainable_visual_context_comparison_registration_20261116.json'
    bound=base.load(registry)['input_sha256']
    code={n:d for n,d in bound.items() if n.startswith(('ilm/','scripts/','tests/','references/'))}
    assert all(base.sha(ROOT/n)==d for n,d in code.items())
    return dict(previous_registration_sha256=old,current_registration_sha256=base.sha(registry),
                current_bound_code_and_public_evidence=len(code),previous_bound_sources=115)


def main():
    protected=owners();bindings=protected_sources();h=base.host()
    assert h['available_gib']>=48 and h['some']<=.5 and h['full']<=.1
    STAGE.mkdir(parents=True,exist_ok=False);before=STAGE/'before';before.mkdir()
    prior=[p for p in HERE.iterdir() if p.is_file() and p.suffix in ('.tex','.sty','.pdf')]
    files={}
    for p in prior:
        shutil.copy2(p,before/p.name);files[p.name]=base.sha(p)
    previous_figures={str(p.relative_to(HERE)):base.sha(p) for p in (HERE/'figures').rglob('*') if p.is_file()}
    (STAGE/'before_manifest.json').write_text(json.dumps(dict(source_commit='41b36272',files_sha256=files,
        figures_sha256=previous_figures),indent=2)+'\n')
    names=['white_target_physical_result_20261116.json','isolated_glyph_partial_falsification_20261116.json',
        'isolated_reader_preprocessing_audit_20261116.json','training_glyph_label_result_20261116.json',
        'frozen_encoder_verifier_calibration_20261116.json','trainable_visual_context_preflight_result_20261116.json']
    paths=[ROOT/'references/evidence'/n for n in names];values={p.name:base.load(p) for p in paths}
    white=values[names[0]];assert not white['physical_gate_passed'] and white['complete']
    assert white['totals']['candidate']['false_ink']==12 and white['totals']['candidate']['false_blanks']==54
    run=ROOT/'artifacts/white_target_ranking_v1/qualify';report=base.load(run/'report.json')
    assert base.sha(run/'report.json')==white['qualification_report_sha256']
    samples={};target=HERE/'figures/visual_context_completed_samples';target.mkdir(exist_ok=False)
    for record in report['records']:
        p=run/'cases'/record['job']['id']/'pixels.npz'
        assert base.sha(p)==record['pixels_sha256']
        with np.load(p,allow_pickle=False) as a:
            gold,parent,candidate=(a[k] for k in ('gold_ink','parent_ink','candidate_ink'))
            nonempty=[np.count_nonzero(x.reshape(len(x),-1),axis=1)>0 for x in (gold,parent,candidate)]
            g,b,c=nonempty
            rules={'cleared_white':~g&b&~c,'residual_false_ink':~g&c,'introduced_false_blank':g&b&~c}
            for rule,mask in rules.items():
                if rule in samples or not mask.any():continue
                index=int(np.flatnonzero(mask)[0]);images=[]
                for role,array in zip(('source','control','candidate'),(gold,parent,candidate)):
                    name=rule+'_'+role+'.png';path=target/name;tile=array[index,0]
                    assert tile.shape==(32,32) and np.isin(tile,[0,1]).all()
                    Image.fromarray((255*(1-tile)).astype(np.uint8)).save(path)
                    images.append(dict(role=role,file=str(path.relative_to(ROOT)),sha256=base.sha(path),
                        original_binary_tile_sha256=base.hashlib.sha256(tile.tobytes()).hexdigest()))
                samples[rule]=dict(rule=rule,case=record['job']['id'],position=index,
                    split=record['job']['split'],kind=record['job']['kind'],layout=record['job']['layout'],
                    source_array_sha256=base.sha(p),images=images)
        if len(samples)==3:break
    assert len(samples)==3
    reader=[]
    for index,expected,observed in [(1,'U+5B9E','A'),(2,'U+7BA1','U+03C0')]:
        p=ROOT/f'references/evidence/isolated_glyph_reader_samples/reference_{index:02}.png'
        q=target/f'reader_reference_{index:02}.png';shutil.copy2(p,q)
        reader.append(dict(file=str(q.relative_to(ROOT)),sha256=base.sha(q),expected=expected,observed=observed))
    groups=[]
    for row in white['complete_groups']:
        if row['role']!='candidate':continue
        control=next(x for x in white['complete_groups'] if x['role']=='parent' and all(x[k]==row[k] for k in ('split','kind','layout')))
        label=('Dev' if row['split']=='development' else 'Train')+' '+row['kind']+' / '+('fixed' if row['layout']=='fixed_cells' else 'prop.')
        groups.append(dict(label=label,**row,control_false_blanks=control['false_blanks']))
    data=dict(schema='visual-context-completed-figure-data-v1',source_sha256={str(p.relative_to(ROOT)):base.sha(p) for p in paths},
        protected_model_sources=bindings,exporter_sha256=base.sha(__file__),white=dict(totals=white['totals'],groups=groups,
        samples=list(samples.values()),physical_gate_passed=False,maximum_false_ink=11,maximum_false_blanks=37),
        reader_samples=reader,encoder_metric=dict(correct=6,correct_total=9,wrong=0,wrong_total=9,uncertain=0,uncertain_total=46),
        source_audit=dict(cells=7255223,renderer_mismatches=0,nonwhite_alias_groups=11,invisible_nonspace_occurrences=34),
        preflight=values[names[-1]],teacher=dict(saved_queries=402,total_queries=526,observed_reference_masks=273,
            failed_reference_masks=203,unobserved_reference_masks=102),
        new_model_calls=0,new_optimizer_updates=0,original_readings_unchanged=True,running_context_language_unmeasured=True)
    protected_sources();assert owners()==protected
    with DATA.open('x') as f:json.dump(data,f,indent=2);f.write('\n')
    print(json.dumps(dict(prior_figure_assets=len(previous_figures),sample_rows=3,original_teacher_references=2,
                          protected_model_code=bindings,new_model_calls=0)))


if __name__=='__main__':
    with (ROOT/'artifacts/publication_cpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);main()
