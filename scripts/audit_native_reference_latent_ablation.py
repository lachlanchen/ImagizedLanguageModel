#!/usr/bin/env python3
"""Recompute the complete reference-latent ablation from original CPU tensors."""
import argparse
import json
import os
from pathlib import Path
import subprocess

from scripts.audit_native_writer_content_control import digest,equal_fields,normalize,require

ROOT=Path(__file__).resolve().parents[1]


def load(path):return json.loads(Path(path).read_text())


def audit(run,output):
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','hide CUDA for independent audit')
    require(not output.exists(),'preserve previous audit attempts')
    meta=load(run/'preparation.json');execution=run/'execution';model=load(execution/'report.json')
    require(load(run/'coordinator.json')['phase']=='completed'
        and load(execution/'runtime.json')['phase']=='completed','both model and scoring children must complete')
    for name,expected in {**meta['code_sha256'],**meta['bound_sources_sha256']}.items():
        require(digest(ROOT/name)==expected,'registered source changed: '+name)
    for name,expected in meta['prepared_files_sha256'].items():require(digest(run/name)==expected,'prepared file changed: '+name)
    require(model['preparation_sha256']==digest(run/'preparation.json')
        and model['outputs_sha256']==digest(execution/'outputs.json'),'execution bindings differ')
    require(model['optimizer_updates']==0 and model['supplied_answer_text']
        and model['tokenizer_and_word_table_retained'] and not model['target_PNG_used_by_generator']
        and model['peak_allocated_vram_GiB']<=23 and model['elapsed_seconds']<1800,'diagnostic boundary differs')
    cases=load(run/'cases.json');counterparts=load(run/'counterparts.json');cal=load(run/'calibration.json')
    require(len(cases)==len({r['id'] for r in cases})==4 and len(cal['rows'])==4,'four fixed cases required')
    for case,row in zip(cases,cal['rows']):
        for key in ('input','target'):require(digest(case[key])==case[key+'_sha256'],'source image changed')
        require(row['id']==case['id'] and row['expected']==case['reference_text']
            and row['png_sha256']==case['target_sha256'] and row['ocr_exact']==(
                normalize(row['raw_transcript'],case['language'])==normalize(case['reference_text'],case['language'])),
                'owning reference calibration computation differs')
    require(cal['cases']==cal['ocr_exact']==sum(r['ocr_exact'] for r in cal['rows'])==4,'four exact original references required')
    generated=load(execution/'outputs.json');assessment=load(run/'assessment.json');observed=assessment['rows']
    expected=[(arm,kind,i) for arm in ('source','adapted') for kind,i in [('replay',0),*[('reference_removed',j) for j in range(4)]]]
    require([(r['arm'],r['kind'],r['case_index']) for r in generated]==expected and len(observed)==10,'complete generation order required')
    require(assessment['model_report_sha256']==digest(execution/'report.json')
        and assessment['calibration_sha256']==digest(run/'calibration.json'),'assessment ownership differs')
    import torch
    import numpy as np
    from PIL import Image
    torch.set_num_threads(4);removed=preserved=0;images=[]
    for row,score in zip(generated,observed):
        arm,kind,i=row['arm'],row['kind'],row['case_index'];case=cases[i];old=counterparts[arm][i]
        require(row['id']==case['id']==old['id'] and row['seed']==case['content_seed']==old['seed']
            and row['guidance']==old['guidance']==4.,'registered paired input differs')
        require((score['arm'],score['kind'],score['id'])==(arm,kind,case['id']),'complete OCR coverage differs')
        path=execution/row['path'];tensor=path.with_name('actual_tensors.pt');probe_path=path.with_name('intervention.pt')
        require(digest(path)==row['sha256']==score['png_sha256']
            and digest(tensor)==row['result']['tensor_sha256'] and digest(probe_path)==row['intervention_sha256'],
            'original output/tensor/intervention digest differs')
        value=torch.load(tensor,map_location='cpu',weights_only=True)
        require(value['pixels'].shape==(1,3,1024,1024) and torch.isfinite(value['pixels']).all(),'finite actual page required')
        actual=torch.clamp(127.5*value['pixels'][0].permute(1,2,0)+128.,0,255).to(torch.uint8).numpy()
        with Image.open(path) as image:require(np.array_equal(actual,np.asarray(image.convert('RGB'))),'actual decoder PNG pixels differ')
        require(digest(old['png'])==old['sha256'] and digest(old['tensors'])==old['result']['tensor_sha256'],'original counterpart changed')
        prior=torch.load(old['tensors'],map_location='cpu',weights_only=True)
        require(equal_fields(prior['forward_inputs'],value['forward_inputs']),'original reader inputs differ')
        probe=torch.load(probe_path,map_location='cpu',weights_only=True);is_removed=kind=='reference_removed'
        require(probe['reference_removed']==is_removed and len(probe['steps'])==50,'complete intervention record required')
        for step in probe['steps']:
            require(step==dict(batch=2,before_reference_present=True,after_reference_none=is_removed,
                other_argument_objects_unchanged=True),'actual denoiser argument observation differs')
        refs=probe['first_forward']['before_reference']
        require(len(refs)==2 and all(len(branch)==1 and branch[0].ndim==3 and torch.isfinite(branch[0]).all() for branch in refs),
                'actual pre-intervention reference tensors required')
        if not is_removed:
            require(torch.equal(prior['pixels'],value['pixels']),'independent full-reference decoder replay differs')
            with Image.open(old['png']) as image:require(np.array_equal(actual,np.asarray(image.convert('RGB'))),'original replay pixels differ')
        elif i==0:
            full=torch.load(execution/(arm+'_replay')/'intervention.pt',map_location='cpu',weights_only=True)['first_forward']
            for key in ('hidden_states','timestep','encoder_hidden_states','pooled_projections'):
                require(torch.equal(full[key],probe['first_forward'][key]),'first-step isolation differs: '+key)
            require(all(torch.equal(a[0],b[0]) for a,b in zip(full['before_reference'],refs)),
                    'first-case actual reference values differ before removal')
        removed+=50*is_removed;preserved+=50*(not is_removed)
        require(score['expected']==case['reference_text'] and score['language']==case['language']
            and score['normalized']==normalize(score['raw_transcript'],case['language']) and score['ocr_exact']==(
                normalize(score['raw_transcript'],case['language'])==normalize(case['reference_text'],case['language'])),
                'independent exact OCR computation differs')
        images.append(dict(arm=arm,kind=kind,id=case['id'],path=row['path'],png_sha256=digest(path),
            actual_tensors_sha256=digest(tensor),intervention_tensors_sha256=digest(probe_path),ocr_exact=score['ocr_exact']))
    require(removed==model['removed_reference_calls']==assessment['actual_reference_removed_calls']==400
        and preserved==model['preserved_reference_calls']==assessment['actual_reference_preserved_calls']==100,'whole-attempt intervention count differs')
    require(model['denoiser_calls']==removed+preserved==500 and model['denoiser_batch_elements']==2*(removed+preserved)==1000,'whole-attempt sampler counts differ')
    old=load(ROOT/'artifacts/native_writer_content_control_v1/assessment.json');summary={}
    for arm in ('source','adapted'):
        baseline=[r for r in old['rows'] if r['arm']==arm and r['kind']=='supplied_content' and r['guidance']==4.]
        changed=[r for r in observed if r['arm']==arm and r['kind']=='reference_removed']
        require([r['id'] for r in baseline]==[r['id'] for r in changed]==[r['id'] for r in cases],'complete paired counts required')
        for row in baseline:require(row['ocr_exact']==(normalize(row['raw_transcript'],row['language'])==normalize(row['expected'],row['language'])),
                                   'original exact-score computation differs')
        summary[arm]=dict(cases=4,full_reference_exact=sum(r['ocr_exact'] for r in baseline),removed_reference_exact=sum(r['ocr_exact'] for r in changed),
            per_language={lang:dict(cases=2,full_reference_exact=sum(r['ocr_exact'] for r in baseline if r['language']==lang),
                removed_reference_exact=sum(r['ocr_exact'] for r in changed if r['language']==lang)) for lang in ('zh','en')})
    require(summary==assessment['summary'],'independent complete paired counts differ')
    report=dict(auditor_sha256=digest(__file__),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        preparation_sha256=digest(run/'preparation.json'),execution_report_sha256=digest(execution/'report.json'),
        assessment_sha256=digest(run/'assessment.json'),actual_new_generations=10,actual_removed_reference_generations=8,
        both_full_reference_replays_bit_exact=True,all8_reader_comparisons_bit_exact=True,
        both_first_step_nonreference_and_before_reference_tensors_bit_exact=True,
        actual_removed_reference_calls=removed,actual_preserved_reference_calls=preserved,
        all10_original_PNGs_tensor_authenticated=True,source_bindings_unchanged=True,reference_OCR_exact=4,reference_cases=4,
        summary=summary,images=images,model_calls_added_by_audit=0,optimizer_updates=0,
        raw_image_review_completed=False,supplied_answer_text=True,semantic_learning_validated=False,general_ilm_goal_complete=False)
    output.mkdir(parents=True,exist_ok=False)
    (output/'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();audit(args.run.resolve(),args.output.resolve())
