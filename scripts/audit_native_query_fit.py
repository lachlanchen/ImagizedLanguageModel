#!/usr/bin/env python3
"""Independently audit the complete saved query-fit execution on CPU.

No model/teacher loading, new optimization, image generation or OCR repair.
Actual saved float32 velocity MSE must agree within max(1e-9,1e-6*|loss|);
this accounts for CPU/GPU reduction ordering and is fixed before inspection.
"""
import argparse
import json
import math
import os
from pathlib import Path
import shutil

from ilm.image_edit.baseline import ROOT, save_json, sha
from ilm.image_edit.native_flow_assessment import authenticate_native_png
from scripts.run_native_flow_condition_probe import metric
from scripts.train_native_query_fit import unchanged, require, load, PREFLIGHT, PROBE
from scripts.evaluate_image_edit_baseline import normalize


def audit(run,output):
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','hide CUDA for independent saved-data audit')
    import torch
    torch.set_num_threads(4)
    meta=unchanged(run);execution=run/'execution'
    require(load(execution/'runtime.json')['phase']=='completed','complete model execution required')
    report=load(execution/'report.json');scoring=load(run/'assessment/result.json')
    require(report['actual_optimizer_updates']==128 and report['actual_new_generations']==18,
        'complete fixed execution required')
    require(sha(execution/'updates.json')==report['updates_sha256']
        and sha(execution/'checkpoint_128.pt')==report['checkpoint_sha256']
        and sha(execution/'report.json')==scoring['execution_report_sha256'], 'saved execution bindings changed')
    schedule=load(run/'schedule.json');history=load(execution/'updates.json')
    require(len(schedule)==len(history)==128,'all actual update records required')
    cases=load(run/'training_cases.json')
    for expected,actual in zip(schedule,history):
        require(all(actual[k]==v for k,v in expected.items()) and actual['id']==cases[expected['case']]['id'],
            'actual update schedule differs')
        require(math.isfinite(actual['loss']) and math.isfinite(actual['gradient_norm'])
            and actual['gradient_norm']>0 and actual['finite_nonzero_query_gradient'], 'invalid actual update record')
        if expected['grid_index']==0:require(actual['sigma']==1.,'actual endpoint sigma differs')
    old=load(PREFLIGHT/'execution/report.json')
    require(sha(PREFLIGHT/'execution/checkpoint.pt')==old['query_checkpoint_sha256'],'original query preflight changed')
    initial=torch.load(PREFLIGHT/'execution/checkpoint.pt',map_location='cpu',weights_only=True)['initial_queries']
    checkpoints=[];final=None
    for step in (32,64,96,128):
        path=execution/f'checkpoint_{step:03}.pt'
        state=torch.load(path,map_location='cpu',weights_only=True)
        optimizer=state['optimizer'];entries=list(optimizer['state'].values());group=optimizer['param_groups']
        require(state['completed_updates']==step and len(entries)==len(group)==1 and int(entries[0]['step'])==step,
            'actual optimizer counter differs')
        require(state['preparation_sha256']==sha(run/'preparation.json')
            and torch.equal(state['initial_queries'],initial),'checkpoint initialization or preparation differs')
        query=state['queries']
        require(query.shape==(128,2048) and query.dtype==torch.float32 and torch.isfinite(query).all(),
            'invalid actual query checkpoint')
        require(group[0]['lr']==5e-5 and tuple(group[0]['betas'])==(.9,.95) and group[0]['weight_decay']==0.,
            'registered optimizer hyperparameters differ')
        for value in entries[0].values():
            if torch.is_tensor(value):require(torch.isfinite(value).all(),'nonfinite saved optimizer state')
        checkpoints.append(dict(step=step,sha256=sha(path),optimizer_step=step,
            changed_FP32_coordinates=int(torch.count_nonzero(query!=initial)),
            changed_BF16_coordinates=int(torch.count_nonzero(query.bfloat16()!=initial.bfloat16())),
            maximum_absolute_query_change=float((query-initial).abs().max()),all_saved_optimizer_tensors_finite=True))
        if step==128:final=query
    actual_tensors=[]
    for step in (1,2,32,64,96,128):
        row=history[step-1];path=execution/row['actual_tensor_file']
        require(sha(path)==row['actual_tensor_sha256'],'saved update tensors changed')
        values=torch.load(path,map_location='cpu',weights_only=True)
        require(all(torch.isfinite(v).all() for v in values.values()),'nonfinite actual training tensor')
        gradient=values['query_gradient']
        require(gradient.shape==initial.shape and torch.count_nonzero(gradient)>0,'saved gradient missing or zero')
        require(torch.equal(values['target_velocity'],values['noise']-values['clean']), 'actual native velocity target differs')
        if row['grid_index']==0:
            require(torch.equal(values['noisy'],values['noise']),'full-noise anchor contains clean target')
        observed=float((values['predicted_velocity'].float()-values['target_velocity'].float()).square().mean())
        tolerance=max(1e-9,abs(row['loss'])*1e-6)
        require(abs(observed-row['loss'])<=tolerance,'actual saved velocity MSE does not reproduce')
        if step==128:require(torch.equal(values['queries_after'],final),'final query checkpoint differs from last actual update')
        actual_tensors.append(dict(step=step,actual_tensor_file_sha256=sha(path),registered_grid_index=row['grid_index'],
            producer_MSE=row['loss'],CPU_recomputed_MSE=observed,allowed_absolute_MSE_difference=tolerance,
            actual_nonzero_gradient_coordinates=int(torch.count_nonzero(gradient)),
            full_noise_writer_input_exactly_noise=row['grid_index']==0,
            all_saved_tensors_finite=True))
    prior=[];fields=[];new_predictions=[];targets=[]
    for i,row in enumerate(report['fields']):
        path=execution/row['actual_tensor_file'];require(sha(path)==row['sha256'],'final endpoint field changed')
        field=torch.load(path,map_location='cpu',weights_only=True)
        before=torch.load(PROBE/'final'/f'{i:03}.pt',map_location='cpu',weights_only=True)
        require(torch.equal(field['noise'],before['noise']) and torch.equal(field['target_velocity'],before['point0_target']),
            'old and new full-noise field probes are not matched')
        u=field['target_velocity'];p=field['predicted_velocity'];p0=before['point0_own']
        mse=lambda a,b:float((a.float()-b.float()).square().mean())
        require(abs(mse(p,u)-row['loss'])<=max(1e-9,abs(row['loss'])*1e-6),'final field MSE differs')
        fields.append(dict(id=row['id'],before_MSE=mse(p0,u),after_MSE=mse(p,u),actual_tensor_file_sha256=sha(path)))
        prior.append(p0);new_predictions.append(p);targets.append(u)
    require(len(fields)==2,'both matched final endpoint fields required')
    alignment=dict(before=metric(*targets,*prior),after=metric(*targets,*new_predictions),
        cases=2,passages=1,split='seen',unbounded_field_measure_not_accuracy=True)
    output.mkdir(parents=True,exist_ok=False);(output/'images').mkdir();(output/'references').mkdir()
    observed=[];summaries={};reference_images={}
    for name,bound in scoring['cohorts'].items():
        folder=run/'assessment'/name
        scored=load(folder/'score/report.json');calibration=load(folder/'calibrate/report.json')
        require(sha(folder/'score/report.json')==bound['report_sha256']
            and sha(folder/'calibrate/report.json')==bound['calibration_sha256'],'offline report changed')
        require(scored['reference_calibration_sha256']==sha(folder/'calibrate/report.json')
            and scored['complete_cohort'] and calibration['complete_cohort'],'original complete calibration required')
        manifest=run/(('final_normal' if name=='initial_normal' else name)+'_references.jsonl')
        references=[json.loads(line) for line in manifest.read_text().splitlines()]
        require(sha(manifest)==scored['manifest_sha256'] and scored['cohort']==[r['id'] for r in references],
            'fixed reference cohort differs')
        rows=load(run/'initial_normal_outputs.json') if name=='initial_normal' else load(execution/name/'outputs.json')
        require(len(rows)==len(references)==len(scored['rows'])==len(calibration['rows']),
            'original generation, reference and OCR row counts differ')
        results=[]
        for i,(result,ref,ocr,clean) in enumerate(zip(rows,references,scored['rows'],calibration['rows'])):
            require(result['id']==ref['id']==ocr['id']==clean['id'] and result['sha256']==ocr['png_sha256'],
                'original generated and scored row differs')
            path=Path(result['path']) if name=='initial_normal' else execution/name/result['path']
            signature=authenticate_native_png(path,result['sha256'],result['result']['tensor_sha256'])
            raw_exact=normalize(ocr['raw_transcript'],ref['language'])==normalize(ref['reference_text'],ref['language'])
            require(raw_exact==ocr['ocr_exact'],'canonical raw OCR equality differs')
            image=f'images/{name}_{i:02}.png';shutil.copyfile(path,output/image)
            target=Path(ref['target']);require(sha(target)==ref['target_sha256'],'original reference image changed')
            if ref['target_sha256'] not in reference_images:
                reference_name=f'references/{len(reference_images):02}.png';shutil.copyfile(target,output/reference_name)
                reference_images[ref['target_sha256']]=reference_name
            item=dict(cohort=name,id=ref['id'],split=ref['split'],language=ref['language'],
                reference_text=ref['reference_text'],raw_OCR_transcript=ocr['raw_transcript'],OCR_exact=raw_exact,
                reference_OCR_exact=clean['ocr_exact'],image=image,PNG_sha256=sha(path),
                actual_tensor_file_sha256=result['result']['tensor_sha256'],pixel_signature=signature,
                reference=reference_images[ref['target_sha256']],reference_sha256=ref['target_sha256'])
            observed.append(item);results.append(item)
        require(len(rows)==len(references)==len(results), 'complete preserved image cohort required')
        summary={s:dict(cases=sum(r['split']==s for r in results),
            OCR_exact=sum(r['split']==s and r['OCR_exact'] for r in results),
            reference_OCR_exact=sum(r['split']==s and r['reference_OCR_exact'] for r in results)) for s in ('train','validation')}
        require(all(summary[s]['cases']==scored['summary'][s]['cases']
            and summary[s]['OCR_exact']==scored['summary'][s]['ocr_exact'] for s in summary),'cohort totals differ')
        summaries[name]=summary
    require(len(observed)==22,'all22 original pages required')
    erased=[r for r in observed if r['cohort']=='final_question_erased']
    require(len(erased)==2 and erased[0]['pixel_signature']==erased[1]['pixel_signature'],
        'erased pair decoder pixels differ')
    count=lambda name,split:summaries[name][split]['OCR_exact']
    copies=lambda name:sum(v['OCR_exact'] for v in summaries[name].values())
    gates=dict(seen_fit=count('final_normal','train')==2,second_noise_seen_fit=count('final_new_noise','train')==2,
        held_pair=count('final_normal','validation')==2,
        question_dependence=count('final_normal','train')>count('final_question_erased','train'),
        passage_dependence=count('final_normal','train')>count('final_passage_erased','train'),
        copying_retention=copies('final_copy')>=copies('initial_copy'))
    require(gates==scoring['developmental_gates'],'registered developmental gates differ')
    require(not torch.cuda.is_initialized(),'independent CPU audit initialized CUDA')
    save_json(output/'report.json',dict(auditor_sha256=sha(__file__),preparation_sha256=sha(run/'preparation.json'),
        source_execution_report_sha256=sha(execution/'report.json'),source_scoring_report_sha256=sha(run/'assessment/result.json'),
        actual_updates=128,registered_and_actual_schedule_match=True,checkpoints=checkpoints,
        independently_verified_saved_update_tensors=actual_tensors,matched_seen_endpoint_fields=fields,
        matched_seen_field_alignment=alignment,cohorts=summaries,developmental_gates=gates,rows=observed,
        actual_new_generations=18,reused_original_baseline_generations=4,all22_original_pages_authenticated=True,
        query_parameters=262144,retained_runtime_parameters=report['retained_runtime_parameters']['total'],
        elapsed_seconds=report['elapsed_seconds'],peak_allocated_vram_GiB=report['peak_allocated_vram_GiB'],
        new_optimizer_updates=0,new_neural_calls=0,raw_image_review_completed=False,
        semantic_learning_validated=False,general_ilm_goal_complete=False))
    print('Audited128 updates, four optimizer checkpoints, six saved gradients and all22 original pages.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=ROOT/'artifacts/native_query_fit_v1')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();audit(args.run.resolve(),args.output.resolve())
