#!/usr/bin/env python3
"""Authenticate and score the complete 384-page native flow comparison on CPU."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path

from ilm.image_edit.baseline import ROOT,save_json,sha
from ilm.image_edit.native_flow_run import identity,validate_checkpoint,schedule,digest
from ilm.image_edit.native_flow_assessment import authenticate_native_png,summarize,developmental_gates
from scripts.evaluate_image_edit_baseline import normalize


def assess(training,output):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('hide CUDA for independent offline assessment')
    import torch
    prepared=json.loads((output/'preparation.json').read_text());meta=json.loads((training/'preparation.json').read_text())
    if identity()!=meta['code_sha256'] or sha(training/'preparation.json')!=prepared['training_preparation_sha256']:
        raise ValueError('registered training implementation/preparation changed')
    for name,digest in prepared['assessment_code_sha256'].items():
        if sha(ROOT/name)!=digest:raise ValueError('registered assessment implementation changed')
    trained=json.loads((training/'report.json').read_text())
    if sha(training/'training_pairs.jsonl')!=meta['training_manifest_sha256']:
        raise ValueError('actual training image manifest changed')
    training_rows=[json.loads(x) for x in (training/'training_pairs.jsonl').read_text().splitlines()]
    checkpoint=training/'checkpoint.pt';state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    plan=json.loads((training/'schedule.json').read_text())
    if plan!=schedule(training_rows) or digest(plan)!=meta['schedule_sha256']:
        raise ValueError('complete training schedule does not reproduce')
    if (validate_checkpoint(state,meta,plan)!=1536 or trained['updates']!=1536 or
        sha(checkpoint)!=trained['checkpoint_sha256'] or trained['metadata']!=meta):
        raise ValueError('complete unchanged final optimization required')
    if len(state['optimizer']['state'])!=len(state['adapters']):raise ValueError('optimizer does not cover all adapter tensors')
    adapter_tensors=len(state['adapters']);adapter_parameters=sum(x.numel() for x in state['adapters'].values());del state
    deployment=training/'deployment';info=json.loads((deployment/'deployment.json').read_text())
    if info['final_checkpoint_sha256']!=trained['checkpoint_sha256'] or sha(deployment/'adapters.safetensors')!=info['weights_sha256']:
        raise ValueError('final deployment differs from completed training')
    if info['steps']!=50 or info['guidance']!=4. or info['output_canvas']!=[1024,1024]:
        raise ValueError('fixed native sampler changed')
    if (adapter_parameters!=info['adapter_parameters'] or adapter_parameters!=meta['adapter_trainable_parameters'] or
        info['runtime_lexical_vocabulary'] or info['external_teacher_or_OCR'] or not info['language_pretraining_retained']):
        raise ValueError('deployed parameter count or information boundary changed')
    source_path=ROOT/'data/visual_qa/cmrc2018_short/train.jsonl'
    if sha(source_path)!=prepared['source_QA_sha256']:raise ValueError('original accepted answers changed')
    source={r['id']:r for r in map(json.loads,source_path.read_text().splitlines())}
    observations=[];summaries={};bindings={}
    for arm in ('initial','final'):
        summaries[arm]={}
        for variant in ('normal','question_erased','passage_erased','copy'):
            name=arm+'_'+variant;folder=output/(name+'_inference')
            model=json.loads((folder/'report.json').read_text());outputs=json.loads((folder/'outputs.json').read_text())
            manifest=training/'evaluation'/(variant+'_references.jsonl');index=training/'evaluation'/(variant+'_inputs.json')
            refs=[json.loads(x) for x in manifest.read_text().splitlines()];inputs=json.loads(index.read_text())
            scores_path=output/(name+'_ocr/report.json');scored=json.loads(scores_path.read_text())
            calibration=Path(prepared['calibrations'][variant]['path'])
            reference=json.loads(calibration.read_text())
            if (sha(manifest)!=prepared['evaluation_files_sha256'][manifest.name] or
                sha(index)!=prepared['evaluation_files_sha256'][index.name] or
                sha(calibration)!=prepared['calibrations'][variant]['sha256']):raise ValueError('fixed assessment data/calibration changed')
            ids=[r['id'] for r in inputs]
            if any([r['id'] for r in rows]!=ids for rows in (refs,outputs,scored['rows'])):
                raise ValueError('complete original-order cohort required')
            if (model['arm']!=('initial_source' if arm=='initial' else 'final') or
                model['learned_adapter_weights_loaded']!=(arm=='final') or
                model['weights_sha256']!=(info['weights_sha256'] if arm=='final' else None) or
                model['exported_weights_file_sha256']!=info['weights_sha256'] or
                model['source']!=info['source_sha256'] or model['code_sha256']!=meta['code_sha256'] or
                model['deployment_sha256']!=sha(deployment/'deployment.json') or
                model['inputs_sha256']!=sha(index) or model['complete_generations']!=len(ids) or
                any(model[k] for k in ('training_manifest_loaded','optimizer_loaded','target_or_transcript_used',
                    'changing_text_condition','output_repair','external_teacher_or_OCR'))):
                raise ValueError('complete unchanged input-only generator required')
            if (scored['stage']!='score' or reference['stage']!='calibrate' or not scored['complete_cohort'] or scored['manifest_sha256']!=sha(manifest) or
                scored['reference_calibration_sha256']!=sha(calibration) or scored['engine']!=reference['engine'] or
                scored['implementation']!=reference['implementation']):raise ValueError('owning frozen evaluator calibration required')
            if not all(model['removal'][k] for k in ('vocabulary_table_removed','output_word_head_removed','tokenizer_removed')):
                raise ValueError('deployed lexical modules were not removed')
            for file,digest in scored['implementation'].items():
                if sha(ROOT/file)!=digest:raise ValueError('offline evaluator changed')
            group=[]
            for given,out,ref,score in zip(inputs,outputs,refs,scored['rows']):
                if (set(given)!={'id','input','input_sha256','seed'} or
                    any(out[k]!=given[k] for k in ('id','seed','input_sha256')) or
                    sha(index.parent/given['input'])!=given['input_sha256'] or
                    out['sha256']!=score['png_sha256'] or out['result']['sha256']!=out['sha256'] or
                    score['expected']!=ref['reference_text'] or sha(manifest.parent/ref['target'])!=ref['target_sha256']):
                    raise ValueError('actual input, output or reference binding differs')
                signature=authenticate_native_png(folder/out['path'],out['sha256'],out['result']['tensor_sha256'])
                text=normalize(score['raw_transcript'],ref['language'])
                canonical=text==normalize(ref['reference_text'],ref['language'])
                if canonical!=score['ocr_exact'] or text!=score['normalized']:raise ValueError('canonical OCR score does not reproduce')
                accepted=([a['text'] for a in source[ref['id']]['references']] if ref['task']=='qa' else [ref['reference_text']])
                if ref['reference_text'] not in accepted:raise ValueError('canonical answer is not source accepted')
                row=dict(arm=arm,variant=variant,id=ref['id'],split=ref['split'],language=ref['language'],pair=ref.get('pair'),
                    input_sha256=given['input_sha256'],seed=given['seed'],answer_png=str((folder/out['path']).relative_to(output)),
                    png_sha256=out['sha256'],tensor_sha256=out['result']['tensor_sha256'],actual_pixel_signature=signature,
                    raw_transcript=score['raw_transcript'],canonical_exact=canonical,
                    accepted_exact=text in {normalize(a,ref['language']) for a in accepted},source_accepted_answers=accepted)
                group.append(row);observations.append(row)
            if variant=='question_erased':
                for i in range(0,32,2):
                    a,b=group[i:i+2]
                    if any(a[k]!=b[k] for k in ('pair','input_sha256','seed','actual_pixel_signature','png_sha256')):
                        raise ValueError('identical erased inputs and seeds do not reproduce exact decoded pixels')
            summaries[arm][variant]=summarize(group,paired=variant!='copy')
            bindings[name]=dict(inference_report_sha256=sha(folder/'report.json'),output_index_sha256=sha(folder/'outputs.json'),
                OCR_report_sha256=sha(scores_path),reference_calibration_sha256=sha(calibration))
    if Counter((r['arm'],r['variant']) for r in observations)!=Counter({(a,v):n for a in ('initial','final')
        for v,n in [('normal',64),('question_erased',32),('passage_erased',32),('copy',64)]}):
        raise ValueError('all384 fixed new generations required')
    # Resumed attempts may carry later monitoring pages; authenticate exactly
    # the fixed steps/IDs, never choose images by appearance.
    previews=list(training.glob('attempt_*/previews/*/record.json'))
    preview_rows=[r for r in training_rows if r['task']=='qa'][:2]
    preview_expected={r['id']:(20269490+i,r['input_sha256']) for i,r in enumerate(preview_rows)}
    verified_previews={}
    for path in previews:
        record=json.loads(path.read_text());key=(record['after_update'],record['id'])
        if (record['id'] not in preview_expected or
            (record['seed'],record['input_sha256'])!=preview_expected[record['id']] or
            not record['monitoring_only'] or record['checkpoint_selected_by_quality']):
            raise ValueError('fixed training preview input, seed or selection policy differs')
        signature=authenticate_native_png(path.parent/'answer.png',record['result']['sha256'],record['result']['tensor_sha256'])
        if key in verified_previews and verified_previews[key]!=signature:raise ValueError('repeated monitoring draw differs after recovery')
        verified_previews[key]=signature
    if len(verified_previews)!=24 or {k[0] for k in verified_previews}!=set(range(128,1537,128)):
        raise ValueError('all24 fixed final training previews required')
    gates=developmental_gates(summaries)
    result=dict(summaries=summaries,rows=observations,bindings=bindings,prospective_developmental_gates=gates,
        developmental_rule_met=all(gates.values()),actual_new_generations=384,actual_qa_generations=256,actual_copy_generations=128,
        completed_training_updates=1536,actual_adapter_tensors=adapter_tensors,verified_fixed_training_previews=len(verified_previews),
        final_checkpoint_sha256=trained['checkpoint_sha256'],deployment_sha256=sha(deployment/'deployment.json'),
        preparation_sha256=sha(output/'preparation.json'),training_report_sha256=sha(training/'report.json'),
        all_PNGs_authenticated_against_actual_decoder_tensors=True,all_question_erased_pairs_bit_exact=True,
        retained_language_pretraining=True,raw_review_completed=False,semantic_learning_validated=False,
        reliable_writing_qualified=False,general_ilm_goal_complete=False)
    save_json(output/'assessment.json',result)
    print(json.dumps(dict(summaries=summaries,gates=gates),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--training-run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    assess(args.training_run.resolve(),args.output.resolve())
