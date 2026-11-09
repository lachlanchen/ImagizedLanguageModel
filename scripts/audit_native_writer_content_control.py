#!/usr/bin/env python3
"""Independent CPU recomputation of the complete supplied-content control."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import unicodedata


ROOT=Path(__file__).resolve().parents[1]


def digest(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):value.update(block)
    return value.hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def require(value,message):
    if not value:raise ValueError(message)


def normalize(text,language):
    text=unicodedata.normalize('NFC',text)
    return ' '.join(text.split()) if language=='en' else ''.join(text.split())


def equal_fields(a,b):
    import torch
    return set(a)==set(b) and all((a[k] is None and b[k] is None) if a[k] is None
        else torch.is_tensor(b[k]) and torch.equal(a[k],b[k]) for k in a)


def audit(run,output):
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','hide CUDA for this CPU audit')
    require(not output.exists(),'preserve previous independent audits')
    meta=load(run/'preparation.json');execution=run/'execution';model=load(execution/'report.json')
    require(load(run/'coordinator.json')['phase']=='completed'
        and load(execution/'runtime.json')['phase']=='completed','complete model and separate scoring children required')
    for name,digest_expected in {**meta['code_sha256'],**meta['bound_sources_sha256']}.items():
        require(digest(ROOT/name)==digest_expected,'registered source changed: '+name)
    for name,digest_expected in meta['prepared_files_sha256'].items():
        require(digest(run/name)==digest_expected,'prepared artifact changed: '+name)
    require(model['preparation_sha256']==digest(run/'preparation.json')
        and model['outputs_sha256']==digest(execution/'outputs.json'),'execution receipt binding differs')
    require(model['complete_generations']==18 and model['supplied_content_generations']==16
        and model['execution_replays']==2 and model['optimizer_updates']==0
        and not model['reference_answer_images_used_by_generator']
        and model['supplied_answer_text'] and model['tokenizer_and_input_word_table_retained']
        and not model['unused_vocabulary_output_head_called'],'supplied-content diagnostic boundary differs')
    require(model['peak_allocated_vram_GiB']<=23 and model['elapsed_seconds']<1800,'registered resource bound exceeded')
    cases=load(run/'cases.json');require(len(cases)==len({r['id'] for r in cases})==4,'four unique fixed cases required')
    expected=[]
    for arm in ('source','adapted'):
        expected.append((arm,'replay',cases[0]['id'],1.))
        for guidance in (1.,4.):
            expected.extend((arm,'supplied_content',r['id'],guidance) for r in cases)
    generated=load(execution/'outputs.json')
    require([(r['arm'],r['kind'],r['id'],r['guidance']) for r in generated]==expected,'fixed complete generation order differs')
    assessment=load(run/'assessment.json');observed=assessment['rows']
    require(assessment['model_report_sha256']==digest(execution/'report.json')
        and [(r['arm'],r['kind'],r['id'],r['guidance']) for r in observed]==expected,
        'complete independent OCR coverage/model binding required')
    calibration=load(run/'calibration.json');by_id={r['id']:r for r in cases}
    require(assessment['calibration_sha256']==digest(run/'calibration.json')
        and [r['id'] for r in calibration['rows']]==[r['id'] for r in cases],'owning calibration required')
    for row in calibration['rows']:
        case=by_id[row['id']]
        require(row['expected']==case['reference_text'] and row['language']==case['language']
            and row['png_sha256']==case['target_sha256']==digest(case['target']), 'reference calibration image/text differs')
        normalized=normalize(row['raw_transcript'],row['language'])
        require(normalized==row['normalized'] and row['ocr_exact']==(normalized==normalize(row['expected'],row['language'])),
                'reference exact-match computation differs')
    require(calibration['ocr_exact']==sum(r['ocr_exact'] for r in calibration['rows'])
        ==assessment['calibration_exact']==4,'all four original references must pass calibration')
    import torch
    import numpy as np
    from PIL import Image
    torch.set_num_threads(4)
    first_fields={};evidence=[];total_calls=0;total_elements=0
    replays=load(run/'replays.json')
    for row,ocr in zip(generated,observed):
        case=by_id[row['id']];path=execution/row['path'];tensor=path.with_name('actual_tensors.pt')
        require(digest(path)==row['sha256']==ocr['png_sha256']
            and digest(tensor)==row['result']['tensor_sha256'],'complete original generated PNG/tensor binding differs')
        value=torch.load(tensor,map_location='cpu',weights_only=True)
        require(set(value)=={'pixels','forward_inputs'} and value['pixels'].shape==(1,3,1024,1024)
            and torch.isfinite(value['pixels']).all(),'finite complete actual decoder tensor required')
        pixels=torch.clamp(127.5*value['pixels'][0].permute(1,2,0)+128.,0,255).to(torch.uint8).numpy()
        with Image.open(path) as image:
            require(image.size==(1024,1024) and np.array_equal(np.asarray(image.convert('RGB')),pixels),
                    'unrepaired original PNG must equal actual decoder pixels')
        calls=row['denoiser_batch_sizes'];require(calls==[1 if row['guidance']==1. else 2]*50,'native sampler branch accounting differs')
        total_calls+=len(calls);total_elements+=sum(calls)
        if row['kind']=='replay':
            old=replays[row['arm']]
            require(old==row['original'] and digest(old['png'])==old['png_sha256']
                and digest(old['tensors'])==old['tensors_sha256'] and row['seed']==old['seed'],'original replay binding differs')
            original=torch.load(old['tensors'],map_location='cpu',weights_only=True)
            require(torch.equal(original['pixels'],value['pixels'])
                and equal_fields(original['forward_inputs'],value['forward_inputs']),'independent native replay differs')
            with Image.open(old['png']) as image:
                require(np.array_equal(np.asarray(image.convert('RGB')),pixels),'original replay PNG pixels differ')
        else:
            prompt=case['content_prompt']
            require(row['seed']==case['content_seed'] and row['input_sha256']==case['input_sha256']==digest(case['input'])
                and hashlib.sha256(prompt.encode()).hexdigest()==row['content_prompt_sha256']==case['content_prompt_sha256'],
                'registered supplied-content case, seed or prompt differs')
            if row['id'] not in first_fields:first_fields[row['id']]=value['forward_inputs']
            require(equal_fields(first_fields[row['id']],value['forward_inputs']),
                'reader inputs must be bit-exact for the same content across guidance and writer-adapter arms')
        normalized=normalize(ocr['raw_transcript'],case['language'])
        require(ocr['expected']==case['reference_text'] and ocr['language']==case['language']
            and ocr['normalized']==normalized and ocr['ocr_exact']==(normalized==normalize(case['reference_text'],case['language'])),
                'generated-page exact-match computation differs')
        evidence.append(dict(arm=row['arm'],kind=row['kind'],guidance=row['guidance'],id=row['id'],
            path=row['path'],png_sha256=digest(path),tensor_sha256=digest(tensor),raw_transcript=ocr['raw_transcript'],
            ocr_exact=ocr['ocr_exact'],actual_decoded_pixels_authenticated=True))
    require(total_calls==model['denoiser_calls']==900 and total_elements==model['denoiser_batch_elements']==1300,
            'whole-attempt sampler totals differ')
    summary={}
    for arm in ('source','adapted'):
        for guidance in (1.,4.):
            rows=[r for r in observed if r['arm']==arm and r['guidance']==guidance and r['kind']=='supplied_content']
            summary[f'{arm}_g{int(guidance)}']=dict(cases=len(rows),ocr_exact=sum(r['ocr_exact'] for r in rows),
                per_language={lang:dict(cases=sum(r['language']==lang for r in rows),
                    ocr_exact=sum(r['language']==lang and r['ocr_exact'] for r in rows)) for lang in ('zh','en')})
    require(summary==assessment['summary'],'independent complete cell counts differ')
    output.mkdir(parents=True,exist_ok=False)
    report=dict(auditor_sha256=digest(__file__),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        preparation_sha256=digest(run/'preparation.json'),execution_report_sha256=digest(execution/'report.json'),
        assessment_sha256=digest(run/'assessment.json'),actual_original_generations=18,actual_supplied_content_generations=16,
        independently_bit_exact_replays=2,all4_case_reader_inputs_bit_exact_across_guidance_and_writer_arms=True,
        all18_actual_PNGs_tensor_authenticated=True,source_and_preparation_bindings_unchanged=True,
        denoiser_calls=total_calls,denoiser_batch_elements=total_elements,reference_OCR_exact=4,reference_cases=4,
        summary=summary,original_images=evidence,model_calls_added_by_audit=0,optimizer_updates=0,
        raw_image_review_completed=False,supplied_answer_text=True,autonomous_QA_validated=False,
        semantic_learning_validated=False,general_ilm_goal_complete=False)
    (output/'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();audit(args.run.resolve(),args.output.resolve())
