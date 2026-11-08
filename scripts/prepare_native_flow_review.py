#!/usr/bin/env python3
"""Retain complete native1024-square pages in an opaque-label review packet.

Every contact-sheet answer occupies its original1024-by1024 pixels. No
resizing, cropping, contrast adjustment, transcription or score is applied.
Training monitoring and completed assessment cohorts are distinct modes.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import shutil

from ilm.image_edit.baseline import ROOT, save_json, sha
from ilm.image_edit.native_flow_assessment import authenticate_native_png

SIZES = dict(normal=64, question_erased=32, passage_erased=32, copy=64)


def assessment_cases(run, names):
    preparation = json.loads((run/'preparation.json').read_text())
    training = Path(preparation['training_run'])
    rows, groups = [], {}
    for name in names:
        arm, variant = name.split('_',1)
        if arm not in ('initial','final') or variant not in SIZES or name in groups:
            raise ValueError('unique original initial/final cohort name required')
        folder = run/(name+'_inference')
        if not (folder/'report.json').exists():
            raise ValueError('wait for the complete original cohort before review packaging: '+name)
        report = json.loads((folder/'report.json').read_text())
        path = folder/'outputs.json';index = training/'evaluation'/(variant+'_inputs.json')
        given = json.loads(index.read_text());actual = json.loads(path.read_text())
        if (len(actual)!=SIZES[variant] or report['complete_generations']!=len(actual)
                or report['inputs_sha256']!=sha(index) or sha(index)!=preparation['evaluation_files_sha256'][index.name]
                or [r['id'] for r in actual]!=[r['id'] for r in given]):
            raise ValueError('complete fixed original-order native cohort required')
        if report['arm']!=('initial_source' if arm=='initial' else 'final'):
            raise ValueError('source model arm differs')
        groups[name] = dict(index=str(path),index_sha256=sha(path),native_report_sha256=sha(folder/'report.json'),cases=len(actual))
        for i,(row,input_row) in enumerate(zip(actual,given)):
            if any(row[k]!=input_row[k] for k in ('id','input_sha256','seed')):
                raise ValueError('original input or seed binding differs')
            if row['sha256']!=row['result']['sha256']:
                raise ValueError('actual answer binding differs')
            rows.append(dict(group=name,source_id=row['id'],source_position=i,source_answer=str((folder/row['path']).resolve()),
                answer_sha256=row['sha256'],actual_tensor_sha256=row['result']['tensor_sha256'],
                seed=row['seed'],input_sha256=row['input_sha256']))
    return rows,groups


def monitoring_cases(training, audit):
    report = json.loads((audit/'report.json').read_text())
    if (report['actual_optimizer_updates']!=1536 or report['verified_fixed_monitoring_previews']!=24
            or report['completed_report_sha256']!=sha(training/'report.json')):
        raise ValueError('complete audited training-monitoring pages required')
    rows=[]
    for i,row in enumerate(report['previews']):
        record = json.loads((training/row['image']).with_name('record.json').read_text())
        if record['result']['sha256']!=row['PNG_sha256'] or record['result']['tensor_sha256']!=row['actual_tensor_file_sha256']:
            raise ValueError('actual monitoring binding differs')
        rows.append(dict(group='training_monitoring',source_id=f'{row["step"]:04}/'+row['id'],source_position=i,
            source_answer=str((training/row['image']).resolve()),answer_sha256=row['PNG_sha256'],
            actual_tensor_sha256=row['actual_tensor_file_sha256'],seed=row['seed'],input_sha256=record['input_sha256']))
    return rows,dict(training_monitoring=dict(audit_report_sha256=sha(audit/'report.json'),cases=24))


def prepare(rows,groups,output,seed,*,monitoring):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='': raise ValueError('hide CUDA for offline image review preparation')
    import torch
    from PIL import Image,ImageDraw,ImageFont
    torch.set_num_threads(4)
    if not rows or len({(r['group'],r['source_id']) for r in rows})!=len(rows):
        raise ValueError('complete unique source cases required')
    for row in rows:
        path=Path(row['source_answer'])
        authenticate_native_png(path,row['answer_sha256'],row['actual_tensor_sha256'])
        with Image.open(path) as image:
            if image.size!=(1024,1024) or image.mode!='RGB' or image.info:
                raise ValueError('complete metadata-free native RGB1024-square page required')
    random.Random(seed).shuffle(rows)
    output.mkdir(parents=True,exist_ok=False)
    packet=output/'packet';(packet/'images').mkdir(parents=True);(packet/'pages').mkdir()
    public,key=[],[]
    for i,row in enumerate(rows,1):
        id=f'R{i:04}';name='images/'+id+'.png'
        shutil.copyfile(row['source_answer'],packet/name)
        if sha(packet/name)!=row['answer_sha256']: raise ValueError('original PNG copy differs')
        public.append(dict(review_id=id,image=name,image_sha256=row['answer_sha256'],size=[1024,1024]))
        key.append(dict(review_id=id,**row))
    save_json(packet/'index.json',public)
    with (packet/'annotations.jsonl').open('x') as handle:
        for row in public:
            handle.write(json.dumps(dict(review_id=row['review_id'],transcription=None,legibility='unreviewed',
                uncertain_regions=[],extra_writing=None,notes=''))+'\n')
    (packet/'READ_FIRST.txt').write_text(
        'Inspect only the visible writing and mark uncertainty explicitly. Do not infer missing words.\n'
        'Each PNG is a complete original model page. Contact sheets add external review IDs only.\n'
        'Do not open the source key or reference answers during this legibility pass.\n'
        'Retain extra marks, uncertain glyphs and blank pages. Do not repair or omit them.\n'
        'This packet withholds condition labels. It does not establish independent or perfect blinding.\n'
        'Image review alone supplies no answer-accuracy score.\n')
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf',20)
    contacts=[]
    for offset in range(0,len(public),4):
        selected=public[offset:offset+4]
        page=Image.new('RGB',(2048,((len(selected)+1)//2)*1060),'white');draw=ImageDraw.Draw(page)
        for i,row in enumerate(selected):
            x,y=(i%2)*1024,(i//2)*1060
            draw.rectangle((x,y,x+1023,y+35),fill='#e5e9ed')
            draw.text((x+12,y+6),row['review_id'],font=font,fill='black')
            with Image.open(packet/row['image']) as image:
                page.paste(image,(x,y+36))
                if page.crop((x,y+36,x+1024,y+1060)).tobytes()!=image.tobytes():
                    raise ValueError('contact sheet changed or clipped actual model pixels')
        name=f'pages/page_{offset//4+1:03}.png';page.save(packet/name);contacts.append(name)
    save_json(output/'private_key.json',dict(groups=groups,seed=seed,cases=key))
    packet_files={str(p.relative_to(packet)):sha(p) for p in sorted(packet.rglob('*')) if p.is_file()}
    receipt=dict(implementation_sha256=sha(Path(__file__)),authenticator_sha256=sha(ROOT/'ilm/image_edit/native_flow_assessment.py'),
        cases=len(rows),source_groups=len(groups),contact_sheets=len(contacts),source_mode='training_monitoring' if monitoring else 'completed_assessment_cohorts',
        seed=seed,private_key_sha256=sha(output/'private_key.json'),packet_files_sha256=packet_files,
        all_source_PNG_bytes_unchanged=True,all_contact_sheet_answer_regions_pixel_identical=True,
        complete_original1024_square_pages=True,resize_or_crop_or_contrast_applied=False,
        expected_answers_loaded=False,source_ids_and_group_names_in_private_key=True,
        preparer_has_source_access=True,independent_or_perfect_blinding_claim=False,
        raw_annotations_completed=False,quality_score_computed=False,neural_models_loaded=0,
        training_monitoring_only=monitoring,heldout_assessment_completed=False,general_ilm_goal_complete=False)
    if torch.cuda.is_initialized(): raise ValueError('CPU preparation initialized CUDA')
    save_json(output/'receipt.json',receipt)
    print(json.dumps({k:receipt[k] for k in ('cases','contact_sheets','source_mode','all_contact_sheet_answer_regions_pixel_identical')},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('assessment','monitoring'),required=True)
    parser.add_argument('--assessment',type=Path);parser.add_argument('--cohort',action='append')
    parser.add_argument('--training-run',type=Path);parser.add_argument('--audit',type=Path)
    parser.add_argument('--seed',type=int,default=20269990);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.mode=='assessment':rows,groups=assessment_cases(args.assessment.resolve(),args.cohort or [])
    else:rows,groups=monitoring_cases(args.training_run.resolve(),args.audit.resolve())
    prepare(rows,groups,args.output.resolve(),args.seed,monitoring=args.mode=='monitoring')
