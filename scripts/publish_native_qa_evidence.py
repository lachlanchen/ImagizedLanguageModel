#!/usr/bin/env python3
"""Package the complete authenticated QA export with recorded raw inspection.

Copies original images exactly. Makes local report paths relative, retaining
the original hashes and the public hashes separately. No model calls or
manual answer-score replacements are performed.
"""
import argparse
import copy
import json
from pathlib import Path
import shutil

from ilm.image_edit.baseline import ROOT,save_json,sha

REVIEWS=(('native_qa_initial_normal_raw_review_v1','all64_original_normal_answer_pages_viewed'),
    ('native_qa_final_normal_raw_review_v1','all64_original_normal_answer_pages_viewed'),
    ('native_qa_erasure_raw_review_v1','all128_original_erasure_pages_viewed'),
    ('native_qa_copy_raw_review_v1','all128_original_copy_pages_viewed'))


def relative_paths(value):
    if isinstance(value,dict):return {k:relative_paths(v) for k,v in value.items()}
    if isinstance(value,list):return [relative_paths(v) for v in value]
    if isinstance(value,str) and value.startswith(str(ROOT)+'/'):return value[len(str(ROOT))+1:]
    if isinstance(value,str) and value.startswith(str(Path.home())+'/'):return '[local-home]/'+value[len(str(Path.home()))+1:]
    return value


def publish(source,output):
    result=json.loads((source/'result.json').read_text())
    if result['complete_output_images']!=384 or result['complete_training_images']!=128:
        raise ValueError('complete registered experiment required')
    expected={}
    for row in result['rows']:
        for arm in ('initial','final'):
            group=arm+'_'+row['variant']
            expected[(group,row['id'])]=(sha(source/row['images'][arm]),row['observations'][arm]['actual_tensor_sha256'])
    if len(expected)!=384:raise ValueError('all384 original assessment cases required')
    observed={};review_records=[]
    for folder,flag in REVIEWS:
        root=ROOT/'artifacts'/folder;receipt=json.loads((root/'receipt.json').read_text())
        inspection=json.loads((root/'raw_inspection.json').read_text())
        key=json.loads((root/'private_key.json').read_text())
        if not inspection[flag] or inspection['receipt_sha256']!=sha(root/'receipt.json'):
            raise ValueError('complete bound raw inspection required')
        if receipt['private_key_sha256']!=sha(root/'private_key.json'):
            raise ValueError('raw-review source binding changed')
        for name,digest in receipt['packet_files_sha256'].items():
            if sha(root/'packet'/name)!=digest:raise ValueError('actual viewed packet changed')
        for case in key['cases']:
            identity=(case['group'],case['source_id'])
            if identity in observed:raise ValueError('duplicate review case')
            observed[identity]=(case['answer_sha256'],case['actual_tensor_sha256'])
        review_records.append(dict(review=folder,cases=receipt['cases'],receipt_sha256=sha(root/'receipt.json'),
            inspection_sha256=sha(root/'raw_inspection.json'),manual_exact_answer_accuracy_assigned=False))
    if observed!=expected:raise ValueError('raw review does not cover every complete exported model image')
    hashes={**result['files_sha256'],**result['reports_sha256']}
    for name,digest in hashes.items():
        if Path(name).is_absolute() or '..' in Path(name).parts or sha(source/name)!=digest:
            raise ValueError('source export file changed: '+name)
    output.mkdir(parents=True,exist_ok=False)
    for file in source.rglob('*'):
        if file.is_file():
            target=output/file.relative_to(source);target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(file,target)
    redacted=[]
    for name in result['reports_sha256']:
        path=output/name
        if path.suffix!='.json':continue
        before=json.loads(path.read_text());after=relative_paths(before)
        if before!=after:save_json(path,after);redacted.append(name)
    text=(output/'README.md').read_text()
    text=text.replace('Numerical assessment is complete; raw-image review is pending.',
        'Numerical assessment and raw visual inspection of all384 answer pages are complete. No manual transcription or answer-score replacement is claimed.')
    text=text.replace('unmodified source reports and independent training verification accompany them.',
        'source reports with local filesystem paths made relative and independent training verification accompany them. Original report hashes and public report hashes are retained separately.')
    (output/'README.md').write_text(text+'\n[Raw visual-review coverage](raw_visual_review.json) binds all384 answer cases.\n')
    save_json(output/'raw_visual_review.json',dict(complete_cases=384,records=review_records,
        all_original_answer_PNGs_covered=True,all_exported_actual_tensor_hashes_match_review=True,
        manual_exact_answer_accuracy_assigned=False,independent_or_perfect_blinding_claim=False,
        semantic_learning_validated=False,reliable_writing_qualified=False))
    public=copy.deepcopy(result)
    public.update(source_export_result_sha256=sha(source/'result.json'),original_reports_sha256=result['reports_sha256'],
        reports_sha256={name:sha(output/name) for name in result['reports_sha256']},
        files_sha256={name:sha(output/name) for name in result['files_sha256']},
        raw_review_completed=True,raw_visual_review_sha256=sha(output/'raw_visual_review.json'),
        raw_review_scope='All384 original answers visually inspected; no manual transcription, score repair or independent blinding claim.',
        public_reports_with_relative_local_paths=redacted,publisher_sha256=sha(Path(__file__)))
    save_json(output/'result.json',public)
    if '/home/' in (output/'result.json').read_text():raise ValueError('private absolute path in public result')
    for name in public['reports_sha256']:
        if '/home/' in (output/name).read_text():raise ValueError('unhandled local path in public report: '+name)
    for name,digest in hashes.items():
        if Path(name).suffix.lower()=='.png' and sha(output/name)!=digest:
            raise ValueError('public export changed actual image pixels/bytes')
    print(json.dumps(dict(complete_answer_pages=384,complete_training_pages=128,raw_visual_review_complete=True,
        private_path_reports_made_relative=redacted,all_original_PNG_bytes_unchanged=True),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source-export',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    publish(args.source_export.resolve(),args.output.resolve())
