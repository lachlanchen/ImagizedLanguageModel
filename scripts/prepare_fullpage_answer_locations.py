#!/usr/bin/env python3
"""Locate source answer spans in the exact existing full-page QA images.

CPU-only supervision preparation for a visual-reading diagnostic. This does
not predict answers, train a model, or qualify copying or free generation.
References enter only this offline annotation program.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path

from PIL import Image,ImageDraw
import numpy as np

from ilm.image_edit.baseline import ROOT,save_json,sha
from ilm.image_edit.data import display_text,font_and_coverage,line_units,page

PAGES=ROOT/'artifacts/image_edit_learning/pages_v2'
SOURCE=ROOT/'data/visual_qa/cmrc2018_short/train.jsonl'


def line_positions(text,font):
    """Replay physical wrapping, retaining offsets in the complete prompt."""
    lines=[];offset=0
    for paragraph in text.split('\n'):
        line='';start=offset
        for unit in line_units(paragraph):
            if font.getlength(unit)>960:raise ValueError('source word exceeds page width')
            if line and font.getlength(line+unit)>960:
                lines.append(dict(text=line,start=start,end=start+len(line)))
                start+=len(line);line=''
            line+=unit
        lines.append(dict(text=line,start=start,end=start+len(line)))
        offset+=len(paragraph)+1
    canvas=Image.new('RGB',(1024,1024),'white');draw=ImageDraw.Draw(canvas)
    for i,line in enumerate(lines):
        if text[line['start']:line['end']]!=line['text']:raise ValueError('physical wrap changed source offsets')
        line['box']=list(draw.textbbox((32,32+48*i),line['text'],font=font,anchor='lt'))
    return lines


def locate(lines,start,end,font):
    """Return advance-coordinate envelopes, possibly spanning several lines.

    These supervise locations. They are not exact per-glyph ink segmentation
    or a guarantee that cutting rectangles reconstructs an answer perfectly.
    """
    pieces=[]
    for line_index,line in enumerate(lines):
        a=max(start,line['start']);b=min(end,line['end'])
        if a>=b:continue
        prefix=line['text'][:a-line['start']];through=line['text'][:b-line['start']]
        left=32+font.getlength(prefix);right=32+font.getlength(through)
        box=[math.floor(left),line['box'][1],math.ceil(right),line['box'][3]]
        if not 0<=box[0]<box[2]<=1024 or not 0<=box[1]<box[3]<=1024:
            raise ValueError('answer location outside full page')
        pieces.append(dict(line=line_index,box=box,prompt_start=a,prompt_end=b,
            horizontal_advance=[float(left),float(right)]))
    if sum(p['prompt_end']-p['prompt_start'] for p in pieces)!=end-start:
        raise ValueError('answer contains an unrepresented newline or missing source region')
    return pieces


def prepare(output):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('hide CUDA for this CPU-only preparation')
    prepared=json.loads((PAGES/'preparation.json').read_text())
    if sha(SOURCE)!=prepared['source_sha256'] or sha(PAGES/'pairs.jsonl')!=prepared['manifest_sha256']:
        raise ValueError('existing source or full-page image manifest changed')
    for name,digest in prepared['implementation_sha256'].items():
        if sha(ROOT/name)!=digest:raise ValueError('owning page-rendering implementation changed')
    original=[r for r in map(json.loads,(PAGES/'pairs.jsonl').read_text().splitlines()) if r['task']=='qa']
    sources={r['id']:r for r in map(json.loads,SOURCE.read_text().splitlines())}
    cases=[]
    for row in original:
        source=sources[row['id']];context=display_text(source['context']);question=display_text(source['question'])
        prompt=f'资料：{context}\n问题：{question}\n回答：'
        if row['reference_text'] not in [r['text'] for r in source['references']]:
            raise ValueError('canonical answer is not source accepted')
        cases.append(dict(row=row,context=context,prompt=prompt,reference=row['reference_text']))
    font,provenance=font_and_coverage(cases)
    for key in ('font_file_sha256','font_face_index','font_size'):
        if provenance[key]!=prepared['font'][key]:raise ValueError('original rendering font changed')
    if Counter(c['row']['split'] for c in cases)!=Counter(train=256,validation=32):
        raise ValueError('all existing 288 natural QA pages required')
    output.mkdir(parents=True,exist_ok=False)
    rows=[];multiple_occurrences=0;multiline_answers=0
    for case in cases:
        row=case['row'];path=PAGES/row['input']
        if sha(path)!=row['input_sha256']:raise ValueError('actual input PNG bytes changed')
        rendered,geometry=page(case['prompt'],(1024,1024),font)
        with Image.open(path) as actual:
            if actual.mode!='RGB' or not np.array_equal(np.asarray(actual),np.asarray(rendered)):
                raise ValueError('source text does not reproduce exact actual input pixels')
        if geometry!=row['input_geometry']:raise ValueError('source page geometry changed')
        lines=line_positions(case['prompt'],font)
        if [line['box'] for line in lines]!=geometry['boxes']:raise ValueError('line offset replay differs from actual rendering')
        spans=[];cursor=0
        while True:
            offset=case['context'].find(case['reference'],cursor)
            if offset<0:break
            start=3+offset;end=start+len(case['reference'])
            if case['prompt'][start:end]!=case['reference']:raise ValueError('answer span entered question/instruction text')
            pieces=locate(lines,start,end,font)
            spans.append(dict(start=start,end=end,pieces=pieces))
            cursor=offset+1
        if not spans:raise ValueError('canonical answer absent from actual displayed passage')
        multiple_occurrences+=len(spans)>1
        multiline_answers+=any(len(s['pieces'])>1 for s in spans)
        rows.append(dict(id=row['id'],split=row['split'],article=row['article'],pair=row['pair'],
            input=os.path.relpath(path,output),input_sha256=row['input_sha256'],
            target_text_sha256=hashlib.sha256(case['reference'].encode()).hexdigest(),
            targets=spans,source_image_size=[1024,1024],annotation_kind='canonical_answer_advance_envelopes',
            candidate_occurrences_all_in_passage=True))
    train_articles={r['article'] for r in rows if r['split']=='train'}
    held_articles={r['article'] for r in rows if r['split']=='validation'}
    if train_articles&held_articles:raise ValueError('article leakage')
    files={}
    for split in ('train','validation'):
        path=output/(split+'_locations.jsonl')
        path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows if r['split']==split))
        files[path.name]=sha(path)
    inputs=output/'input_images.json'
    save_json(inputs,[{k:r[k] for k in ('id','input','input_sha256')} for r in rows]);files[inputs.name]=sha(inputs)
    report=dict(implementation_sha256=sha(Path(__file__)),source_QA_sha256=sha(SOURCE),
        source_image_manifest_sha256=sha(PAGES/'pairs.jsonl'),source_preparation_sha256=sha(PAGES/'preparation.json'),
        original_renderer_sha256=prepared['implementation_sha256'],original_font=provenance,
        complete_QA_pages=len(rows),counts=dict(Counter(r['split'] for r in rows)),
        QA_articles=dict(train=len(train_articles),validation=len(held_articles)),articles_disjoint=True,
        all288_input_PNGs_reproduced_pixel_exact=True,all_location_line_boxes_equal_original_geometry=True,
        examples_with_multiple_canonical_occurrences=multiple_occurrences,
        examples_with_multiline_answer_envelopes=multiline_answers,
        all_canonical_occurrences_marginalizable=True,answer_coordinates_not_in_input_index=True,
        supervision_files_contain_no_answer_strings_or_character_IDs=True,
        reference_access_in_offline_preparation=True,exact_glyph_segmentation_claim=False,
        source_pixel_answer_reconstruction_tested=False,files_sha256=files,
        neural_models_loaded=False,optimizer_updates=0,learned_localizer_implemented=False,
        language_learning_validated=False,general_ilm_goal_complete=False)
    save_json(output/'report.json',report)
    print(json.dumps({k:report[k] for k in ('complete_QA_pages','counts','QA_articles',
        'examples_with_multiple_canonical_occurrences','examples_with_multiline_answer_envelopes',
        'all288_input_PNGs_reproduced_pixel_exact','neural_models_loaded')},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    prepare(p.parse_args().output.resolve())
