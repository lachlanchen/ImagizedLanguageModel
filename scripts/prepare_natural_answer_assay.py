#!/usr/bin/env python3
"""Reserve the next complete-answer cohort before natural-writer outcomes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image,ImageDraw

from ilm.image_edit.baseline import ROOT,sha,save_json
from scripts.score_pixel_language_first_blocks import require,host_resources

OWN='scripts/prepare_natural_answer_assay.py'
PROTOCOL='references/natural_answer_assay_protocol.md'


def load(p):return json.loads(Path(p).read_text())
def rel(p):return str(Path(p).resolve().relative_to(ROOT))
def order(s):return hashlib.sha256(('natural-first-answer-fresh32-v1\0'+s).encode()).hexdigest()


def main(output):
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU-only cohort registration')
    require(not output.exists(),'preserve existing cohorts')
    require(host_resources()['available_gib']>=48,'cohort preparation RAM reserve')
    code={}
    for name in [OWN,PROTOCOL,'ilm/visual_lm/causal_visual_prefix.py','ilm/visual_lm/complete_writing_data.py']:
        require(subprocess.check_output(['git','show','HEAD:'+name],cwd=ROOT)==(ROOT/name).read_bytes(),'commit cohort protocol first')
        code[name]=sha(ROOT/name)
    import torch
    from ilm.visual_lm.complete_writing_data import CompleteWritingImageData
    from ilm.visual_lm.causal_visual_prefix import fold_writing_prompt
    torch.set_num_threads(2)
    root=ROOT/'data/general_instructions/complete_lines_development_20261105'
    dataset=CompleteWritingImageData(root,purpose='development')
    source=ROOT/'data/visual_qa/cmrc2018_short/development.jsonl'
    rows=list(map(json.loads,source.read_text().splitlines()));by_id={r['id']:r for r in rows}
    bindings={}
    def bind(path):bindings[rel(path)]=sha(path);return rel(path)
    bind(source);bind(ROOT/'data/visual_qa/cmrc2018_short/train.jsonl')
    natural=ROOT/'data/visual_qa/natural_first_images_v1/records.json';bind(natural)
    excluded={r['context_hash'] for r in load(natural)}
    for name in ['glyph_tail_adaptation_v2','balanced_glyph_writer_v1']:
        p=ROOT/'artifacts'/name/'preparation.json';bind(p)
        excluded.update(r['context_hash'] for r in load(p)['fresh_cases'])
    oldpath=ROOT/'artifacts/pixel_language_diversity_v1/manifest.json';bind(oldpath)
    excluded.update(by_id[r['id']]['context_hash'] for r in load(oldpath)['development'] if r['source']=='cmrc')
    groups={}
    for row in rows:
        if row['context_hash'] not in excluded and 'cmrc/'+row['id']+'.pt' in dataset.names:
            groups.setdefault(row['context_hash'],[]).append(row)
    pairs={};no_disjoint=0
    for context,group in groups.items():
        group=sorted(group,key=lambda r:order(r['id']));chosen=None
        for i,a in enumerate(group):
            for b in group[i+1:]:
                if a['question']!=b['question'] and {v['text'] for v in a['references']}.isdisjoint({v['text'] for v in b['references']}):
                    chosen=(a,b);break
            if chosen:break
        if chosen:pairs[context]=chosen
        else:no_disjoint+=1
    selected=sorted(pairs,key=order)[:16];require(len(selected)==16,'sixteen unused paired contexts required')
    output.mkdir(parents=True);(output/'inputs').mkdir();fresh=[];references={}
    for pair,context in enumerate(selected):
        for row in pairs[context]:
            filename='cmrc/'+row['id']+'.pt';full,boundary=dataset.get(filename)
            folded=fold_writing_prompt(full[...,:boundary*64])
            pixels=((folded.pixels[0,0].numpy()+1)*127.5).astype(np.uint8)
            path=output/'inputs'/('fresh_'+row['id']+'.png');Image.fromarray(pixels).save(path)
            index=len(fresh);fresh.append(dict(index=index,id=row['id'],pair=pair,context_hash=context,
                prompt=bind(path),source_tensor_sha256=dataset.report['files_sha256'][filename]))
            references[f'fresh/{index:04}/normal']=[v['text'] for v in row['references']]
    previews=[]
    for pair in range(16):
        left,right=[Image.open(ROOT/r['prompt']).convert('L') for r in fresh[pair*2:pair*2+2]]
        canvas=Image.new('L',(left.width+right.width+24,max(left.height,right.height)+28),245)
        draw=ImageDraw.Draw(canvas);draw.text((4,4),f'third cohort pair {pair:02}; complete question inputs',fill=0)
        canvas.paste(left,(4,24));canvas.paste(right,(left.width+16,24))
        path=output/f'input_preview_{pair:02}.png';canvas.save(path);previews.append(dict(file=path.name,sha256=sha(path)))
    save_json(output/'evaluation_references.json',references)
    primitive=ROOT/'artifacts/semantic_glyph_bridge_v1'
    for n in ['training.pt','inventory.json']:bind(primitive/n)
    save_json(output/'registration.json',dict(code_sha256=code,bindings_sha256=bindings,fresh_cases=fresh,
        primitive=rel(primitive),input_preview_boards=previews,
        evaluation_references_sha256=sha(output/'evaluation_references.json'),
        eligible_disjoint_contexts=len(pairs),remaining_contexts_without_disjoint_pair=no_disjoint,
        prior_reserved_cohorts_and_all_natural_training_contexts_excluded=True,
        fresh_pairs_with_disjoint_references=16,generated_model_outputs_inspected=False,
        unsupported_reference_forms_not_filtered=True,canonical_fonts_qualify_no_historical_script_claim=True,
        initial_checkpoint=rel(ROOT/'artifacts/balanced_glyph_writer_v1/fit/checkpoint_latest.pt'),
        initial_checkpoint_sha256='dcac4ee207614adca78a8b99cc4fe422b3197204d9028192bef30f111d61586d',
        candidate_run='artifacts/natural_first_image_writer_v1',maximum_generation_glyphs=64,
        complete_streams_per_condition=72,pilot_minimum_clear_answers=24,pilot_minimum_paired_contexts=10,
        gate_before_assessment='registered natural first-image qualification must pass and all ten boards must be actually viewed',
        full_reader_copy_and_stopping_qualification_required=True,goal_complete=False))
    print(json.dumps(dict(fresh_questions=32,contexts=16,eligible_contexts=len(pairs),previews=16)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=Path('artifacts/natural_first_answer_assay_v1'))
    main(p.parse_args().output.resolve())
