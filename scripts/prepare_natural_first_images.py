#!/usr/bin/env python3
"""Prepare complete training-only CMRC prompts for natural first-state learning."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image

from ilm.image_edit.baseline import ROOT,sha,save_json
from scripts.score_pixel_language_first_blocks import require,host_resources

FILES=['scripts/prepare_natural_first_images.py','references/natural_first_image_protocol.md',
       'ilm/visual_lm/full_text_renderer.py','ilm/visual_lm/causal_visual_prefix.py']


def load(p):return json.loads(Path(p).read_text())
def visible(text):return ''.join(c for c in text if not c.isspace())
def digest(text):return hashlib.sha256(text.encode()).hexdigest()


def main(output):
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU-only preparation')
    require(not output.exists(),'preserve previous preparations')
    code={}
    for n in FILES:
        require(subprocess.check_output(['git','show','HEAD:'+n],cwd=ROOT)==(ROOT/n).read_bytes(),'commit natural data preparation first')
        code[n]=sha(ROOT/n)
    require(host_resources()['available_gib']>=48,'CPU preparation RAM reserve')
    import torch
    from ilm.visual_lm.full_text_renderer import CompleteLineRenderer
    from ilm.visual_lm.causal_visual_prefix import fold_writing_prompt
    torch.set_num_threads(2);renderer=CompleteLineRenderer()
    train=ROOT/'data/raw/cmrc2018/cmrc2018_train.json';dev=ROOT/'data/raw/cmrc2018/cmrc2018_dev.json'
    require(sha(train)=='5497aa2f81908e31d6b0e27d99b1f90ab63a8f58fa92fffe5d17cf62eba0c212','pinned training source changed')
    require(sha(dev)=='e9ff74231f05c230c6fa88b84441ee334d97234cbb610991cd94b82db00c7f1f','pinned development source changed')
    oldfile=ROOT/'data/visual_qa/cmrc2018_short/train.jsonl'
    old={r['id']:r for r in map(json.loads,oldfile.read_text().splitlines())}
    oldcontexts={r['context_hash'] for r in old.values()}
    excluded_titles=set();excluded_contexts=set()
    for a in load(dev)['data']:
        excluded_titles.add(a['title'])
        excluded_contexts.update(digest(visible(p['context'])) for p in a['paragraphs'])
    inventory=ROOT/'artifacts/semantic_glyph_bridge_v1/inventory.json'
    charmap={c:i for i,row in enumerate(load(inventory)['records']) for c in row['chars']}
    output.mkdir(parents=True);(output/'inputs').mkdir();rows=[];failures=[];counts=Counter();seen={}
    for article in load(train)['data']:
        for paragraph in article['paragraphs']:
            original=paragraph['context'];context=visible(original);ch=digest(context)
            for q in paragraph['qas']:
                counts['source_questions']+=1
                if article['title'] in excluded_titles or ch in excluded_contexts:
                    counts['development_overlap_removed']+=1;continue
                refs=[]
                for answer in q['answers']:
                    text=answer['text'];start=answer['answer_start'];v=visible(text)
                    if original[start:start+len(text)]==text and v and v not in refs:refs.append(v)
                if not refs:
                    counts['no_valid_reference']+=1;continue
                # Retain the exact target already used by the inherited fit
                # when the question has several legitimate source answers.
                if q['id'] in old:refs=[r['text'] for r in old[q['id']]['references']]+refs
                firsts=list(dict.fromkeys(r[0] for r in refs))
                label=charmap.get(firsts[0],-1)
                held=ch not in oldcontexts and int(digest('natural-first-held-v1\0'+ch),16)%5==0
                question=visible(q['question'])
                prompts=[f'资料：{context}问题：{question}回答：',
                         f'短文：{context}问题：{question}答案：',
                         f'请根据资料回答问题。资料：{context}问题：{question}只写答案：']
                for style,prompt in enumerate(prompts):
                    key=q['id']+'_'+str(style)
                    try:
                        strip,meta=renderer.render(prompt,maximum_blocks=1024)
                        pixels=fold_writing_prompt(torch.from_numpy(strip.copy()).float()[None,None]/127.5-1).pixels
                    except (ValueError,OverflowError) as error:
                        failures.append(dict(id=key,reason=str(error)));counts['render_rejections']+=1;continue
                    page=((pixels[0,0].numpy()+1)*127.5).astype(np.uint8)
                    pixelhash=hashlib.sha256(page.tobytes()).hexdigest()
                    if pixelhash in seen:
                        counts['duplicate_physical_input']+=1
                        require(seen[pixelhash]==label,'identical physical prompt has conflicting first target')
                        continue
                    seen[pixelhash]=label
                    file='inputs/'+key+'.png';Image.fromarray(page).save(output/file)
                    rows.append(dict(index=len(rows),id=q['id'],style=style,context_hash=ch,
                        split='held_context' if held else 'train',file=file,sha256=sha(output/file),
                        label=label,reference_firsts=firsts,reference_labels=[charmap.get(c,-1) for c in firsts],
                        inherited_question=q['id'] in old,complete_prompt=True,prompt_pixels_sha256=pixelhash))
                    counts['held' if held else 'train']+=1
    require(counts['source_questions']==10142,'complete raw training partition required')
    save_json(output/'records.json',rows);save_json(output/'rejections.json',failures)
    previews=[]
    for split in ['train','held_context']:
        group=[r for r in rows if r['split']==split]
        for j,index in enumerate(np.linspace(0,len(group)-1,4,dtype=int)):
            row=group[index];name=f'preview_{split}_{j}.png';Image.open(output/row['file']).save(output/name)
            previews.append(dict(file=name,sha256=sha(output/name),record_index=row['index']))
    save_json(output/'report.json',dict(code_sha256=code,
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in [train,dev,oldfile,inventory]},
        counts=dict(counts),renderer=renderer.provenance,records_sha256=sha(output/'records.json'),
        rejections_sha256=sha(output/'rejections.json'),preview_images=previews,
        raw_training_partition_only=True,all_raw_development_contexts_and_titles_excluded=True,
        held_contexts_never_inherited_training_questions=True,three_fixed_prompt_styles=True,
        all_prompt_pixels_preserved=True,answer_pixels_not_in_prompt=True,
        supported_training_first_forms=len({r['label'] for r in rows if r['split']=='train' and r['label']>=0}),
        unsupported_train_firsts=sum(r['label']<0 for r in rows if r['split']=='train'),
        held_unsupported_reference_cases=sum(not any(i>=0 for i in r['reference_labels']) for r in rows if r['split']=='held_context'),
        teacher_or_model_calls=0,language_quality_measured=False))
    print(json.dumps(dict(counts=counts,prepared_records=len(rows))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=Path('data/visual_qa/natural_first_images_v1'))
    main(p.parse_args().output.resolve())
