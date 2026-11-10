#!/usr/bin/env python3
"""Review raw glyph streams before evaluator-only canonical reference scoring."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image,ImageDraw
from ilm.image_edit.baseline import ROOT,save_json,sha
from scripts.score_pixel_language_first_blocks import require


def load(path):return json.loads(Path(path).read_text())


def review(output):
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','hide CUDA during raw-image review')
    import torch
    from ilm.visual_lm.semantic_glyph_loop import tensor_hash
    torch.set_num_threads(2)
    require(load(output/'assessment/status.json')['phase']=='completed','complete autonomous assessment first')
    report=load(output/'assessment/report.json')
    folder=output/'raw_review';folder.mkdir(exist_ok=False)
    checked=[];boards=[]
    for row in report['records']:
        path=output/'assessment'/row['file'];require(sha(path)==row['sha256'],'raw answer tensor changed')
        value=torch.load(path,weights_only=True,map_location='cpu');images=value['images']
        require(images.shape==(row['glyphs'],1,32,32) and bool(((images==-1)|(images==1)).all()),'invalid glyph pixels')
        require(len(value['feedback'])==len(images) and len(value['stop_probabilities'])==len(images)+1,
                'feedback or autonomous decision count differs')
        require(len(value['states'])==len(value['stop_probabilities']),'contextual decision state count differs')
        for i,feedback in enumerate(value['feedback']):
            require(tensor_hash(images[i:i+1])==feedback['image_sha256'],'actual feedback image differs')
        probabilities=value['stop_probabilities']
        require(all(p<.5 for p in probabilities[:-1]),'output after a learned stop')
        require((probabilities[-1]>=.5)==row['stopped'],'recorded stop threshold differs')
        require(row['stopped']!=row['capped'],'stop/cap status ambiguous')
        if row['capped']:require(len(images)==load(output/'preparation.json')['maximum_generation_glyphs'],'wrong cap')
        actual=np.full((32,max(1,len(images))*32),255,dtype=np.uint8)
        for i,tile in enumerate(images):actual[:,32*i:32*i+32]=((tile[0].numpy()+1)*127.5).astype(np.uint8)
        png=output/'assessment'/row['png'];require(sha(png)==row['png_sha256'],'raw PNG changed')
        require(np.array_equal(np.asarray(Image.open(png).convert('L')),actual),'PNG differs from actual generated pixels')
        checked.append(dict(**row,feedback_image_hashes_verified=True,PNG_matches_raw_tensor=True))
    for split,route in [('train','normal'),('development','normal'),('development','blank')]:
        group=[r for r in checked if r['split']==split and r['route']==route]
        for start in range(0,len(group),4):
            rows=group[start:start+4]
            width=max(Image.open(output/'assessment'/r['png']).width for r in rows)
            canvas=Image.new('L',(max(640,width+12),len(rows)*70),245);draw=ImageDraw.Draw(canvas)
            for j,row in enumerate(rows):
                draw.text((6,j*70+3),f"{split} {row['index']:04} {route}  glyphs={row['glyphs']} stop={row['stopped']}",fill=0)
                canvas.paste(Image.open(output/'assessment'/row['png']).convert('L'),(6,j*70+25))
            name=f'{split}_{route}_{start:02}.png';canvas.save(folder/name);boards.append(name)
    save_json(folder/'report.json',dict(records=checked,boards=boards,source_references_read=False,
        output_pixels_repaired=False,semantic_review_completed=False,
        feedback_embeddings_independently_reencoded=False))
    print(json.dumps(dict(streams=len(checked),boards=len(boards),all_PNG_and_feedback_image_hashes_verified=True)))


def score(output):
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU evaluator only')
    import torch
    torch.set_num_threads(2)
    folder=output/'raw_review';require((folder/'report.json').exists(),'prepare and view raw-image boards first')
    require((folder/'manual_review.json').exists(),'record actual output-first visual review before reference scoring')
    meta=load(output/'preparation.json');primitive=ROOT/meta['primitive']
    data=torch.load(primitive/'training.pt',weights_only=True,map_location='cpu')
    inventory=load(primitive/'inventory.json')['records']
    charmap={c:i for i,r in enumerate(inventory) for c in r['chars']}
    exact_forms={data['ink'][i,0].numpy().tobytes():r['chars'] for i,r in enumerate(inventory)}
    corpus=load(ROOT/'artifacts/pixel_language_diversity_v1/manifest.json')
    train={r['id']:r for r in map(json.loads,(ROOT/'data/visual_qa/cmrc2018_short/train.jsonl').read_text().splitlines())}
    dev={r['id']:r for r in map(json.loads,(ROOT/'data/visual_qa/cmrc2018_short/development.jsonl').read_text().splitlines())}
    measurements=[]
    for row in load(folder/'report.json')['records']:
        value=torch.load(output/'assessment'/row['file'],weights_only=True,map_location='cpu')
        actual=(value['images']<0).to(torch.uint8)[:,0].numpy()
        if row['split']=='train':
            source=corpus['train'][row['index']];item=train[Path(source['source']).stem]
        else:item=dev[corpus['development'][row['index']]['id']]
        references=[r['text'] for r in item['references']];scores=[];unsupported=[]
        for text in references:
            if any(c not in charmap for c in text):unsupported.append(text);continue
            gold=np.stack([data['ink'][charmap[c],0].numpy() for c in text])
            n=max(len(gold),len(actual));a=np.zeros((n,32,32),dtype=np.uint8);g=a.copy()
            a[:len(actual)]=actual;g[:len(gold)]=gold
            denom=int(a.sum()+g.sum());f1=2*int((a&g).sum())/denom if denom else 1.
            scores.append(dict(reference=text,f1=f1,pixel_exact=bool(value['stopped'] and np.array_equal(a,g)
                                                                     and len(actual)==len(gold))))
        labels=[exact_forms.get(tile.tobytes()) for tile in actual]
        transcription=''.join(v[0] if v and len(v)==1 else ('{'+','.join(v)+'}' if v else '<?>') for v in labels)
        measurements.append(dict(split=row['split'],index=row['index'],route=row['route'],references=references,
            supported_reference_scores=scores,unsupported_reference_strings=unsupported,
            exact_canonical_reference=any(s['pixel_exact'] for s in scores),
            best_f1=max((s['f1'] for s in scores),default=None),
            exact_bitmap_only_transcription=transcription,exact_glyph_forms=sum(v is not None for v in labels),
            generated_glyphs=len(actual),stopped=value['stopped'],capped=value['capped']))
    groups={}
    for split,route in [('train','normal'),('development','normal'),('development','blank')]:
        rows=[r for r in measurements if r['split']==split and r['route']==route]
        scores=[r['best_f1'] for r in rows if r['best_f1'] is not None]
        groups[split+'_'+route]=dict(cases=len(rows),canonical_reference_exact=sum(r['exact_canonical_reference'] for r in rows),
            cases_with_supported_reference=len(scores),mean_f1=sum(scores)/len(scores) if scores else None,
            stopped=sum(r['stopped'] for r in rows),capped=sum(r['capped'] for r in rows))
    save_json(folder/'canonical_scores.json',dict(records=measurements,groups=groups,
        evaluator_only_glyph_inventory=True,runtime_answers_replaced=False,
        approximate_nearest_glyph_decoding_used=False,formal_semantic_accuracy_measured=False,
        reused_development_not_sealed=True))
    print(json.dumps(groups,ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['review','score'],required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();{'review':review,'score':score}[args.stage](args.output.resolve())
