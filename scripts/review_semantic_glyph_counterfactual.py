#!/usr/bin/env python3
"""Output-first review and numeric reference comparison for date interventions."""
import argparse
import json
import os
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from ilm.image_edit.baseline import ROOT,save_json,sha
from scripts.score_pixel_language_first_blocks import require


def load(path):return json.loads(Path(path).read_text())


def review(output):
    import torch
    from ilm.visual_lm.semantic_glyph_loop import tensor_hash
    torch.set_num_threads(2)
    require(load(output/'execution/status.json')['phase']=='completed','finish every intervention first')
    rows=load(output/'execution/report.json')['records'];require(len(rows)==32,'all32 streams required')
    folder=output/'raw_review';folder.mkdir(exist_ok=False);boards=[]
    for row in rows:
        path=output/'execution'/row['tensor'];require(sha(path)==row['sha256'],'raw tensor changed')
        value=torch.load(path,weights_only=True,map_location='cpu');tiles=value['images']
        require(tiles.shape==(row['glyphs'],1,32,32) and bool(((tiles==-1)|(tiles==1)).all()),'glyph bits differ')
        require(len(value['feedback'])==len(tiles) and len(value['stop_probabilities'])==len(tiles)+1,'decision count differs')
        require(all(p<.5 for p in value['stop_probabilities'][:-1]),'continued after stop')
        require((value['stop_probabilities'][-1]>=.5)==row['stopped'],'stop event differs')
        for i,feedback in enumerate(value['feedback']):require(tensor_hash(tiles[i:i+1])==feedback['image_sha256'],'feedback image differs')
        array=np.full((32,max(1,len(tiles))*32),255,dtype=np.uint8)
        for i,tile in enumerate(tiles):array[:,i*32:(i+1)*32]=((tile[0].numpy()+1)*127.5).astype(np.uint8)
        png=output/'execution'/row['png'];require(sha(png)==row['png_sha256'],'PNG changed')
        require(np.array_equal(array,np.asarray(Image.open(png).convert('L'))),'actual PNG bits differ')
    for parent in (12,15):
        for route in ('normal','blank'):
            group=[r for r in rows if r['parent']==parent and r['route']==route]
            for start in (0,4):
                batch=group[start:start+4];width=max(Image.open(output/'execution'/r['png']).width for r in batch)
                canvas=Image.new('L',(max(640,width+12),4*70),245);draw=ImageDraw.Draw(canvas)
                for j,row in enumerate(batch):
                    draw.text((6,j*70+3),f"parent{parent} variant{start+j} {route} glyphs={row['glyphs']} stop={row['stopped']}",fill=0)
                    canvas.paste(Image.open(output/'execution'/row['png']).convert('L'),(6,j*70+25))
                name=f'parent{parent}_{route}_{start:02}.png';canvas.save(folder/name);boards.append(name)
    save_json(folder/'report.json',dict(streams=32,boards=boards,all_PNG_and_feedback_image_hashes_verified=True,
        expected_years_hidden_from_board_labels=True,output_pixels_repaired=False,
        fully_blinded=False,manual_review_pending=True))
    print(json.dumps(dict(streams=32,boards=boards)))


def score(output):
    import torch
    torch.set_num_threads(2)
    folder=output/'raw_review';manual=load(folder/'manual_review.json')
    require(manual['all32_raw_outputs_viewed'],'actual raw-image review required')
    primitive=ROOT/'artifacts/semantic_glyph_bridge_v1'
    data=torch.load(primitive/'training.pt',weights_only=True,map_location='cpu')
    cmap={c:i for i,row in enumerate(load(primitive/'inventory.json')['records']) for c in row['chars']}
    prepared=load(output/'preparation.json');measurements=[]
    for i,(job,row) in enumerate(zip(prepared['jobs'],load(output/'execution/report.json')['records'])):
        require((job['parent'],job['year'],job['route'])==(row['parent'],row['year'],row['route']),'case order differs')
        value=torch.load(output/'execution'/row['tensor'],weights_only=True,map_location='cpu')
        actual=(value['images'][:,0]<0).numpy().astype(np.uint8)
        gold=np.stack([data['ink'][cmap[c],0].numpy() for c in job['expected_answer']])
        n=max(len(actual),len(gold));a=np.zeros((n,32,32),dtype=np.uint8);g=a.copy();a[:len(actual)]=actual;g[:len(gold)]=gold
        denom=int(a.sum()+g.sum());f1=2*int((a&g).sum())/denom if denom else 1.
        transcription=manual['normal_transcriptions'].get(f"{job['parent']}_{i//2%8}") if job['route']=='normal' else None
        measurements.append(dict(parent=job['parent'],year=job['year'],route=job['route'],intervention=job['intervention'],
            stopped=row['stopped'],capped=row['capped'],foreground_f1=f1,
            canonical_pixel_exact=bool(row['stopped'] and len(actual)==len(gold) and np.array_equal(a,g)),
            output_first_manual_transcription=transcription,
            manual_transcription_matches_target=bool(row['stopped'] and transcription==job['expected_answer']),
            stale_original_year=bool(job['intervention'] and row['stopped'] and transcription==job['original_year']+'年')))
    groups={}
    for route in ('normal','blank'):
        for intervention in (False,True):
            rows=[r for r in measurements if r['route']==route and r['intervention']==intervention]
            groups[route+('_new_dates' if intervention else '_baselines')]=dict(cases=len(rows),
                canonical_pixel_exact=sum(r['canonical_pixel_exact'] for r in rows),
                output_first_manual_transcription_matches=sum(r['manual_transcription_matches_target'] for r in rows),
                stale_original_year=sum(r['stale_original_year'] for r in rows),
                stopped=sum(r['stopped'] for r in rows),capped=sum(r['capped'] for r in rows),
                mean_f1=sum(r['foreground_f1'] for r in rows)/len(rows))
    # Actual cold reruns of the two exactly reproduced original inputs.
    replay={}
    for parent,year in [(12,'1874'),(15,'2001')]:
        a=torch.load(ROOT/f'artifacts/semantic_glyph_language_v1/assessment/development_{parent:04}_normal.pt',weights_only=True,map_location='cpu')
        b=torch.load(output/f'execution/parent{parent}_{year}_normal.pt',weights_only=True,map_location='cpu')
        replay[str(parent)]=dict(images_bit_exact=torch.equal(a['images'],b['images']),
            stops_exact=a['stop_probabilities']==b['stop_probabilities'],feedback_hashes_exact=a['feedback']==b['feedback'])
    save_json(folder/'scores.json',dict(records=measurements,groups=groups,cold_replay=replay,
        independent_human_semantic_grading=False,parent_selection_from_previous_outputs=True,
        prospective_variants_no_further_training=True,runtime_outputs_repaired=False))
    print(json.dumps(groups,ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--stage',choices=['review','score'],required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU-only review required')
    {'review':review,'score':score}[args.stage](args.output.resolve())
