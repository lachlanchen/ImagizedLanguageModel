#!/usr/bin/env python3
"""Frozen-model date interventions, prepared before any new output is observed."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image
from ilm.image_edit.baseline import ROOT,save_json,sha
from scripts.score_pixel_language_first_blocks import diagnostic_gpu,require
from scripts.train_semantic_glyph_language import (Monitor,bridge_model,reader,numerics,
                                                  atomic_tensor,image,check)

TRAINED=ROOT/'artifacts/semantic_glyph_language_v1'
YEARS=('1874','1936','1648','1789','2001','2007','2019','1993')
FILES=('scripts/check_semantic_glyph_counterfactual.py',
       'references/semantic_glyph_counterfactual_protocol.md',
       'ilm/visual_lm/full_text_renderer.py','ilm/visual_lm/causal_visual_prefix.py')


def load(path):return json.loads(Path(path).read_text())
def relative(path):return str(Path(path).resolve().relative_to(ROOT))


def identity():
    result={name:sha(ROOT/name) for name in FILES}
    for name,digest in result.items():
        require(hashlib.sha256(subprocess.check_output(['git','show','HEAD:'+name],cwd=ROOT)).hexdigest()==digest,
                'commit counterfactual implementation first: '+name)
    return result


def prepare(output):
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU-only counterfactual preparation required')
    require(not output.exists(),'preserve prior interventions')
    source_identity=identity();meta=check(TRAINED)
    require(load(TRAINED/'assessment/status.json')['phase']=='completed','finish original assessment first')
    import torch
    from ilm.visual_lm.full_text_renderer import CompleteLineRenderer
    from ilm.visual_lm.causal_visual_prefix import fold_writing_prompt
    torch.set_num_threads(1)
    renderer=CompleteLineRenderer()
    corpus=ROOT/'artifacts/pixel_language_diversity_v1';manifest=load(corpus/'manifest.json')
    source=ROOT/'data/visual_qa/cmrc2018_short/development.jsonl'
    records={r['id']:r for r in map(json.loads,source.read_text().splitlines())}
    output.mkdir(parents=True);jobs=[]
    bindings={relative(TRAINED/'preparation.json'):sha(TRAINED/'preparation.json'),
        relative(TRAINED/'fit/checkpoint_latest.pt'):sha(TRAINED/'fit/checkpoint_latest.pt'),
        relative(source):sha(source),relative(corpus/'manifest.json'):sha(corpus/'manifest.json')}
    for index,old in [(12,'1874'),(15,'2001')]:
        row=manifest['development'][index];item=records[row['id']]
        require(old+'年' in item['context'] and old not in item['question'],'unambiguous source date required')
        normal=np.asarray(Image.open(corpus/row['normal']).convert('L')).copy()
        blank=np.asarray(Image.open(corpus/row['blank']).convert('L')).copy()
        for route in ('normal','blank'):bindings[relative(corpus/row[route])]=row[route+'_sha256']
        erased=normal!=blank
        require(np.all(blank[erased]==255),'source question control is not a pure erasure')
        baseline,_=renderer.render(item['prompt'],240)
        folded=fold_writing_prompt(torch.from_numpy(baseline.copy()).float()[None,None]/127.5-1)
        restored=((folded.pixels[0,0]+1)*127.5).to(torch.uint8).numpy()
        require(np.array_equal(restored,normal),'new host rendering does not reproduce original input exactly')
        for year in YEARS:
            prompt=item['prompt'].replace(old,year)
            strip,layout=renderer.render(prompt,240)
            require(strip.shape==baseline.shape,'date replacement changes physical strip width')
            # Keep original wrapping, and verify that no boundary splits ink.
            page=np.full_like(normal,255)
            for j,(start,end) in enumerate(folded.source_ranges):
                if end<strip.shape[-1]:require(np.all(strip[:,end-1]==255),'counterfactual ink crosses an original wrap')
                page[j*32:(j+1)*32,:end-start]=strip[:,start:end]
            require(np.array_equal(page[erased],normal[erased]),'date intervention changes question ink')
            if year==old:require(np.array_equal(page,normal),'baseline pixels changed')
            erased_page=page.copy();erased_page[erased]=255
            for route,array in [('normal',page),('blank',erased_page)]:
                name=f'parent{index:02}_{year}_{route}.png';Image.fromarray(array).save(output/name)
                jobs.append(dict(parent=index,year=year,route=route,image=name,sha256=sha(output/name),
                    original_year=old,intervention=year!=old,context_replacements=item['context'].count(old),
                    expected_answer=year+'年',source_id=row['id']))
    save_json(output/'preparation.json',dict(code_sha256=source_identity,bindings_sha256=bindings,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        trained_run=relative(TRAINED),jobs=jobs,years=list(YEARS),training_updates=0,
        parents_selected_from_previous_outputs=True,prospective_intervention_outputs=True,
        reused_parent_development_not_sealed=True,changed_dates_are_counterfactual_not_historical_claims=True,
        renderer=renderer.provenance,original_baseline_PNGs_exact=True,maximum_glyphs=64,
        source_output_head_allowed=False,maximum_seconds=1200.))
    print(json.dumps(dict(prepared=True,cases=len(jobs),new_date_normal_cases=14,training_updates=0)))


def run(output):
    import torch
    from ilm.visual_lm.semantic_glyph_loop import SemanticGlyphLanguageActor
    prepared=load(output/'preparation.json');require(identity()==prepared['code_sha256'],'intervention code changed')
    for name,digest in prepared['bindings_sha256'].items():require(sha(ROOT/name)==digest,'bound counterfactual input changed')
    meta=check(TRAINED);folder=output/'execution';folder.mkdir(exist_ok=False)
    monitor=Monitor(folder,prepared['maximum_seconds']);monitor.status('admitting')
    try:
        with diagnostic_gpu() as (device,admission),torch.no_grad():
            monitor.admission=admission;numerics();monitor.guard()
            bridge=bridge_model(meta,device,TRAINED/'fit/checkpoint_latest.pt')
            interface,provenance=reader(device)
            actor=SemanticGlyphLanguageActor(interface,bridge,language_training_complete=True).eval()
            hook=interface.actor.context.register_forward_pre_hook(lambda module,args:monitor.guard())
            records=[]
            for i,job in enumerate(prepared['jobs']):
                monitor.position=dict(completed=i,total=len(prepared['jobs']),parent=job['parent'],year=job['year'],route=job['route'])
                path=output/job['image'];require(sha(path)==job['sha256'],'prepared prompt changed')
                # Target year/text never enters the actor or stopping rule.
                value=actor.generate(image(path),maximum_glyphs=prepared['maximum_glyphs'])
                stem=Path(job['image']).stem;atomic_tensor(folder/(stem+'.pt'),value)
                tiles=value['images'];array=np.full((32,max(1,len(tiles))*32),255,dtype=np.uint8)
                for j,tile in enumerate(tiles):array[:,32*j:32*j+32]=((tile[0].numpy()+1)*127.5).astype(np.uint8)
                Image.fromarray(array).save(folder/(stem+'.png'))
                records.append(dict(parent=job['parent'],year=job['year'],route=job['route'],
                    intervention=job['intervention'],png=stem+'.png',png_sha256=sha(folder/(stem+'.png')),
                    tensor=stem+'.pt',sha256=sha(folder/(stem+'.pt')),glyphs=len(tiles),
                    stopped=value['stopped'],capped=value['capped']))
                monitor.status('generating_counterfactual_answers');print(json.dumps(records[-1]),flush=True)
            hook.remove()
            save_json(folder/'report.json',dict(records=records,training_updates=0,
                no_target_length_or_text_input_to_actor=True,teacher=False,runtime_glyph_table=False,
                provenance=provenance,**monitor.receipt()))
            monitor.status('completed')
    except BaseException as error:
        monitor.status('failed',reason=repr(error));raise
    finally:
        if (folder/'status.json').exists():save_json(folder/'status.json',dict(load(folder/'status.json'),pid=None))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['prepare','run'],required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();{'prepare':prepare,'run':run}[args.stage](args.output.resolve())
