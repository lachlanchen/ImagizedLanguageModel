#!/usr/bin/env python3
"""Resumable natural image-prefill cache with an unchanged visual reader."""
import argparse
import json
import os
from pathlib import Path
import subprocess

from ilm.image_edit.baseline import ROOT,sha,save_json
from scripts.score_pixel_language_first_blocks import require

DATA=ROOT/'data/visual_qa/natural_first_images_v1'
CHECKPOINT=ROOT/'artifacts/balanced_glyph_writer_v1/fit/checkpoint_latest.pt'
PARENT_SHA='dcac4ee207614adca78a8b99cc4fe422b3197204d9028192bef30f111d61586d'
FILES=['scripts/cache_natural_first_states.py','references/natural_first_image_protocol.md',
       'scripts/infer_glyph_tail_language.py','ilm/visual_lm/semantic_glyph_loop.py',
       'ilm/visual_lm/glyph_tail_adapter.py','ilm/visual_lm/semantic_image_memory.py',
       'ilm/visual_lm/glyph_runtime_resources.py','scripts/train_semantic_glyph_language.py',
       'scripts/distill_semantic_glyph_readout.py']


def load(p):return json.loads(Path(p).read_text())
def rel(p):return str(Path(p).resolve().relative_to(ROOT))


def identity():
    values={}
    for name in FILES:
        require(subprocess.check_output(['git','show','HEAD:'+name],cwd=ROOT)==(ROOT/name).read_bytes(),'commit natural cache code first')
        values[name]=sha(ROOT/name)
    return values


def prepare(run):
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU-only run preparation')
    require(not run.exists(),'preserve previous runs')
    report=load(DATA/'report.json');require(sha(CHECKPOINT)==PARENT_SHA,'fixed reader identity differs')
    require(report['raw_training_partition_only'] and report['all_prompt_pixels_preserved'],'complete natural prompts required')
    for n,d in report['source_sha256'].items():require(sha(ROOT/n)==d,'natural raw source changed')
    for n,d in report['code_sha256'].items():require(sha(ROOT/n)==d,'natural renderer changed')
    require(sha(DATA/'records.json')==report['records_sha256'],'natural record list changed')
    review=load(DATA/'input_review.json');require(review['report_sha256']==sha(DATA/'report.json')
        and review['all_eight_previews_actually_viewed'],'actually inspect fixed input previews first')
    run.mkdir(parents=True)
    save_json(run/'preparation.json',dict(code_sha256=identity(),data=rel(DATA),
        data_report_sha256=sha(DATA/'report.json'),data_records_sha256=sha(DATA/'records.json'),
        input_review_sha256=sha(DATA/'input_review.json'),checkpoint=rel(CHECKPOINT),checkpoint_sha256=PARENT_SHA,
        frozen_reader=True,source_projection_loaded=False,output_history_used=False,
        maximum_stage_seconds=10800.,maximum_pressure_wait_seconds=3600.,
        minimum_entry_ram_gib=48.,minimum_runtime_ram_gib=24.,maximum_vram_gib=22.,
        shard_records=64,model_updates=0,usable_language_demonstrated=False))


def verify(run):
    m=load(run/'preparation.json');require(identity()==m['code_sha256'],'prepared cache code changed')
    for n,key in [('report.json','data_report_sha256'),('records.json','data_records_sha256'),('input_review.json','input_review_sha256')]:
        require(sha(ROOT/m['data']/n)==m[key],'natural preparation changed')
    require(sha(ROOT/m['checkpoint'])==m['checkpoint_sha256'],'reader weights changed')
    return m


def cache(run):
    import torch
    from scripts.infer_glyph_tail_language import load_actor
    from scripts.train_semantic_glyph_language import numerics,image,atomic_tensor
    from ilm.visual_lm.semantic_image_memory import native_budget_image
    from ilm.visual_lm.glyph_runtime_resources import full_reader_gpu
    from scripts.distill_semantic_glyph_readout import Monitor
    m=verify(run);data=ROOT/m['data'];records=load(data/'records.json')
    folder=run/'cache';folder.mkdir(exist_ok=True)
    require(not (folder/'report.json').exists(),'completed cache is immutable')
    attempts=folder/'attempts';attempts.mkdir(exist_ok=True)
    attempt=attempts/f'{len(list(attempts.iterdir())):03}';attempt.mkdir(exist_ok=False)
    monitor=Monitor(attempt,m);monitor.status('admitting')
    index=load(folder/'index.json') if (folder/'index.json').exists() else []
    done=0
    for row in index:
        require(row['first_index']==done and sha(folder/row['file'])==row['sha256'],'previous frozen cache shard differs')
        done+=row['records']
    require(done<=len(records),'cache index exceeds inputs')
    try:
        monitor.guard()
        with full_reader_gpu() as (device,admission),torch.no_grad():
            numerics();save_json(attempt/'admission.json',admission)
            actor,provenance=load_actor(ROOT/m['checkpoint'],device);monitor.guard()
            for start in range(done,len(records),m['shard_records']):
                group=records[start:start+m['shard_records']];states=[];geometries=[]
                for r in group:
                    monitor.position=dict(completed=r['index'],total=len(records));monitor.guard()
                    path=data/r['file'];require(sha(path)==r['sha256'],'natural input image changed')
                    arguments,geometry=actor.interface.assemble(native_budget_image(image(path)))
                    h=actor.interface.actor.context(**arguments,use_cache=False).last_hidden_state[:,-1].float().cpu()
                    require(h.shape==(1,4096) and torch.isfinite(h).all(),'invalid actual first state')
                    states.append(h[0]);geometries.append(geometry)
                    del arguments,h
                path=folder/f'shard_{start:06}.pt'
                require(not path.exists(),'unindexed partial shard requires explicit preservation before resume')
                atomic_tensor(path,dict(states=torch.stack(states),record_indices=torch.tensor([r['index'] for r in group]),
                    labels=torch.tensor([r['label'] for r in group])))
                index.append(dict(file=path.name,sha256=sha(path),first_index=start,records=len(group),
                    inputs_sha256=[r['sha256'] for r in group]))
                save_json(folder/'index.json',index)
                monitor.status('caching_real_image_first_states',completed=start+len(group),total=len(records))
                print(json.dumps(dict(completed=start+len(group),total=len(records),peak_GiB=torch.cuda.max_memory_allocated()/1024**3)),flush=True)
            require(sum(r['records'] for r in index)==len(records),'incomplete natural cache')
            save_json(attempt/'report.json',dict(records_newly_processed=len(records)-done,
                resumed_frozen_records=done,checkpoint_sha256=m['checkpoint_sha256'],provenance=provenance,
                source_lexical_head_loaded=False,tokenizer_called=False,font_renderer_called=False,
                actual_complete_prompt_pixels=True,output_pixels_or_answer_supplied=False,optimizer_updates=0,
                **monitor.receipt()))
            save_json(folder/'report.json',dict(files=index,index_sha256=sha(folder/'index.json'),
                records=len(records),preparation_sha256=sha(run/'preparation.json'),
                actual_first_states_from_unchanged_adapted_body=True,source_lexical_head_loaded=False,
                training_references_not_passed_to_reader=True,successful_attempt=str(attempt.relative_to(folder)),
                attempt_report_sha256=sha(attempt/'report.json')))
        monitor.status('completed')
    except BaseException as error:monitor.status('failed',reason=repr(error));raise
    finally:save_json(attempt/'status.json',dict(load(attempt/'status.json'),pid=None))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',choices=['prepare','cache'],required=True)
    p.add_argument('--run',type=Path,default=Path('artifacts/natural_first_image_writer_v1'));a=p.parse_args()
    globals()[a.stage](a.run.resolve())
