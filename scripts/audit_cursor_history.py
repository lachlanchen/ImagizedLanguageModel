#!/usr/bin/env python3
"""Establish an actual whitespace-state collision and lossless cursor remedy."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from ilm.visual_lm.cursor_canvas import WritingCanvas
from ilm.visual_lm.full_text_renderer import CompleteLineRenderer
from ilm.visual_lm.semantic_image_memory import native_budget_image
from scripts.probe_visual_history_reading import image_tensor,prefix_image,page,LABEL
from scripts.cache_semantic_images_v72 import ROOT,sha,atomic_json,committed_identity

SOURCES=('scripts/audit_cursor_history.py','ilm/visual_lm/cursor_canvas.py',
    'references/cursor_history_protocol.md','ilm/visual_lm/full_text_renderer.py',
    'ilm/visual_lm/semantic_image_memory.py','scripts/probe_visual_history_reading.py',
    'scripts/cache_semantic_images_v72.py')


def fingerprint(a):
    a=np.ascontiguousarray(a)
    return hashlib.sha256(str((a.shape,str(a.dtype))).encode()+a.tobytes()).hexdigest()


def run(out):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or torch.cuda.is_initialized():raise ValueError('CPU-only audit required')
    torch.set_num_threads(4)
    if out.exists():raise FileExistsError('preserve existing cursor evidence')
    producer=committed_identity(SOURCES);out.mkdir(parents=True);started=time.monotonic()
    renderer=CompleteLineRenderer();label,_=renderer.render(LABEL)
    question,_=renderer.render('请抄写“春 风”。')
    a,b=prefix_image(renderer,'春'),prefix_image(renderer,'春 ')
    old_a,old_b=page(question,a,label),page(question,b,label)
    old_observed_a=native_budget_image(image_tensor(old_a));old_observed_b=native_budget_image(image_tensor(old_b))
    if not np.array_equal(old_a,old_b) or not torch.equal(old_observed_a,old_observed_b):
        raise ValueError('the specified actual renderer does not exhibit the claimed collision')
    source,_=renderer.render('春');wind,_=renderer.render('风')
    if not np.all(source[:,32:]==255) or not np.all(wind[:,32:]==255):raise ValueError('complete single glyph does not fit one tile')
    white=np.full((32,32),255,dtype=np.uint8);tiles=source[:,:32]
    first=WritingCanvas.empty().append(tiles);second=first.append(white)
    new_a,new_b=page(question,first.page(),label),page(question,second.page(),label)
    if np.array_equal(new_a,new_b):raise ValueError('cursor does not distinguish a white emission')
    files={}

    def save(name,value):
        Image.fromarray(value).save(out/name);files[name]=sha(out/name)

    save('old_prefix_chun.png',old_a);save('old_prefix_chun_space.png',old_b)
    save('cursor_prefix_chun.png',new_a);save('cursor_prefix_chun_space.png',new_b)
    save('next_space.png',white);save('next_wind.png',wind[:,:32])
    rng=np.random.default_rng(2026110804);lengths=(0,1,31,32,33,255,256,257,512,4096);roundtrips=[]
    for n in lengths:
        payload=rng.integers(0,256,size=(n,32,32),dtype=np.uint8)
        if n:payload[::7]=255
        canvas=WritingCanvas(payload);physical=canvas.page();decoded=WritingCanvas.from_page(physical)
        if len(decoded)!=n or not np.array_equal(decoded.tiles,payload) or not np.array_equal(decoded.page(),physical):
            raise ValueError('payload bytes, order or cursor do not roundtrip')
        if n:
            before=canvas.page().copy();after=canvas.append(white)
            if len(after)!=n+1 or not np.array_equal(canvas.page(),before):raise ValueError('append mutated prior history')
        roundtrips.append(dict(tiles=n,payload_sha256=fingerprint(payload),page_sha256=fingerprint(physical),shape=list(physical.shape)))
    unique={};raw={}
    for n in range(257):
        physical=WritingCanvas(np.full((n,32,32),255,dtype=np.uint8)).page()
        if len(WritingCanvas.from_page(physical))!=n:raise ValueError('white writing length is not recoverable')
        observed=native_budget_image(image_tensor(page(question,physical,label))).numpy()
        key=fingerprint(observed)
        if key in unique:raise ValueError('declared bounded observer states alias: '+str((n,unique[key])))
        unique[key]=n;raw[str(n)]=dict(raw_sha256=fingerprint(physical),observer_sha256=key,observer_shape=list(observed.shape))
    corrupted=first.page().copy();corrupted[40,0]=0
    try:WritingCanvas.from_page(corrupted)
    except ValueError:pass
    else:raise ValueError('invalid gutter was accepted')
    future=first.page().copy();future[0,64]=0
    try:WritingCanvas.from_page(future)
    except ValueError:pass
    else:raise ValueError('hidden future writing was accepted')
    result=dict(producer=producer,renderer=renderer.provenance,files_sha256=files,
        actual_old_prefix_pages_identical=True,actual_old_observer_tensors_identical=True,
        old_page_sha256=fingerprint(old_a),new_pages_distinct=True,
        alias_prefixes=['春','春 '],common_reference='春 风',different_next_images=['next_space.png','next_wind.png'],
        ambiguous_pair_maximum_average_exact_probability=.5,roundtrips=roundtrips,
        all_white_observer_states=raw,all257_bounded_observer_states_distinct=True,
        invalid_gutter_and_future_ink_rejected=True,elapsed_seconds=time.monotonic()-started,
        source_neural_calls=0,optimizer_updates=0,learned_cursor_recognition_proven=False,
        old_model_failures_attributed_to_this_alias=False,new_language_generation_proven=False,general_ilm_goal_complete=False)
    atomic_json(out/'result.json',result)
    print(json.dumps(dict(collision_verified=True,lossless_lengths=list(lengths),distinct_white_states=len(unique),
        elapsed_seconds=result['elapsed_seconds'],language_generation_proven=False),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=ROOT/'artifacts/cursor_history_v1')
    a=p.parse_args();run(a.output.resolve())
