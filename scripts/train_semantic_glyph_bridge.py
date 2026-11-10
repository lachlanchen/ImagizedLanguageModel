#!/usr/bin/env python3
"""Pretrain a continuous image/semantic glyph interface, not a language model.

Source IDs and fonts are used only by CPU preparation. The trained checkpoint
contains neural weights and global affine statistics, never a glyph inventory.
"""
import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import unicodedata

from ilm.image_edit.baseline import ROOT, save_json, sha
from scripts.score_pixel_language_first_blocks import host_resources, owner_gone, require

SOURCE = ROOT/'data/visual_qa/cmrc2018_short/train.jsonl'
FONT = Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
FILES = ('ilm/visual_lm/semantic_glyph_loop.py', 'scripts/train_semantic_glyph_bridge.py',
         'references/semantic_glyph_loop_protocol.md', 'ilm/image_edit/baseline.py',
         'scripts/score_pixel_language_first_blocks.py', 'scripts/check_native_visual_transfer_v71.py',
         'ilm/visual_lm/gguf_visual_language.py', 'ilm/visual_lm/gguf_language_trunk.py')


def identity():
    result = {name: sha(ROOT/name) for name in FILES}
    for name, digest in result.items():
        require(hashlib.sha256(subprocess.check_output(['git','show','HEAD:'+name],cwd=ROOT)).hexdigest()
                == digest, 'commit glyph implementation first: '+name)
    return result


def atomic_tensor(path, data):
    import torch
    path=Path(path)
    temporary=path.with_name(path.name+'.partial')
    torch.save(data,temporary)
    temporary.replace(path)


def prepare(output, limit):
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='', 'hide CUDA during preparation')
    require(not output.exists(), 'preserve existing glyph preparations')
    code=identity()
    import numpy as np
    import torch
    from PIL import Image, ImageDraw, ImageFont
    from fontTools.ttLib import TTFont
    from scripts.check_native_visual_transfer_v71 import archive_and_tokenizer, embed_rows
    torch.set_num_threads(4)
    frequency=Counter()
    for line in SOURCE.read_text().splitlines():
        row=json.loads(line)
        frequency.update(row['context']+' '+row['question']+' '+row['references'][0]['text'])
    # Font collection index2 is the locally verified Noto Sans CJK SC face.
    face=TTFont(FONT,fontNumber=2)
    family=str(face['name'].getDebugName(1))
    require('SC' in family, 'the configured font collection face is not Simplified Chinese')
    cmap=face.getBestCmap()
    font=ImageFont.truetype(str(FONT),26,index=2)
    candidates=list(dict.fromkeys([chr(c) for c in range(32,127)]+[
        c for c,_ in frequency.most_common() if unicodedata.category(c)[0] in 'LNPSZ']))
    archive,tokenizer,_=archive_and_tokenizer()
    chosen=[]; excluded=[]
    for char in candidates:
        if len(chosen)>=limit: break
        ids=tokenizer.encode(char,add_special_tokens=False).ids
        if ord(char) not in cmap or len(ids)!=1 or tokenizer.decode(ids)!=char:
            excluded.append(dict(char=char,reason='font coverage or single-source-unit anchor unavailable'))
            continue
        canvas=Image.new('L',(32,32),255)
        draw=ImageDraw.Draw(canvas)
        width=draw.textlength(char,font=font)
        draw.text(((32-width)/2,26),char,font=font,fill=0,anchor='ls')
        array=(np.asarray(canvas)<128).astype(np.uint8)
        if not array.any() and char!=' ':
            excluded.append(dict(char=char,reason='empty rendered form'))
            continue
        chosen.append(dict(char=char,id=ids[0],ink=array))
    require(len(chosen)>=1000, 'insufficient training-only glyph anchors')
    vectors=embed_rows(archive,[v['id'] for v in chosen])
    # Visually identical forms cannot identify different source embeddings.
    # Merge their training targets rather than imposing contradictory E losses.
    groups=defaultdict(list)
    for i,row in enumerate(chosen): groups[row['ink'].tobytes()].append(i)
    images=[]; semantics=[]; records=[]
    for indexes in groups.values():
        images.append(chosen[indexes[0]]['ink'])
        semantics.append(vectors[indexes].mean(0))
        records.append(dict(chars=[chosen[i]['char'] for i in indexes],
                            source_ids=[chosen[i]['id'] for i in indexes]))
    images=torch.from_numpy(np.stack(images)[:,None].copy())
    semantic=torch.stack(semantics)
    mean=semantic.mean(0); scale=semantic.std(0).clamp_min(1e-4)
    output.mkdir(parents=True)
    atomic_tensor(output/'training.pt',dict(ink=images,semantic=semantic,mean=mean,scale=scale))
    save_json(output/'inventory.json',dict(records=records,excluded=excluded))
    save_json(output/'preparation.json',dict(stage='glyph_primitives_only',
        code_sha256=code,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        source_sha256=sha(SOURCE),font_sha256=sha(FONT),font_face=family,font_collection_index=2,
        training_sha256=sha(output/'training.pt'),inventory_sha256=sha(output/'inventory.json'),
        requested_limit=limit,anchored_characters=len(chosen),unique_images=len(images),
        merged_alias_groups=sum(len(v)>1 for v in groups.values()),
        selected_from_training_only=True,development_used=False,semantic_width=4096,
        language_training_complete=False,seed=202611101,
        maximum_steps=6000,batch_size=128,learning_rate=.0003,maximum_seconds=1800.,
        maximum_allocated_vram_gib=5.,minimum_entry_ram_gib=64.,minimum_runtime_ram_gib=32.,
        max_some_pressure_avg10=.5,max_full_pressure_avg10=.1,total_pressure_wait_seconds=300.,
        mixed_precision='BF16 convolution/linear autocast; FP32 parameters, targets and losses',
        heldout_augmentation_is_not_unseen_character_generalization=True))
    print(json.dumps(dict(prepared=True,characters=len(chosen),unique_images=len(images),
                         excluded=len(excluded),language_model_trained=False)))


def project_processes():
    result=[]
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit(): continue
        try:
            cwd=(proc/'cwd').resolve()
            command=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
        except (OSError,RuntimeError): continue
        if cwd==ROOT or ROOT in cwd.parents:
            result.append(dict(pid=int(proc.name),command=command[:300]))
    return result


@contextmanager
def bridge_gpu(prepared):
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='0', 'physical GPU0 only')
    owner_gone(ROOT/'artifacts/pixel_language_diversity_training_v2')
    with (ROOT/'artifacts/ilm_gpu0.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        entry=host_resources()
        require(entry['available_gib']>=prepared['minimum_entry_ram_gib'], 'bridge RAM entry reserve unavailable')
        require(entry['some_pressure_avg10']<=.5 and entry['full_pressure_avg10']<=.1,
                'bridge memory pressure entry limit')
        inventory=project_processes()
        obsolete=[p for p in inventory if any(s in p['command'] for s in
                  ('Xvfb ', 'x11vnc ', 'websockify ', 'train_pixel_language_diversity.py --stage run'))]
        require(not obsolete, 'inspect project runtime before bridge admission')
        free=int(subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=memory.free',
                                         '--format=csv,noheader,nounits'],text=True).strip())
        require(free>=20*1024,'another GPU0 owner may be active')
        import torch
        torch.set_num_threads(4)
        torch.cuda.set_device(0)
        torch.cuda.reset_peak_memory_stats()
        yield torch.device('cuda:0'),dict(entry=entry,project_processes=inventory,
            mode='separate_small_bridge_training_RAM_and_pressure_protocol',
            original_training_admission_unchanged=True,foreign_processes_changed=False,
            high_global_swap_occupancy_recorded=entry['swap_fraction']>.75)


def train(output):
    import torch
    from torch.nn import functional as F
    from ilm.visual_lm.semantic_glyph_loop import SemanticGlyphBridge,reconstruction_loss
    prepared=json.loads((output/'preparation.json').read_text())
    require(identity()==prepared['code_sha256'],'prepared source changed')
    require(sha(output/'training.pt')==prepared['training_sha256'],'training tensors changed')
    execution=output/'execution';execution.mkdir(exist_ok=False)
    began=time.perf_counter();waited=0.;samples=[];step=0
    def status(phase,**extra):
        save_json(execution/'status.json',{'phase':phase,'pid':os.getpid(),
            'start_ticks':Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],
            'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            'step':step,'elapsed_seconds':time.perf_counter()-began,**extra})
    status('admitting')
    try:
        with bridge_gpu(prepared) as (device,admission):
            save_json(execution/'admission.json',admission)
            def guard():
                nonlocal waited
                require(torch.cuda.max_memory_allocated()/1024**3<=prepared['maximum_allocated_vram_gib'],
                        'bridge allocation cap reached')
                require(time.perf_counter()-began<=prepared['maximum_seconds'],'bridge time budget reached')
                while True:
                    value=host_resources();samples.append(value)
                    require(value['available_gib']>=prepared['minimum_runtime_ram_gib'],'bridge RAM reserve reached')
                    if value['some_pressure_avg10']<=.5 and value['full_pressure_avg10']<=.1: break
                    remaining=prepared['total_pressure_wait_seconds']-waited
                    require(remaining>0,'bridge pressure waiting exhausted')
                    status('waiting_for_memory_pressure',resource=value)
                    start=time.perf_counter();time.sleep(min(10.,remaining));waited+=time.perf_counter()-start
            torch.manual_seed(prepared['seed']);torch.cuda.manual_seed_all(prepared['seed'])
            torch.backends.cuda.matmul.allow_tf32=True
            torch.backends.cudnn.allow_tf32=True
            torch.backends.cudnn.benchmark=False
            data=torch.load(output/'training.pt',map_location='cpu',weights_only=True)
            images=(1-2*data['ink'].float()).to(device)
            targets=((data['semantic']-data['mean'])/data['scale']).to(device)
            bridge=SemanticGlyphBridge().to(device)
            bridge.semantic_mean.copy_(data['mean']);bridge.semantic_scale.copy_(data['scale'])
            bridge.readout.requires_grad_(False);bridge.stop.requires_grad_(False)
            params=[p for p in bridge.parameters() if p.requires_grad]
            optimizer=torch.optim.AdamW(params,lr=prepared['learning_rate'],weight_decay=.01)
            status('training',parameters=sum(p.numel() for p in params))
            def checkpoint():
                state=dict(model={k:v.detach().cpu() for k,v in bridge.state_dict().items()},
                    optimizer=optimizer.state_dict(),step=step,
                    preparation_sha256=sha(output/'preparation.json'),language_training_complete=False,
                    torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state())
                atomic_tensor(execution/'checkpoint_latest.pt',state)
            def evaluate():
                metrics=[]
                bridge.eval()
                with torch.no_grad():
                    for first in range(0,len(images),64):
                        guard()
                        image=images[first:first+64];target=targets[first:first+64]
                        encoded=bridge.encode(image)
                        for route,code in [('semantic',target),('image_roundtrip',encoded)]:
                            logits=bridge.decode_logits(code)
                            actual=logits>=0;gold=image<0
                            tp=(actual&gold).sum((1,2,3));denom=actual.sum((1,2,3))+gold.sum((1,2,3))
                            f1=torch.where(denom>0,2*tp/denom,torch.ones_like(denom,dtype=torch.float))
                            for j in range(len(image)):
                                metrics.append(dict(index=first+j,route=route,f1=float(f1[j]),
                                    pixel_exact=bool((actual[j]==gold[j]).all())))
                bridge.train()
                return dict(step=step,records=metrics,
                    semantic_cosine=float(F.cosine_similarity(bridge.encode(images[:128]),targets[:128]).mean()),
                    reconstruction_is_not_language_validation=True)
            save_json(execution/'assessment_00000.json',evaluate())
            while step<prepared['maximum_steps']:
                if step%20==0: guard()
                indexes=torch.randint(len(images),(prepared['batch_size'],),device=device)
                image=images[indexes];target=targets[indexes]
                # Whole-batch integer shifts, white padding: no wraparound strokes.
                dx=int(torch.randint(-1,2,()));dy=int(torch.randint(-1,2,()))
                padded=F.pad(image,(1,1,1,1),value=1.)
                augmented=padded[:,:,1+dy:33+dy,1+dx:33+dx]
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    encoded=bridge.encode(augmented)
                    direct=bridge.decode_logits(target+.015*torch.randn_like(target))
                    cycle=bridge.decode_logits(encoded)
                    semantic=F.mse_loss(encoded.float(),target)+.2*(1-F.cosine_similarity(encoded.float(),target).mean())
                    loss=semantic+reconstruction_loss(direct,image)+.5*reconstruction_loss(cycle,image)
                require(bool(torch.isfinite(loss)),'nonfinite bridge loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(params,1.);optimizer.step();step+=1
                fraction=step/prepared['maximum_steps']
                for group in optimizer.param_groups:
                    group['lr']=prepared['learning_rate']*(.1+.9*.5*(1+__import__('math').cos(__import__('math').pi*fraction)))
                if step%100==0:
                    row=dict(step=step,loss=float(loss),semantic_loss=float(semantic),seconds=time.perf_counter()-began)
                    with (execution/'history.jsonl').open('a') as stream: stream.write(json.dumps(row)+'\n')
                    status('training',**{k:v for k,v in row.items() if k!='step'})
                    print(json.dumps(row),flush=True)
                if step%1000==0:
                    checkpoint();save_json(execution/f'assessment_{step:05}.json',evaluate())
            checkpoint()
            report=dict(step=step,elapsed_seconds=time.perf_counter()-began,resource_wait_seconds=waited,
                peak_allocated_vram_gib=torch.cuda.max_memory_allocated()/1024**3,
                minimum_available_gib=min(v['available_gib'] for v in samples),
                source_commit=prepared['source_commit'],preparation_sha256=sha(output/'preparation.json'),
                checkpoint_sha256=sha(execution/'checkpoint_latest.pt'),language_training_complete=False,
                glyphs=len(images),glyph_rendering_only=True)
            save_json(execution/'report.json',report);status('completed',pid=None)
    except BaseException as error:
        status('failed',error=repr(error));raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['prepare','train'],required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--limit',type=int,default=4096)
    args=parser.parse_args()
    if args.stage=='prepare': prepare(args.output.resolve(),args.limit)
    else: train(args.output.resolve())


if __name__=='__main__': main()
