#!/usr/bin/env python3
"""Retrospective writer-only check on previously saved real first states.

No new image prefill, source lexical projection, language rollout or training.
These already-observed states cannot be called a new held-out language test.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image,ImageDraw

from ilm.image_edit.baseline import ROOT,sha,save_json
from scripts.score_pixel_language_first_blocks import require

NAMES=['scripts/check_transferred_first_images.py','references/transferred_first_images_protocol.md']


def load(p):return json.loads(Path(p).read_text())


def main(run,output):
    require(not output.exists(),'preserve previous diagnostics')
    for name in NAMES:
        require(subprocess.check_output(['git','show','HEAD:'+name],cwd=ROOT)==(ROOT/name).read_bytes(),'commit diagnostic first')
    from scripts.transfer_glyph_first_function import verify,read_shards
    from scripts.train_semantic_glyph_bridge import bridge_gpu,atomic_tensor
    from scripts.train_semantic_glyph_language import bridge_model,numerics
    from scripts.distill_semantic_glyph_readout import Monitor
    import torch
    m=verify(run);require(load(run/'qualify/status.json')['phase']=='completed','finish function qualification first')
    # A failed function gate remains failed. This check is explanatory and does
    # not license further language claims or a different checkpoint selection.
    audit=ROOT/'references/evidence/glyph_first_policy_audit_v1/report.json'
    records=load(audit)['records'];require(len(records)==128,'all prior first states required')
    inventory=load(ROOT/m['primitive']/'inventory.json')['records']
    charmap={c:i for i,r in enumerate(inventory) for c in r['chars']}
    gold=torch.load(ROOT/m['primitive']/'training.pt',weights_only=True,map_location='cpu')['ink'][:,0].numpy().astype(np.uint8)
    bindings={str(audit.relative_to(ROOT)):sha(audit)};states=[]
    for row in records:
        group,index,mode=row['key'].split('/')
        p=ROOT/'artifacts'/row['run']/'assessment'/f'{group}_{index}_{mode}.pt'
        bindings[str(p.relative_to(ROOT))]=sha(p)
        v=torch.load(p,weights_only=True,map_location='cpu');states.append(v['states'][0])
    h=torch.stack(states).float();require(h.shape==(128,4096),'actual first-state dimensions differ')
    output.mkdir(parents=True)
    limits=dict(maximum_stage_seconds=600.,maximum_vram_gib=6.,minimum_entry_ram_gib=48.,
                minimum_runtime_ram_gib=32.,maximum_pressure_wait_seconds=300.)
    monitor=Monitor(output,limits);monitor.status('admitting')
    try:
        with bridge_gpu(limits) as (device,admission),torch.no_grad():
            numerics();save_json(output/'admission.json',admission);monitor.guard()
            summaries={};raw={};boards=[];all_records=[]
            for version,path in [('balanced_parent',ROOT/m['initial_checkpoint']),('transferred',run/'fit/checkpoint_latest.pt')]:
                model=bridge_model({},device,path);logits,_,_=model.predict(h.to(device));ink=(logits>=0).to(torch.uint8)[:,0].cpu().numpy()
                raw[version]=torch.from_numpy(ink)
                for start in range(0,128,32):
                    name=records[start]['run'];rows=[]
                    canvas=Image.new('L',(600,580),255);d=ImageDraw.Draw(canvas)
                    d.text((4,4),version+' / '+name+'; ref | actual',fill=0)
                    for j in range(start,start+32):
                        row=records[j];refs=row['allowed_reference_first_characters'];scores=[]
                        for c in refs:
                            if c not in charmap:continue
                            target=gold[charmap[c]];actual=ink[j];den=int(target.sum()+actual.sum())
                            scores.append(dict(char=c,f1=2*int((target&actual).sum())/den if den else 1.,exact=bool(np.array_equal(target,actual))))
                        result=dict(version=version,run=row['run'],key=row['key'],reference_first_scores=scores,
                            source_first_reference_match=row['teacher_first_character_reference_match'],
                            physical_f1_at_least095=any(v['f1']>=.95 for v in scores),
                            canonical_exact=any(v['exact'] for v in scores))
                        rows.append(result);all_records.append(result)
                        local=j-start;x=(local%4)*149;y0=25+(local//4)*69;d.text((x,y0),row['key'],fill=0)
                        # The first supported reference is only a display. All
                        # allowed references remain in the physical scores.
                        refs_supported=[c for c in refs if c in charmap]
                        target=gold[charmap[refs_supported[0]]] if refs_supported else np.zeros((32,32),np.uint8)
                        for k,img in enumerate([target,ink[j]]):canvas.paste(Image.fromarray(255*(1-img)),(x+35*k,y0+16))
                    summaries[version+'/'+name]=dict(cases=32,physical_f1_at_least095=sum(v['physical_f1_at_least095'] for v in rows),
                        canonical_first_exact=sum(v['canonical_exact'] for v in rows),
                        source_correct_but_physical_below095=sum(v['source_first_reference_match'] and not v['physical_f1_at_least095'] for v in rows))
                    board=output/(version+'_'+name+'.png');canvas.save(board);boards.append(dict(file=board.name,sha256=sha(board)))
                del model
            atomic_tensor(output/'outputs.pt',raw)
            save_json(output/'report.json',dict(code_sha256={n:sha(ROOT/n) for n in NAMES},bindings_sha256=bindings,
                summaries=summaries,records=all_records,boards=boards,raw_sha256=sha(output/'outputs.pt'),
                checkpoint_sha256=sha(run/'fit/checkpoint_latest.pt'),retrospective=True,new_heldout_evidence=False,
                source_head_loaded=False,full_body_loaded=False,source_states_used_for_fit=False,
                optimizer_updates=0,closed_loop_language_measured=False,**monitor.receipt()))
            print(json.dumps(summaries),flush=True)
        monitor.status('completed')
    except BaseException as error:monitor.status('failed',reason=repr(error));raise
    finally:save_json(output/'status.json',dict(load(output/'status.json'),pid=None))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,default=Path('artifacts/glyph_first_function_transfer_v1'))
    p.add_argument('--output',type=Path,default=Path('artifacts/transferred_first_images_v1'));a=p.parse_args()
    main(a.run.resolve(),a.output.resolve())
