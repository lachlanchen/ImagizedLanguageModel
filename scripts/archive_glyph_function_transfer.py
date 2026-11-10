#!/usr/bin/env python3
"""Archive completed function-transfer failure and every qualification pixel."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil

import numpy as np

from ilm.image_edit.baseline import ROOT,sha,save_json
from scripts.score_pixel_language_first_blocks import require,host_resources


def load(path):return json.loads(Path(path).read_text())


def main(run,diagnostic,destination):
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU-only archive')
    require(not destination.exists(),'preserve earlier evidence')
    with (ROOT/'artifacts/ilm_gpu0.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        require(host_resources()['available_gib']>=48,'archive host reserve')
        import torch
        torch.set_num_threads(2)
        for folder in [run/'calibrate',run/'fit',run/'qualify',diagnostic]:
            require(load(folder/'status.json')['phase']=='completed','complete all stages before archive')
        for folder in [run/'qualify',diagnostic]:
            review=load(folder/'visual_review.json');report=load(folder/'report.json')
            require(review['all_boards_actually_viewed'] and review['report_sha256']==sha(folder/'report.json'),'actual matching review required')
            for r in report['boards']:require(sha(folder/r['file'])==r['sha256'],'reviewed board changed')
        m=load(run/'preparation.json');fit=load(run/'fit/report.json')
        checkpoint=run/'fit/checkpoint_latest.pt';require(sha(checkpoint)==fit['checkpoint_sha256'],'candidate changed')
        model=torch.load(checkpoint,weights_only=True,map_location='cpu')
        parent=torch.load(ROOT/m['initial_checkpoint'],weights_only=True,map_location='cpu')
        names=[k for k in model['model'] if k.startswith(('readout.','decoder_input.'))]
        frozen=[k for k in model['model'] if k not in names]
        require(all(torch.equal(model['model'][k],parent['model'][k]) for k in frozen),'frozen model differs')
        require(all(torch.equal(model['body_tail_adapters'][k],parent['body_tail_adapters'][k]) for k in model['body_tail_adapters']), 'body adapters changed')
        steps=[int(v['step']) for v in model['optimizer']['state'].values()]
        require(len(steps)==len(names) and set(steps)=={18000},'actual optimizer steps differ')
        destination.mkdir(parents=True)
        def copy(source,target):
            target=destination/target;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
        for n in ['preparation.json','calibrate/report.json','fit/report.json','fit/history.jsonl','qualify/report.json','qualify/visual_review.json']:
            copy(run/n,n)
        for n in ['report.json','visual_review.json']:copy(diagnostic/n,'real_first/'+n)
        for folder,target in [(run/'qualify','qualify'),(diagnostic,'real_first')]:
            for row in load(folder/'report.json')['boards']:copy(folder/row['file'],target+'/'+row['file'])
        packed={}
        for row in load(run/'calibrate/report.json')['shards']:
            packed['calibrate/'+row['file']]=row['sha256']
        # All binary qualification outputs, not merely the preview selections.
        for file in sorted((run/'qualify').glob('*.pt')):
            value=torch.load(file,weights_only=True,map_location='cpu');ink=value.pop('ink').numpy().astype(np.uint8)
            arrays={n:v.numpy() for n,v in value.items()}
            path=destination/'qualify'/(file.stem+'.npz')
            np.savez_compressed(path,shape=np.array(ink.shape),ink_bits=np.packbits(ink.reshape(-1)),**arrays)
            with np.load(path) as z:
                require(np.array_equal(np.unpackbits(z['ink_bits'],count=ink.size).reshape(ink.shape),ink),'packed image changed')
            packed[str(file.relative_to(run))]=sha(file)
        value=torch.load(diagnostic/'outputs.pt',weights_only=True,map_location='cpu')
        arrays={}
        for n,v in value.items():
            a=v.numpy().astype(np.uint8);arrays[n+'_bits']=np.packbits(a.reshape(-1));arrays[n+'_shape']=np.array(a.shape)
        np.savez_compressed(destination/'real_first/outputs.npz',**arrays)
        copy(ROOT/'data/raw/cmrc2018/LICENCE','CMRC_LICENCE')
        save_json(destination/'checkpoint_audit.json',dict(checkpoint_sha256=sha(checkpoint),
            inherited_checkpoint_sha256=sha(ROOT/m['initial_checkpoint']),trained_model_tensors=names,
            unchanged_frozen_tensors=frozen,all_frozen_parameters_and_body_deltas_bit_exact=True,
            actual_AdamW_tensor_steps=steps,new_updates=18000,teacher_head_in_checkpoint=False))
        save_json(destination/'archive.json',dict(files_sha256={str(p.relative_to(destination)):sha(p) for p in sorted(destination.rglob('*')) if p.is_file()},
            original_large_local_files_sha256=packed,all_qualification_and_retrospective_pixels_retained=True,
            all_16_fixed_preview_boards_actually_viewed=True,function_gate_passed=False,
            complete_answer_generation_run=False,new_heldout_language_test=False,
            deployed_checkpoint_changed=False,goal_complete=False))
        print(destination,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,default=Path('artifacts/glyph_first_function_transfer_v1'))
    p.add_argument('--diagnostic',type=Path,default=Path('artifacts/transferred_first_images_v1'))
    p.add_argument('--destination',type=Path,default=Path('references/evidence/glyph_first_function_transfer_v1'))
    a=p.parse_args();main(a.run.resolve(),a.diagnostic.resolve(),a.destination.resolve())
