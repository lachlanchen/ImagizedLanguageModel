#!/usr/bin/env python3
"""Check actual source decisions under mean removal and exact RMS-shell reflection."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np

from ilm.image_edit.baseline import ROOT,sha,save_json
from scripts.score_pixel_language_first_blocks import require,host_resources

OWN='scripts/audit_glyph_normalization.py'
PROTOCOL='references/glyph_normalization_audit_protocol.md'


def load(path):return json.loads(Path(path).read_text())


def main(output):
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU-only normalization audit')
    require(not output.exists(),'preserve prior normalization checks')
    names=[OWN,PROTOCOL,'scripts/check_native_visual_transfer_v71.py','ilm/visual_lm/semantic_glyph_loop.py']
    for name in names:
        require(subprocess.check_output(['git','show','HEAD:'+name],cwd=ROOT)==(ROOT/name).read_bytes(),'commit normalization audit')
    source=ROOT/'artifacts/glyph_first_policy_audit_v1.json';prior=load(source)
    with (ROOT/'artifacts/ilm_gpu0.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        entry=host_resources();require(entry['available_gib']>=48 and entry['some_pressure_avg10']<=.5
            and entry['full_pressure_avg10']<=.1,'normalization CPU resource entry')
        import torch
        from torch.nn import functional as F
        from scripts.check_native_visual_transfer_v71 import archive_and_tokenizer
        torch.set_num_threads(2);began=time.monotonic();minimum=entry['available_gib']
        archive,tokenizer,_=archive_and_tokenizer();raw=archive.omitted['output.weight']
        a=np.array(archive.gguf.dequantize(raw.data,raw.tensor_type),dtype=np.float32,copy=True)
        head=torch.from_numpy(a).to(torch.bfloat16).float();del a
        gamma=torch.from_numpy(archive.read('language_model.norm.weight')).double()
        zeros=int((gamma==0).sum());require(zeros==0,'record zero-scale exception before defining weighted reflection')
        weight=gamma.square().reciprocal();eog=set(map(int,archive.field('tokenizer.ggml.eos_token_ids')))
        rows=[];bindings={str(source.relative_to(ROOT)):sha(source)}
        def decision(h):
            values=F.linear(h.float(),head)[0];top=values.topk(2);winner=int(top.indices[0])
            text=tokenizer.decode([winner],skip_special_tokens=False)
            first=text[0] if text and '\ufffd' not in text and winner not in eog else None
            return dict(source_id=winner,source_unit=text,first=first,margin=float(top.values[0]-top.values[1]))
        for old in prior['records']:
            observed=host_resources();minimum=min(minimum,observed['available_gib'])
            require(minimum>=24 and time.monotonic()-began<600,'normalization resource/deadline limit')
            run=ROOT/'artifacts'/old['run'];path=run/'assessment'/(old['key'].replace('/','_')+'.pt')
            relative=str(path.relative_to(ROOT));require(sha(path)==prior['bindings_sha256'][relative],'actual first state changed')
            bindings[relative]=sha(path);h=torch.load(path,weights_only=True,map_location='cpu')['states'][0].double()[None]
            raw_decision=decision(h);require(raw_decision['source_id']==old['source_id'],'prior CPU source decision not reproduced')
            centered=decision(h-h.mean(-1,keepdim=True))
            mean=(h*weight).sum(-1,keepdim=True)/weight.sum();reflected=h-2*mean
            shell=float(((reflected/gamma).square().sum()-(h/gamma).square().sum()).abs())
            ln_error=float((F.layer_norm(h,(4096,))-F.layer_norm(reflected,(4096,))).abs().max())
            require(shell<1e-8 and ln_error<1e-10,'weighted-shell/LN identity failed in float64')
            mirror=decision(reflected);refs=old['allowed_reference_first_characters']
            rows.append(dict(run=old['run'],key=old['key'],raw=raw_decision,centered=centered,reflected=mirror,
                raw_first_match=raw_decision['first'] in refs,centered_first_match=centered['first'] in refs,
                source_unit_changed_after_centering=centered['source_id']!=raw_decision['source_id'],
                reflected_first_changed=mirror['first']!=raw_decision['first'],
                reflected_unit_changed=mirror['source_id']!=raw_decision['source_id'],
                actual_mean=float(h.mean()),actual_weighted_mean=float(mean),
                RMS_shell_squared_sum_error=shell,LayerNorm_maximum_error_float64=ln_error))
        pool=ROOT/'artifacts/balanced_glyph_writer_v1/cache/pools.pt'
        p=torch.load(pool,weights_only=True,map_location='cpu')['first'].double()
        weighted=(p*weight).sum(-1)/weight.sum();bindings[str(pool.relative_to(ROOT))]=sha(pool)
        groups={}
        for name in sorted({r['run'] for r in rows}):
            part=[r for r in rows if r['run']==name]
            groups[name]=dict(cases=len(part),raw_first_matches=sum(r['raw_first_match'] for r in part),
                centered_first_matches=sum(r['centered_first_match'] for r in part),
                changed_units_after_centering=sum(r['source_unit_changed_after_centering'] for r in part),
                contrived_reflection_first_character_changes=sum(r['reflected_first_changed'] for r in part),
                contrived_reflection_source_unit_changes=sum(r['reflected_unit_changed'] for r in part),
                actual_weighted_mean_range=[min(r['actual_weighted_mean'] for r in part),max(r['actual_weighted_mean'] for r in part)])
        require(not torch.cuda.is_initialized(),'CPU normalization audit initialized CUDA')
        save_json(output,dict(code_sha256={n:sha(ROOT/n) for n in names},bindings_sha256=bindings,records=rows,groups=groups,
            training_actual_first_weighted_mean_range=[float(weighted.min()),float(weighted.max())],
            training_actual_first_positive_weighted_mean=int((weighted>0).sum()),training_first_cases=len(weighted),
            maximum_LayerNorm_reflection_error=max(r['LayerNorm_maximum_error_float64'] for r in rows),
            maximum_RMS_shell_squared_sum_error=max(r['RMS_shell_squared_sum_error'] for r in rows),
            contrived_pairs_not_proven_reachable_by_real_image_prompts=True,
            normalization_proven_cause_of_answer_failure=False,new_heldout_evidence=False,
            body_forward_calls=0,writer_forward_calls=0,optimizer_updates=0,new_images=0,
            source_head_offline_only=True,training_data_reuse_allowed=False,
            minimum_available_gib=minimum,elapsed_seconds=time.monotonic()-began,CUDA_initialized=False))
        print(json.dumps(groups,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    main(p.parse_args().output.resolve())
