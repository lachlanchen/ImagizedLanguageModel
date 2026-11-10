#!/usr/bin/env python3
"""Retrospective CPU separation of source first policy and actual glyph pixels."""
import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np

from ilm.image_edit.baseline import ROOT,sha,save_json
from scripts.score_pixel_language_first_blocks import require,host_resources

OWN='scripts/audit_glyph_first_policy.py'
PROTOCOL='references/glyph_first_policy_audit_protocol.md'
RUNS=['glyph_tail_adaptation_v2','glyph_tail_readout_control_v2','balanced_glyph_writer_v1','balanced_glyph_parent_v1']


def load(path):return json.loads(Path(path).read_text())


def main(output):
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU-only source head audit')
    require(not output.exists(),'preserve prior diagnostic reports')
    names=[OWN,PROTOCOL,'scripts/check_native_visual_transfer_v71.py','ilm/visual_lm/grounded_image_weights.py']
    for name in names:
        require(subprocess.check_output(['git','show','HEAD:'+name],cwd=ROOT)==(ROOT/name).read_bytes(),'commit first-policy auditor')
    with (ROOT/'artifacts/ilm_gpu0.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        entry=host_resources();require(entry['available_gib']>=48 and entry['some_pressure_avg10']<=.5
            and entry['full_pressure_avg10']<=.1,'CPU audit resource entry limit')
        import torch
        from torch.nn import functional as F
        from scripts.check_native_visual_transfer_v71 import archive_and_tokenizer
        torch.set_num_threads(2);began=time.monotonic();minimum=entry['available_gib']
        archive,tokenizer,_=archive_and_tokenizer();head_source=archive.omitted['output.weight']
        array=np.array(archive.gguf.dequantize(head_source.data,head_source.tensor_type),dtype=np.float32,copy=True)
        head=torch.from_numpy(array).to(torch.bfloat16).float();del array
        eog=set(map(int,archive.field('tokenizer.ggml.eos_token_ids')))
        bindings={};results=[];initials={}
        def bind(path):bindings[str(path.relative_to(ROOT))]=sha(path)
        for run_name in RUNS:
            run=ROOT/'artifacts'/run_name;score_path=run/'raw_review/scores.json';bind(score_path)
            score=load(score_path);m=load(run/'preparation.json');bind(run/'preparation.json')
            primitive=ROOT/m['primitive'];bind(primitive/'training.pt');bind(primitive/'inventory.json')
            data=torch.load(primitive/'training.pt',weights_only=True,map_location='cpu')['ink'][:,0].numpy().astype(np.uint8)
            charmap={c:i for i,row in enumerate(load(primitive/'inventory.json')['records']) for c in row['chars']}
            rows=[r for r in score['records'] if r['key'].startswith('fresh/')];require(len(rows)==32,'retain complete reserved cohort')
            for r in rows:
                observed=host_resources();minimum=min(minimum,observed['available_gib'])
                require(minimum>=24 and time.monotonic()-began<600,'CPU first-policy resource/deadline limit')
                path=run/'assessment'/r['file'];bind(path);require(sha(path)==r['sha256'],'actual answer tensor changed')
                value=torch.load(path,weights_only=True,map_location='cpu');h=value['states'][0].float()[None]
                require(tuple(h.shape)==(1,4096) and torch.isfinite(h).all(),'actual first state differs')
                logits=F.linear(h,head)[0];top=torch.topk(logits,2);winner=int(top.indices[0])
                text=tokenizer.decode([winner],skip_special_tokens=False)
                supported=bool(text and '\ufffd' not in text and winner not in eog)
                first=text[0] if supported else None
                first_refs=sorted({s[0] for s in r['references'] if s})
                physical_scores=[]
                actual=(value['images'][0,0].numpy()<0).astype(np.uint8) if len(value['images']) else None
                for char in first_refs:
                    if char not in charmap:continue
                    gold=data[charmap[char]]
                    denominator=int(actual.sum()+gold.sum()) if actual is not None else int(gold.sum())
                    f1=(2*int((actual&gold).sum())/denominator if denominator else 1.) if actual is not None else 0.
                    physical_scores.append(dict(reference_first=char,foreground_f1=f1,
                        canonical_exact=actual is not None and bool(np.array_equal(actual,gold))))
                initials[run_name,r['key']]=h.clone()
                results.append(dict(run=run_name,key=r['key'],source_unit=text,source_id=winner,
                    original_head_on_actual_adapted_state=run_name!='glyph_tail_readout_control_v2',
                    teacher_first_character=first,teacher_terminal=winner in eog,
                    teacher_unsupported_or_partial=not supported and winner not in eog,
                    teacher_first_character_reference_match=first in first_refs,
                    source_top_two_logit_margin=float(top.values[0]-top.values[1]),
                    allowed_reference_first_characters=first_refs,physical_first_scores=physical_scores,
                    physical_first_canonical_exact=any(x['canonical_exact'] for x in physical_scores),
                    physical_first_F1_at_least095=any(x['foreground_f1']>=.95 for x in physical_scores),
                    cases_with_supported_first_reference=bool(physical_scores),
                    existing_clear_complete_answer_correct=r['manual_clear_complete_exact']))
        require(all(torch.equal(initials['balanced_glyph_writer_v1',r['key']],initials['balanced_glyph_parent_v1',r['key']])
            for r in results if r['run']=='balanced_glyph_writer_v1'),'unchanged reader first states differ')
        groups={}
        for name in RUNS:
            part=[r for r in results if r['run']==name]
            joint={f'teacher_{a}_physical095_{b}':sum(r['teacher_first_character_reference_match']==a
                and r['physical_first_F1_at_least095']==b for r in part) for a in [False,True] for b in [False,True]}
            groups[name]=dict(cases=len(part),teacher_first_match=sum(r['teacher_first_character_reference_match'] for r in part),
                physical_first_F1_at_least095=sum(r['physical_first_F1_at_least095'] for r in part),
                physical_first_canonical_exact=sum(r['physical_first_canonical_exact'] for r in part),joint=joint,
                existing_clear_complete_matches=sum(r['existing_clear_complete_answer_correct'] for r in part),
                minimum_source_top_two_margin=min(r['source_top_two_logit_margin'] for r in part))
        require(not torch.cuda.is_initialized(),'CPU diagnostic initialized CUDA')
        save_json(output,dict(code_sha256={n:sha(ROOT/n) for n in names},bindings_sha256=bindings,
            records=results,groups=groups,unchanged_reader_first_states_bit_exact=True,
            helper_only_lexical_projection=True,new_heldout_evidence=False,retrospective_analysis=True,
            complete_source_answer_accuracy_measured=False,body_forward_calls=0,optimizer_updates=0,
            training_data_reuse_allowed=False,source_projection_exported=False,CPU_FP32_matmul=True,
            source_matrix_rounding='BF16 then FP32; CPU accumulation may differ from GPU',
            minimum_available_gib=minimum,elapsed_seconds=time.monotonic()-began,CUDA_initialized=False))
        print(json.dumps(groups,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    main(p.parse_args().output.resolve())
