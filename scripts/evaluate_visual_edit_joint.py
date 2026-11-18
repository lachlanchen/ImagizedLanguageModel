"""Evaluate local joint edit/pattern choice and conditional real-history diagnostics."""
import fcntl
import time
from collections import Counter

import numpy as np
import torch
from safetensors.torch import load_file

from ilm.visual_lm.idiom_ink import nibbles
from ilm.visual_lm.visual_edit_fertility import FertilityEditor
from ilm.visual_lm.visual_edit_writer import DROP,INSERT,REPLACE,START,VisualEditWriter,visual_input
from scripts.evaluate_visual_edit_translation import OUT,ROOT,bank,evaluate,guard,inputs,load,local_program,require,write_new
from scripts.train_visual_edit_translation import padded


class JointEditor(VisualEditWriter):
    def generate(self,*args,**kwargs):
        return super().generate(*args,joint_choice=True,**kwargs)


class LimitedJointEditor(FertilityEditor):
    def generate(self,*args,**kwargs):
        return super().generate(*args,joint_choice=True,**kwargs)


@torch.inference_mode()
def teacher(net,tiles):
    rows=[r for r in inputs('heldout') if r['x']]
    ink=torch.as_tensor(np.unpackbits(tiles,axis=1),device='cuda').bool()
    features=visual_input(ink,net.landmarks)
    q=torch.tensor(load(OUT/'preparation.json')['question_tiles'][0][0],device='cuda')[None]
    counts=Counter()
    for first in range(0,len(rows),32):
        batch=rows[first:first+32]
        x,l=padded([r['x'] for r in batch])
        a,ns=padded([r['a'] for r in batch])
        o,_=padded([r['o'] for r in batch])
        x,l,a,ns,o=[torch.as_tensor(v,device='cuda').long() for v in (x,l,a,ns,o)]
        pos=torch.arange(a.shape[1],device='cuda')[None]
        mask=pos<ns[:,None]
        advance=(a!=INSERT)&(a!=4)&mask
        cursor=torch.nn.functional.pad(advance.cumsum(1)[:,:-1],(1,0))
        previous=torch.nn.functional.pad(a[:,:-1],(1,0),value=START)
        emits=(a<=INSERT)&mask
        last=torch.where(emits,pos+1,0).cummax(1).values
        ids=torch.nn.functional.pad(o,(1,0)).gather(1,torch.nn.functional.pad(last[:,:-1],(1,0)))
        valid=torch.arange(x.shape[1],device='cuda')[None]<l[:,None]
        qq=q.expand(len(batch),-1)
        memory,g=net.encode(features[x],valid,features[qq],torch.ones_like(qq,dtype=torch.bool))
        h=net.teacher(memory,g,cursor,features[ids],previous)
        logits=net.action_logits(h,cursor==l[:,None])
        predicted=logits.argmax(-1)
        paint=mask & ((a==INSERT)|(a==REPLACE))
        states=h[paint]
        bits,logp=net.draw.choose(states,return_score=True)
        target=ink[o[paint]]
        # Recount the beam's complete code probability from a separate teacher pass.
        own=net.draw.teacher(states,bits).log_softmax(-1).gather(-1,nibbles(bits)[:,:,None]).squeeze(-1).sum(-1)
        torch.testing.assert_close(own,logp,atol=3e-4,rtol=3e-4)
        bitmap=net.motor(bits)>0
        counts.update(positions=int(mask.sum()),action_correct=int(((predicted==a)&mask).sum()),
            source_draw_positions=int(paint.sum()),draw_action_correct=int((predicted[paint]==a[paint]).sum()),
            measurements_correct=int((bits.bool()==target[:,net.landmarks]).all(-1).sum()),
            emitted_glyph_correct=int((bitmap==target).all(-1).sum()),
            true_code_render_correct=int(((net.motor(target[:,net.landmarks].float())>0)==target).all(-1).sum()))
    return dict(counts)


def main():
    begin=time.monotonic()
    guard(begin)
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('high')
    train=inputs('train')
    program=local_program(train)
    tiles=bank()
    folder=OUT/'joint_choice'
    folder.mkdir()
    for name,kind,path in [('joint',JointEditor,OUT/'motor_refined/editor.safetensors'),
                           ('joint_limited',LimitedJointEditor,OUT/'fertility/editor.safetensors')]:
        state=load_file(str(path))
        args=(state['landmarks'],) if name=='joint' else (state['landmarks'],load(OUT/'fertility/report.json')['maximum'])
        net=kind(*args).cuda().eval()
        net.load_state_dict(state)
        output=folder/name
        output.mkdir()
        if name=='joint':
            write_new(folder/'teacher_history.json',teacher(net,tiles))
        stats=evaluate('heldout',net,tiles,program,train,begin,output/'heldout')
        if stats['outcome']['translation']['hard_exact']:
            evaluate('confirmation',net,tiles,program,train,begin,output/'confirmation')
    write_new(folder/'report.json',dict(seconds=time.monotonic()-begin,peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2))


if __name__=='__main__':
    with (ROOT/'artifacts/ilm_gpu0.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        main()
