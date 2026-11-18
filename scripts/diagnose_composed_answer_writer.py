"""Score a frozen writer with actual preceding answer segments supplied, on CPU."""
from collections import Counter
import numpy as np
import torch
from safetensors.torch import load_file,load_model

from ilm.visual_lm.composed_answer_writer import COPY,WRITE,STOP,MAX_SPAN,ComposedAnswerWriter,descriptors
from ilm.visual_lm.ordered_passage_planner import OrderedPassageComposer
from scripts.train_composed_answer_writer import (
    BATCH, OUT, PRIOR, Split, batch_inputs, segments, source_fields, teacher_batch, tensor,
)
from scripts.prepare_pictured_entry_answers import write_json


@torch.inference_mode()
def diagnose(net, prior, memory_tiles, split):
    device="cpu"
    ds=Split(split)
    fields=source_fields(prior,ds,ds.q,device)
    idx={bytes(g):i for i,g in enumerate(memory_tiles)}
    lookup={i:idx.get(bytes(g),-1) for i,g in enumerate(ds.tiles)}
    alignments=[segments(p,q,y,lookup) for p,q,y in zip(ds.p,ds.q,ds.y,strict=True)]
    image=tensor(np.unpackbits(ds.tiles,axis=1),device)
    dictionary=tensor(np.unpackbits(memory_tiles,axis=1),device)
    ink=net.ink(descriptors(image))
    literals=net.ink(descriptors(dictionary))
    count=Counter()
    confusion=np.zeros((3,3),np.int64)
    for i in range(0,len(ds.y),BATCH):
        ix=list(range(i,min(i+BATCH,len(ds.y))))
        ids,f,mask=batch_inputs(ds,ds.q,fields,ix,device)
        data=teacher_batch(ds,alignments,ix,ids.shape[1])
        act,start,end,cur,_,first,last,mean_ids,counts,aliases=[tensor(a,device) for a in data]
        x,h=net.encode(ink[ids],f,mask)
        spanmask=torch.arange(MAX_SPAN)<counts[...,None]
        mean=(ink[mean_ids]*spanmask.unsqueeze(-1)).sum(-2)/counts[...,None]
        fb=net.feedback_ink(ink[first],mean,ink[last])
        fb[:,0]=0
        state,_=net.recurrence(fb,h)
        gate,pointer,focus=net.plan(x,f,mask,state,cur)
        best=gate.argmax(-1)
        for target in (COPY,WRITE,STOP):
            confusion[target]+=torch.bincount(best[act==target],minlength=3).numpy()
        copy=act==COPY
        guess=pointer.argmax(-1)
        good=aliases.gather(-1,guess[...,None]).squeeze(-1)
        endings=net.endings(x,f,mask,focus,guess).argmax(-1)
        count["copy_steps"]+=int(copy.sum())
        count["right_start"]+=int((copy&good).sum())
        count["right_span"]+=int((copy&good&((endings-guess)==(end-start))).sum())
        write=act==WRITE
        if write.any():
            ctx=focus[write]
            target_ids=[ds.y[idx0][a] for idx0 in ix for kind,_,_,_,_,a,_ in alignments[idx0] if kind==WRITE]
            truth=image[tensor(np.array(target_ids),device)]
            shortlist=net.bitmap_scores(ctx,literals).topk(5,dim=-1).indices
            template=dictionary[shortlist[:,0]]
            raster=net.raster_logits(ctx,template)>=0
            correct=template.eq(truth).all(1)
            count["write_steps"]+=len(truth)
            count["known_targets"]+=sum(lookup[t]>=0 for t in target_ids)
            count["finite_exact"]+=int(correct.sum())
            count["finite_top_five"]+=int(dictionary[shortlist].eq(truth[:,None]).all(-1).any(1).sum())
            rendered_correct=(raster==truth).all(-1)
            count["raster_exact"]+=int(rendered_correct.sum())
            count["correct_anchors_damaged"]+=int((correct & ~rendered_correct).sum())
            count["wrong_anchors_repaired"]+=int((~correct & rendered_correct).sum())
            count["correct_anchors_preserved"]+=int((correct & rendered_correct).sum())
            count["raster_changed"]+=int((raster!=template).any(-1).sum())
    return dict(confusion_reference_by_prediction=confusion.tolist(),**count)


if __name__=="__main__":
    torch.set_num_threads(4)
    model=ComposedAnswerWriter().eval()
    model.load_state_dict(load_file(str(OUT/"final.safetensors")),strict=True)
    prior=OrderedPassageComposer().eval()
    load_model(prior,str(PRIOR))
    tiles=np.load(OUT/"memory.npy")
    results={name:diagnose(model,prior,tiles,name) for name in ("development","train")}
    print(results)
    write_json(OUT/"supplied_history.json",results)
