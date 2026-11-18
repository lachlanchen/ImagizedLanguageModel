"""Matched planner training with a fixed or variable pictured instruction."""
import fcntl
from time import monotonic
import numpy as np
import torch
from torch.nn import functional as F
from safetensors.torch import load_file, save_model
from ilm.visual_lm.passage_composer import PassageComposer, IGNORE, READ, JOIN, READ_STOP
from scripts import test_official_passage_composition as external
from scripts.train_paired_passage import Corpus, WORK, ROOT, DEST, MODEL, SEED, pad
from scripts.prepare_pictured_entry_answers import Rasterizer, load, sha, write_json
from scripts.train_visual_extractive_reader import load_cases

OUT = WORK/"instruction_comparison"
TRAIN = ("请依次回答这两个问题：", "请回答：", "按顺序回答下面的问题：", "作答：",
         "请按给出的先后顺序写出两个答案：", "阅读材料后回答下列问题：")
TEST = (*external.WRAPPERS, "请按次序作答：", "Answer in order: ")


def encode_instructions(corpus):
    all_tiles=list(map(bytes,corpus.bank))
    index={s:i for i,s in enumerate(all_tiles)}
    render=Rasterizer()
    prefixes=[]
    for text in TRAIN:
        row=[]
        for ch in text:
            tile=render.tile(ch)
            if tile not in index:
                index[tile]=len(all_tiles)
                all_tiles.append(tile)
            row.append(index[tile])
        prefixes.append(np.array(row,np.int32))
    assert np.array_equal(prefixes[0],corpus.prefix)
    return np.frombuffer(b"".join(all_tiles),np.uint8).reshape(-1,128),prefixes


def fit(varied,corpus,tiles,prefixes):
    torch.manual_seed(SEED)
    net=PassageComposer().cuda()
    net.reader.load_state_dict(load_file(str(MODEL/"mixed_reader.safetensors")))
    net.landmarks.copy_(load_file(str(MODEL/"reader_writer.safetensors"))["landmarks"])
    net.reader.requires_grad_(False)
    features=net.features(tiles)
    opt=torch.optim.AdamW(net.planner.parameters(),lr=.001)
    rng=np.random.default_rng(SEED)
    weights=features.new_tensor([1.,1.,5.,5.])
    history=[]
    for epoch in range(6):
        prompts=[]
        labels=[]
        for i,(_,a,b) in enumerate(corpus.pairs):
            pre=prefixes[(i+epoch)%len(prefixes) if varied else 0]
            first,second=corpus.questions[a],corpus.questions[b]
            prompts.append(np.concatenate((pre,first,corpus.join,second)))
            labels.append(np.array([IGNORE]*len(pre)+[READ]*len(first)+[JOIN]+[READ]*(len(second)-1)+[READ_STOP]))
        for q in corpus.questions:
            prompts.append(q)
            labels.append(np.array([READ]*(len(q)-1)+[READ_STOP]))
        loss_sum=normalizer=0.
        for ids in np.array_split(rng.permutation(len(prompts)),(len(prompts)+127)//128):
            x,lens=pad([prompts[i] for i in ids],features.device)
            y,_=pad([labels[i] for i in ids],features.device,-100)
            logits=net.plan(features,x,lens)
            loss=F.cross_entropy(logits.transpose(1,2),y.long(),weight=weights,reduction="sum")
            norm=weights[y[y>=0].long()].sum()
            opt.zero_grad(set_to_none=True)
            (loss/norm).backward()
            opt.step()
            loss_sum+=float(loss.detach())
            normalizer+=float(norm)
        history.append(loss_sum/normalizer)
    return net.eval(),history


@torch.inference_mode()
def internal(net,data):
    prompts,labels=data.requests()
    features=net.features(data.bank)
    tags=[]
    for l in range(0,len(prompts),128):
        q,lens=pad(prompts[l:l+128],features.device)
        tags.extend(net.plan(features,q,lens).argmax(-1).cpu().numpy())
    # Deduplicate identical passage+question inputs for the frozen reader.
    cache={}
    ended=[]
    groups=list(data.pairs[:,0])+data.groups
    from scripts.fit_mixed_ink_reader import predict
    jobs=[]
    plans=[]
    for g,q,actions in zip(groups,prompts,tags,strict=True):
        parts,stop=net.regions(actions[:len(q)])
        codes=[]
        for kind,a,b in parts:
            if kind=="prompt": codes.append(data.bank[q[a:b]].tobytes())
            else:
                k=(g,q[a:b].tobytes())
                if k not in cache:
                    cache[k]=len(jobs)
                    jobs.append((data.passages[g],q[a:b]))
                codes.append(cache[k])
        plans.append(codes)
        ended.append(stop)
    spans=predict(net.reader,features,jobs)
    ink=[data.bank[p[a:b]].tobytes() for (p,_),(_,a,b) in zip(jobs,spans,strict=True)]
    outputs=[b"".join(ink[x] if isinstance(x,int) else x for x in plan) for plan in plans]
    return outputs,ended


def main():
    for name,digest in load(ROOT/"references/evidence/passage_instruction_registration.json")["files"].items():
        assert sha(ROOT/name)==digest
    OUT.mkdir(exist_ok=False)
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.86)
    began=monotonic()
    raw=load_cases(began)
    training,dev=(Corpus(s,c) for s,c in zip(("train","development"),raw,strict=True))
    tiles,prefixes=encode_instructions(training)
    # Use the unchanged official harness with this experiment's wrappers and output root.
    external.WRAPPERS=TEST
    external.OUT=OUT
    official=external.prepare()
    stats=dict(official_questions=len(official[1]),official_pairs=len(official[3]),official_passages=len(official[0]))
    for key,varied in (("fixed",False),("varied",True)):
        start=monotonic()
        net,history=fit(varied,training,tiles,prefixes)
        fit_seconds=monotonic()-start
        file=OUT/(key+".safetensors")
        save_model(net,str(file))
        official_score=external.evaluate_checkpoint(net,key,official)
        images,stops=internal(net,dev)
        n=len(dev.pairs)
        exact=np.array([s and im==dev.target(i) for i,(s,im) in enumerate(zip(stops,images,strict=True))])
        file2=OUT/(key+"_internal.npz")
        np.savez_compressed(file2,pixels=np.frombuffer(b"".join(images),np.uint8).reshape(-1,128),
                            offsets=np.cumsum([0]+[len(im)//128 for im in images]),stopped=stops)
        stats[key]=dict(official=official_score,pair_exact=int(exact[:n].sum()),
                        single_exact=int(exact[n:].sum()),both_orders=int(exact[:n].reshape(-1,2).all(1).sum()),
                        losses=history,fit_seconds=fit_seconds,weights_sha256=sha(file),internal_sha256=sha(file2))
        print(key,stats[key],flush=True)
    stats.update(seconds=monotonic()-began,peak_allocated_bytes=torch.cuda.max_memory_allocated())
    write_json(OUT/"comparison.json",stats)


if __name__=="__main__":
    with (WORK/"instruction_owner.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        main()
