"""Score the autonomously written position, accompanying human quotation, and previous capabilities."""
import numpy as np
import torch

from ilm.visual_lm.pictured_record_selection import select_visible_record
from scripts.audit_paired_stance_images import INSTRUCTION, NEGATIVE, POSITIVE
from scripts.audit_traditional_passage_training import clustered_delta
from scripts.evaluate_english_request_writer import prompts
from scripts.evaluate_general_entry_writer import pictured
from scripts.evaluate_same_passage_writer import generate, grade, image_archive, reference_sets
from scripts.train_general_entry_writer import PairCorpus, small
from scripts.train_paired_stance_writer import (ARMS, CHECKPOINT, DEVICE, Entries, PUBLIC, Rasterizer, SPLIT, WORK,
        corpus, load, load_visual_writer, memory_bank, physical_tables, sha, write_json)


def metrics(flags):
    flags=np.asarray(flags,dtype=bool).reshape(-1,2)
    return dict(responses=int(flags.size),single=int(flags.sum()),both=int(flags.all(1).sum()))


def score(images,stops,targets,passages,prefixes):
    tiles=lambda x: tuple(x[j:j+128] for j in range(0,len(x),128))
    out={k:[] for k in ("position","full","body","valid_prefix","verbatim_body","first_body","stopped")}
    for image,end,target,source in zip(images,stops,targets,passages,strict=True):
        prefix,body=image[:384],image[384:]
        assert target[:384] in prefixes
        legal=prefix in prefixes
        out["position"].append(prefix==target[:384])
        out["valid_prefix"].append(legal)
        out["full"].append(end and image==target)
        out["body"].append(end and legal and body==target[384:])
        out["first_body"].append(legal and body[:128]==target[384:512])
        q,p=tiles(body),tiles(source)
        literal=bool(q) and any(q==p[i:i+len(q)] for i in range(len(p)-len(q)+1))
        out["verbatim_body"].append(end and legal and literal)
        out["stopped"].append(end)
    return out


def known_baselines(ds,rows,bank,r):
    markers=[[tuple(pictured(bank,r,v)) for v in xs] for xs in (POSITIVE,NEGATIVE)]
    punctuation=set(pictured(bank,r,"。！？!?；;"))
    correct,mid,first=[],[],[]
    for i in rows:
        p=tuple(ds.p[i]); target=tuple(ds.y[i][3:])
        def contains(m):
            return any(p[k:k+len(m)]==m for k in range(len(p)-len(m)+1))
        signs=[sum(map(contains,ms)) for ms in markers]
        correct.append((signs[0]>=signs[1])==(i%2==0))
        spans=[];left=0
        for j,c in enumerate(p):
            if c in punctuation:
                spans.append((left,j+1));left=j+1
        if left<len(p):
            spans.append((left,len(p)))
        a,b=next((a,b) for a,b in spans if a<=len(p)//2<b)
        mid.append(target==p[a:b])
        first.append(target==p[:spans[0][1]])
    return dict(marker=correct,center_sentence=mid,first_sentence=first)


@torch.inference_mode()
def evaluate():
    split=load(SPLIT)
    bank=memory_bank()
    valid=corpus(bank,"train")
    official=corpus(bank,"development")
    r=Rasterizer()
    intro=pictured(bank,r,INSTRUCTION)
    polar=[b"".join(r.tile(c) for c in text) for text in ("肯定。","否定。")]
    blank=np.array([bank.lookup[bytes(128)]])
    entries=Entries(bank,"development")
    panels=prompts(bank,entries)
    bounds=pictured(bank,r,"【】")
    du,groups=small(bank,"development")
    dr=PairCorpus(bank,"development")
    ids=np.unique(dr.rows[:,1:3]);pairs=np.searchsorted(ids,dr.rows[:,1:3])
    dr_refs=reference_sets(dr,ids,bank)
    def image(seq):
        return b"".join(bank.tiles[k] for k in seq)
    du_refs=[{image(y)} for y in du.y]
    cohorts={}
    baselines={}
    for name,ds in (("validation",valid),("official",official)):
        rows=(2*np.array(split[name])[:,None]+np.arange(2)).ravel()
        cohorts[name]=(rows,[ds.p[i] for i in rows],[ds.q[i] for i in rows],[image(ds.y[i]) for i in rows])
        baselines[name]=known_baselines(ds,rows,bank,r)
    feats,memory,_=physical_tables(bank,DEVICE)
    records={}; flags={}
    for arm in ("parent",*ARMS):
        directory=WORK/arm
        if arm=="parent":
            directory.mkdir()
        net=load_visual_writer(CHECKPOINT if arm=="parent" else directory,DEVICE)
        ink=net.ink(feats)
        keys=net.write_key(ink[memory])
        report={}; arm_flags={}
        for name,(rows,p,q,targets) in cohorts.items():
            for mode,ps,qs in (("full",p,q),("passage_erased",[blank]*len(q),q),
                               ("question_erased",p,[intro]*len(q))):
                tag=name+"_"+mode
                images,stops=generate(net,ink,memory,keys,ps,qs,bank)
                image_archive(directory/(tag+".npz"),images,stops)
                credited=score(images,stops,targets,[image(seq) for seq in p],polar)
                report[tag]={k:metrics(v) for k,v in credited.items()}
                if mode=="full":
                    arm_flags[name]=credited
                if mode=="passage_erased":
                    assert images[::2]==images[1::2] and not report[tag]["position"]["both"]
        for tag in ("english_free","chinese_free"):
            qs,rows=panels[tag]
            ps=[select_visible_record(entries.cards[entries.card_of[i]],q,*bounds)[0] for i,q in zip(rows,qs,strict=True)]
            images,ends=generate(net,ink,memory,keys,ps,qs,bank)
            image_archive(directory/(tag+".npz"),images,ends)
            report[tag]=grade(images,ends,[{image(entries.y[i])} for i in rows])[0]
        for name,p,q,refs,g in (("dureader",du.p,du.q,du_refs,groups),
                                ("drcd",[dr.passages[dr.groups[i]] for i in ids],[dr.questions[i] for i in ids],dr_refs,None)):
            images,ends=generate(net,ink,memory,keys,p,q,bank)
            image_archive(directory/(name+".npz"),images,ends)
            report[name],ok=grade(images,ends,refs,g)
            if name=="drcd":
                report[name]["both"]=int(ok[pairs].all(1).sum())
        flags[arm]=arm_flags
        records[arm]=report
        print(arm,{k:v["position"] for k,v in report.items() if k.endswith("_full")},flush=True)
    contrasts={}
    for name in cohorts:
        candidate=np.asarray(flags["matched"][name]["position"]).reshape(-1,2).all(1)
        for lab,ind in (("reassigned",flags["reassigned"][name]["position"]),("marker",baselines[name]["marker"])):
            other=np.asarray(ind).reshape(-1,2).all(1)
            contrasts[name+"_"+lab]=clustered_delta(candidate,other,np.arange(len(candidate)))
    result=dict(arms=records,contrasts=contrasts,baselines={n:{k:metrics(v) for k,v in base.items()}
                for n,base in baselines.items()},fits={a:load(WORK/a/"fit.json") for a in ARMS},
                archives={str(p.relative_to(WORK)):sha(p) for p in WORK.rglob("*.npz")})
    write_json(PUBLIC/"result.json",result)
