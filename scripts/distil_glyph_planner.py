"""Transfer physical next-pattern probabilities from a frozen passage writer."""
import argparse
from collections import Counter
import fcntl
from pathlib import Path
import resource
import time

import numpy as np
import torch
from torch.nn import functional as F

from ilm.visual_lm.glyph_planner import GlyphPlanner
from ilm.visual_lm.idiom_ink import GROUPS, LANDMARKS, STOP, PicturedIdiom, nibbles, visual_input
from scripts.search_glyph_planner import search
from scripts.train_glyph_planner import (
    DATA, DIVERSITY, FRESH, HINTS, JOINT, NEIGHBOR, OUT as PLAN, PREVIOUS, ROOT, WEIGHT,
    corpora, guard, infer, load, load_split, local_counts, matching_pairs, paired_interval,
    physical_bank, prepare, require, sha, summarize, verified_split, write_new,
)

OUT = ROOT / "artifacts/glyph_planner_transfer_v1"


def restore(kind, folder):
    path = folder / "weights.pt"
    require(sha(path) == load(folder / "fit.json")["weights_sha256"], "Changed checkpoint")
    state = torch.load(path, map_location="cpu", weights_only=True)
    model = kind(state["landmarks"]).cuda().eval()
    model.load_state_dict(state)
    return model, state


@torch.no_grad()
def teacher_probabilities(teacher, pictures, bits, feedback):
    context = teacher.read(pictures)
    tokens = nibbles(bits).flatten(1)
    previous = F.pad(tokens, (1, 0), value=17)
    time_index = torch.arange(previous.shape[1], device=context.device)
    history = F.pad(feedback, (0,0,1,0))[:, time_index//GROUPS]
    seq = teacher.inputs(previous, (time_index%GROUPS).expand_as(previous), history,
                         context[:,None].expand(-1,len(time_index),-1))
    h, _ = teacher.causal(seq, context.unsqueeze(0).contiguous())
    logits = teacher.next_pattern(h).float()
    return logits[:, :-1, :STOP].softmax(-1), logits[:, ::GROUPS].softmax(-1)[..., STOP]


def student_logits(model, pictures, bits, feedback):
    batch, count, _ = bits.shape
    context = model.read(pictures)
    plans = torch.cat((context[:,None], model.advance(context,context,feedback)), dim=1)
    boundaries = model.stop(plans).squeeze(-1).float()
    state = plans[:,:-1].reshape(batch*count,-1)
    labels = nibbles(bits).reshape(-1,GROUPS)
    previous = F.pad(labels[:,:-1],(1,0),value=17)
    pos = torch.arange(GROUPS,device=bits.device)
    valid = torch.arange(LANDMARKS,device=bits.device)[None,:]//4 < pos[:,None]
    prefix = bits.reshape(-1,1,LANDMARKS)*valid
    inputs = model.inputs(previous,pos.expand_as(previous),prefix,state[:,None].expand(-1,GROUPS,-1))
    h,_ = model.causal(inputs,state[None].contiguous())
    return model.next_pattern(h)[...,:STOP].float().reshape(batch,-1,STOP), boundaries


def losses(model, pictures, bits, feedback, teacher_targets):
    logits, stop = student_logits(model,pictures,bits,feedback)
    labels = nibbles(bits).flatten()
    should_stop = torch.zeros_like(stop)
    should_stop[:,-1] = 1
    decisions = labels.numel()+stop.numel()
    hard = (F.cross_entropy(logits.reshape(-1,STOP),labels,reduction="sum") +
            F.binary_cross_entropy_with_logits(stop,should_stop,reduction="sum")) / decisions
    ink_probs, stop_probs = teacher_targets
    soft = (-(logits.log_softmax(-1)*ink_probs).sum()+
            F.binary_cross_entropy_with_logits(stop,stop_probs,reduction="sum")) / decisions
    return hard, soft


def train(start):
    require(load(PLAN/"search/result.json")["arms"]["aligned"]["exact"] == 176, "Missing frozen search comparison")
    _, train_data = corpora()
    x,y = [torch.as_tensor(train_data[k].astype(np.int32),device="cuda") for k in ("context","target")]
    teacher, teacher_weights = restore(PicturedIdiom,DIVERSITY/"diverse")
    teacher.requires_grad_(False)
    features,bits,feedback,_ = physical_bank(teacher_weights,np.load(DIVERSITY/"pool/pixels.npy"))
    OUT.mkdir()
    for arm in ("reference","distilled"):
        net,state = restore(GlyphPlanner,PLAN/"aligned")
        require(torch.equal(state["landmarks"],teacher_weights["landmarks"]),"Incompatible physical coordinates")
        net.train()
        net.reader.eval()
        net.motor.requires_grad_(False)
        trainable = [v for v in net.parameters() if v.requires_grad]
        opt = torch.optim.AdamW(trainable,lr=.0002,weight_decay=.01)
        rng = np.random.default_rng(725)
        torch.manual_seed(725)
        begin = time.monotonic()
        history = []
        for epoch in range(4):
            total_hard, total_soft = 0., 0.
            order = rng.permutation(len(x))
            for offset in range(0,len(order),192):
                if offset%3072==0:
                    guard(start)
                rows=order[offset:offset+192]
                pics=features[x[rows]]
                target=bits[y[rows]]
                past=feedback[y[rows]]
                with torch.autocast("cuda",dtype=torch.bfloat16):
                    targets=teacher_probabilities(teacher,pics,target,past)
                    hard,soft=losses(net,pics,target,past,targets)
                    weight=.5 if arm=="distilled" else 0.
                    loss=(1-weight)*hard+weight*soft
                    if epoch==0 and offset==0:
                        with torch.no_grad():
                            expected=net.trajectory_loss(pics,target,past)
                            torch.testing.assert_close(expected,hard,atol=1e-6,rtol=1e-6)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable,1.,error_if_nonfinite=True)
                opt.step()
                total_hard+=hard.item()*len(rows)
                total_soft+=soft.item()*len(rows)
            history.append(dict(reference_nll=total_hard/len(x),teacher_cross_entropy=total_soft/len(x)))
            print(arm,epoch+1,history[-1],flush=True)
        folder=OUT/arm
        folder.mkdir()
        final={k:v.detach().cpu() for k,v in net.state_dict().items()}
        require(all(torch.equal(final[k],v) for k,v in state.items() if k.startswith("motor.") or k=="landmarks"),
                "Changed renderer")
        require(all(torch.equal(teacher.state_dict()[k].cpu(),v) for k,v in teacher_weights.items()),"Teacher changed")
        torch.save(final,folder/"weights.pt")
        write_new(folder/"fit.json",dict(weights_sha256=sha(folder/"weights.pt"),history=history,
                  seconds=time.monotonic()-begin,examples_seen=len(x)*4,updates=4*((len(x)+191)//192)))


@torch.inference_mode()
def evaluate(start):
    original,combined=corpora()
    folder=OUT/"fresh"
    folder.mkdir()
    older=(FRESH,HINTS/"confirmation",JOINT/"confirmation",PREVIOUS,DIVERSITY/"fresh",DIVERSITY/"pool",
           WEIGHT/"fresh",NEIGHBOR/"fresh",PLAN/"fresh")
    test,bank=prepare(original,load_split(DATA/"validation.npz"),np.load(PLAN/"fresh/pixels.npy"),
                      buckets=range(132,140),destination=folder,excluded=[verified_split(p) for p in older])
    train_frequency=Counter(map(bytes,original["target"]))
    pairs=matching_pairs(test["context"],test["target"],test["group"])
    ink=torch.from_numpy(np.unpackbits(bank,axis=1)).cuda().bool()
    ids=torch.from_numpy(test["context"].astype(np.int32)).cuda()
    known=set(map(bytes,bank[np.unique(combined["target"])]))
    hits={"local_counts":local_counts(combined,test)}
    reports={}
    for name,path in (("teacher",DIVERSITY/"diverse"),("start",PLAN/"aligned"),("reference",OUT/"reference"),
                      ("distilled",OUT/"distilled")):
        dest=folder/name
        if name=="teacher":
            infer(path/"weights.pt",dest,combined,test,bank,start)
        else:
            model,_=restore(GlyphPlanner,path)
            features=visual_input(ink,model.landmarks)
            chunks=[]
            for b in ids.split(128):
                guard(start)
                chunks.append(search(model,model.read(features[b]),4))
            packed,length,stopped=(np.concatenate(v) for v in zip(*chunks))
            dest.mkdir()
            np.savez_compressed(dest/"generated.npz",packed=packed,length=length,stopped=stopped)
        stream=load_split(dest/"generated.npz")
        hit,row=summarize(stream,test,bank,combined,train_frequency,pairs)
        pos=(stream["packed"][:,:4]==bank[test["target"]]).all(-1)&(np.arange(4)<stream["length"][:,None])
        active=np.arange(8)<stream["length"][:,None]
        row.update(prefix_exact=np.logical_and.accumulate(pos,axis=1).sum(0).tolist(),position_exact=pos.sum(0).tolist(),
                   known=sum(tile.tobytes() in known for tile in stream["packed"][active]),emitted=int(active.sum()),
                   archive_sha256=sha(dest/"generated.npz"))
        swapped=np.roll(np.arange(len(pos)),-1)
        swap_hit=(stream["packed"][swapped,:4]==bank[test["target"]]).all((1,2))
        row["swapped_exact"]=int((swap_hit&(stream["length"][swapped]==4)&stream["stopped"][swapped]).sum())
        hits[name],reports[name]=hit,row
        print(name,row["exact"],row["prefix_exact"],flush=True)
    a=hits["distilled"]
    comp={k:dict(gained=int((a&~v).sum()),lost=int((~a&v).sum()),interval_pp=paired_interval(a,v,test["group"]))
          for k,v in hits.items() if k!="distilled"}
    r=reports["distilled"]
    passed=(all(comp[k]["interval_pp"][0]>0 for k in ("reference","teacher","local_counts"))
            and r["known"]>=.99*r["emitted"] and r["both_correct"]>=reports["teacher"]["both_correct"])
    write_new(OUT/"evaluation.json",dict(rows=len(a),groups=len(np.unique(test["group"])),arms=reports,
              local_exact=int(hits["local_counts"].sum()),comparisons=comp,transfer=comp["reference"]["interval_pp"][0]>0,
              qualifies_for_confirmation=bool(passed)))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase",choices=("train","evaluate"))
    phase=parser.parse_args().phase
    with (OUT.parent/"ilm_gpu0.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        begin=time.monotonic()
        guard(begin)
        torch.set_num_threads(2)
        torch.set_float32_matmul_precision("high")
        require("4090" in torch.cuda.get_device_name(),"Expected registered GPU")
        (train if phase=="train" else evaluate)(begin)
        guard(begin)
        write_new(OUT/(phase+"_resources.json"),dict(seconds=time.monotonic()-begin,
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(),rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  program_sha256=sha(Path(__file__)),protocol_sha256=sha(ROOT/"references/glyph_planner_transfer_protocol.md"),
                  search_sha256=sha(ROOT/"scripts/search_glyph_planner.py"),device=torch.cuda.get_device_name()))
