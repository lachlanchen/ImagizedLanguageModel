"""Test the effect of additional pictured traditional Chinese question/answer evidence."""
import fcntl
from itertools import combinations
from time import monotonic

import numpy as np
import torch
from safetensors.torch import load_model, save_model

from ilm.visual_lm.ordered_passage_planner import OrderedPassageComposer
from scripts import test_official_passage_composition as official
from scripts.compare_passage_instructions import TEST as CMRC_PREFIXES, internal
from scripts.fit_mixed_ink_reader import collate
from scripts.prepare_pictured_entry_answers import Rasterizer, load, sha, write_json
from scripts.train_paired_passage import ROOT, Corpus, unpack_sequences, pad
from scripts.train_visual_extractive_reader import load_cases

DATA = ROOT / "data/traditional_passage_evidence_v1"
OUT = ROOT / "artifacts/traditional_passage_training_v1"
INITIAL = ROOT / "models/pictured_passage_zh/varied_planner.safetensors"
REG = ROOT / "references/evidence/traditional_passage_training_registration.json"
DRCD_PREFIXES = ("依次作答：", "請依次回答這兩個問題：")
SEED = 19307


class TraditionalPassages:
    def __init__(self, split):
        folder = DATA / split
        receipts = load(DATA / "preparation.json")[split]["files"]
        for name, expected in receipts.items():
            assert sha(folder/name) == expected
        self.bank = np.load(folder / "pixels.npy")
        self.passages = unpack_sequences(folder/"passages.npz")
        self.questions = unpack_sequences(folder/"questions.npz")
        self.ids = load(folder/"question_ids.json")
        with np.load(folder/"evidence.npz") as z:
            self.groups = z["group"]
            self.intervals = np.split(z["spans"],z["offsets"][1:-1])

    def for_scoring(self):
        aliases = []
        members = [[] for p in self.passages]
        for i,(g,spans) in enumerate(zip(self.groups,self.intervals,strict=True)):
            members[g].append(i)
            p = self.passages[g]
            aliases.append({self.bank[p[a:b]].tobytes() for a,b in spans})
        tiles = list(map(bytes,self.bank))
        index = {tile:i for i,tile in enumerate(tiles)}
        render = Rasterizer()
        def encode(text):
            seq = []
            for ch in text:
                ink = render.tile(ch)
                if ink not in index:
                    index[ink] = len(tiles)
                    tiles.append(ink)
                seq.append(index[ink])
            return np.asarray(seq,np.int32)
        prefixes = [encode(text) for text in DRCD_PREFIXES]
        join = encode("；")
        sep = tiles[join[0]]
        pairs = []
        for g,indices in enumerate(members):
            for a,b in combinations(indices,2):
                if np.array_equal(self.questions[a],self.questions[b]) or aliases[a] & aliases[b]:
                    continue
                forward = {x+sep+y for x in aliases[a] for y in aliases[b]}
                reverse = {y+sep+x for x in aliases[a] for y in aliases[b]}
                if forward.isdisjoint(reverse):
                    pairs.extend(((g,a,b),(g,b,a)))
        bank = np.frombuffer(b"".join(tiles),np.uint8).reshape(-1,128)
        return self.passages,self.questions,aliases,pairs,self.groups,prefixes,join,bank,{}


def training_data(cmrc, drcd):
    combined = np.concatenate((cmrc.bank,drcd.bank))
    tiles, remap = np.unique(combined,axis=0,return_inverse=True)
    first,second = np.split(remap.astype(np.int32),[len(cmrc.bank)])
    cmrc_ps = [first[p] for p in cmrc.passages]
    original = []
    for q,a,g in zip(cmrc.questions,cmrc.answers,cmrc.groups,strict=True):
        p = cmrc.passages[g]
        spans = [(l,l+len(a)) for l in np.flatnonzero(p==a[0]) if np.array_equal(p[l:l+len(a)],a)]
        assert spans
        original.append((cmrc_ps[g],first[q],np.asarray(spans,np.int32)))
    other_ps = [second[p] for p in drcd.passages]
    other = [(other_ps[g],second[q],s) for g,q,s in zip(drcd.groups,drcd.questions,drcd.intervals,strict=True)]
    return tiles,original,other


def restore():
    net = OrderedPassageComposer()
    load_model(net,str(INITIAL))
    net.planner.requires_grad_(False)
    return net.cuda()


def replay_ids(rng, n, needed):
    return np.concatenate([rng.permutation(n) for _ in range((needed+n-1)//n)])[:needed]


def fit(mixed, tiles, original, other):
    net = restore()
    features = net.features(tiles)
    opt = torch.optim.AdamW(net.reader.parameters(),lr=.00020,weight_decay=.01)
    rng = np.random.default_rng(SEED)
    n = len(other)
    history = []
    for epoch in range(4):
        began = monotonic()
        # The shared half and the substitution schedule are identical in both fits.
        common = replay_ids(rng,len(original),n)
        replay = replay_ids(rng,len(original),n)
        # Match replay replacements by passage-length rank to reduce padding differences.
        sorted_other = np.argsort([len(row[0]) for row in other],kind="stable")
        sorted_replay = sorted(replay,key=lambda i:len(original[i][0]))
        replay[sorted_other] = sorted_replay
        rows = [original[i] for i in common] + (other if mixed else [original[i] for i in replay])
        order = rng.permutation(2*n)
        total = 0.
        positions = padded = steps = 0
        for block in np.array_split(order,(len(order)+511)//512):
            def reference_length(i):
                return len(original[common[i]][0]) if i<n else len(other[i-n][0])
            ordered = sorted(block,key=reference_length)
            chunks = [ordered[i:i+32] for i in range(0,len(ordered),32)]
            for j in rng.permutation(len(chunks)):
                batch = [rows[i] for i in chunks[j]]
                p,q,lengths,qlens = collate(batch,features.device)
                a,num_labels = pad([row[2][:,0] for row in batch],features.device)
                b,_ = pad([row[2][:,1]-1 for row in batch],features.device)
                _,start,end,duration = net.reader(features,p,q,lengths,qlens)
                endings = net.reader.end_scores(end,duration,a,lengths).log_softmax(-1)
                scores = start.log_softmax(-1).gather(1,a.long())+endings.gather(2,b.long().unsqueeze(-1)).squeeze(-1)
                valid = torch.arange(a.shape[1],device=a.device)<num_labels[:,None]
                loss = -scores.masked_fill(~valid,-torch.inf).logsumexp(1).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.reader.parameters(),2.)
                opt.step()
                total += float(loss.detach())*len(batch)
                positions += sum(len(row[0]) for row in batch)
                padded += p.numel()
                steps += 1
        history.append(dict(epoch=epoch+1,examples=len(order),steps=steps,loss=total/len(order),
                            source_cells=positions,padded_source_cells=padded,seconds=monotonic()-began))
        print("mixed" if mixed else "replay",history[-1],flush=True)
    return net.eval(),history


def score(net,arm,cmrc_dev,cmrc_test,drcd_test):
    score = {}
    for name,wrappers,data in (("cmrc",CMRC_PREFIXES,cmrc_test),("drcd",DRCD_PREFIXES,drcd_test)):
        official.WRAPPERS = wrappers
        official.OUT = OUT
        score[name] = official.evaluate_checkpoint(net,arm+"_"+name,data)
    images,ends = internal(net,cmrc_dev)
    n = len(cmrc_dev.pairs)
    path = OUT/(arm+"_internal.npz")
    np.savez_compressed(path,pixels=np.frombuffer(b"".join(images),np.uint8).reshape(-1,128),
                        offsets=np.cumsum([0]+[len(s)//128 for s in images]),stopped=ends)
    ok=np.array([s and im==cmrc_dev.target(i) for i,(s,im) in enumerate(zip(ends,images,strict=True))])
    score["internal"] = dict(singles=int(ok[n:].sum()),ordered=int(ok[:n].sum()),
                             both=int(ok[:n].reshape(-1,2).all(1).sum()),archive_sha256=sha(path))
    return score


def run():
    for name,digest in load(REG)["files"].items():
        assert sha(ROOT/name)==digest,name
    OUT.mkdir(exist_ok=False)
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.86)
    began = monotonic()
    train_cases,dev_cases = load_cases(began)
    cmrc = Corpus("train",train_cases)
    dev = Corpus("development",dev_cases)
    drcd = TraditionalPassages("train")
    drcd_test = TraditionalPassages("development").for_scoring()
    tiles,orig,other = training_data(cmrc,drcd)
    official.WRAPPERS = CMRC_PREFIXES
    cmrc_test = official.prepare()
    summary = dict(cmrc_questions=len(cmrc_test[1]),cmrc_pairs=len(cmrc_test[3]),
                   drcd_questions=len(drcd_test[1]),drcd_pairs=len(drcd_test[3]),
                   train_cmrc=len(orig),train_drcd=len(other),prep_seconds=monotonic()-began)
    for arm in ("released","replay","mixed"):
        torch.cuda.reset_peak_memory_stats()
        if arm=="released":
            model,history = restore().eval(),[]
            weights = INITIAL
        else:
            model,history = fit(arm=="mixed",tiles,orig,other)
            weights = OUT/(arm+".safetensors")
            save_model(model,str(weights))
        scored = score(model,arm,dev,cmrc_test,drcd_test)
        summary[arm]=dict(scores=scored,history=history,weights_sha256=sha(weights),
                          peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
        write_json(OUT/(arm+"_result.json"),summary[arm])
        print(arm,"DRCD singles",scored["drcd"]["single_exact"],"CMRC singles",scored["cmrc"]["single_exact"],flush=True)
        del model
    summary["seconds"]=monotonic()-began
    write_json(OUT/"result.json",summary)


if __name__=="__main__":
    with (ROOT/"artifacts/traditional_passage_training.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run()
