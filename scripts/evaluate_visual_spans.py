"""Measure autonomous translations, inserted phrases and role swaps from trained image editors."""
import fcntl
import hashlib
import itertools
import time
from collections import Counter, defaultdict
from zipfile import ZipFile

import numpy as np
import torch
from rapidfuzz.distance import Levenshtein

from ilm.visual_lm.idiom_ink import visual_input
from ilm.visual_lm.visual_edit_writer import COPY, DROP, INSERT, REPLACE
from scripts.audit_edit_contrast_streams import recount
from scripts.compare_edit_diversity import local_control
from scripts.compare_visual_edit_contrast import PARENT, restore as original_editor
from scripts.evaluate_visual_edit_translation import ci, evaluate, streams
from scripts.prepare_edit_diversity import COHORTS, records
from scripts.train_visual_spans import ARMS, OUT, ROOT, WORK, frozen, guard, load, require, restore, sha, tensor_pad, tiles, write_new


def insert_runs(actions):
    cursor = j = 0
    for a, group in itertools.groupby(actions):
        length = sum(1 for _ in group)
        if a==INSERT:
            yield cursor, j, j+length
        if a in (COPY,REPLACE,DROP):
            cursor += length
        if a in (COPY,REPLACE,INSERT):
            j += length


@torch.inference_mode()
def inserted(net, bank, records_, archive, mapping, *, supplied=True):
    ink = torch.as_tensor(np.unpackbits(bank,axis=1),device='cuda').bool()
    q = load(OUT/'preparation.json')['question_tiles'][0][0]
    q = visual_input(ink[q],net.landmarks)[None]
    features = visual_input(ink,net.landmarks)
    buckets = defaultdict(list)
    for i,r in enumerate(records_):
        if r['x']:
            buckets[len(r['x'])].append((i,r))
    counts = Counter()
    with ZipFile(archive) as z:
        for bucket in buckets.values():
            for start in range(0,len(bucket),32):
                batch = bucket[start:start+32]
                x = torch.tensor([r['x'] for _,r in batch],device='cuda')
                y,_ = tensor_pad([r['y'] for _,r in batch],x.device)
                if supplied:
                    mem,hist = net.prefix_states(features[x],ink[y][:,:,net.landmarks],q.expand(len(x),-1,-1),
                                                 torch.full((len(x),),q.shape[1],device=x.device))
                indices,spans = [], []
                for b,(i,r) in enumerate(batch):
                    raw = np.frombuffer(z.read(f'translation/{i}.bin'),np.uint8).reshape(-1,128)
                    ids = [mapping.get(bytes(a),-1) for a in raw]
                    equal = [(op.src_start,op.src_end) for op in Levenshtein.opcodes(r['y'],ids) if op.tag=='equal']
                    for cursor,j,end in insert_runs(r['a']):
                        length = end-j
                        counts.update(spans=1,glyphs=length,
                                      autonomous_exact=sum(l<=j and end<=h for l,h in equal))
                        indices.extend((b,cursor,pos) for pos in range(j,end))
                        spans.append(length)
                if supplied and indices:
                    b,c,j = torch.tensor(indices,device=x.device).T
                    state = (mem[b,c]+hist[b,j]).tanh()
                    code = net.draw.choose(state)
                    output = net.motor(code)>0
                    output[:,net.landmarks]=code.bool()
                    correct = (output==ink[y[b,j]]).all(-1)
                    chose_insert = net.actions(state,c==x.shape[1]).argmax(-1)==INSERT
                    counts.update(supplied_glyph_exact=int(correct.sum()),
                                  supplied_span_exact=sum(bool(s.all()) for s in correct.split(spans)),
                                  supplied_action_and_span_exact=sum(bool(s.all()) for s in (correct&chose_insert).split(spans)))
    return dict(counts)


@torch.inference_mode()
def roles(net, bank, start):
    cases = load(WORK/'roles.json')
    require(sha(WORK/'roles.json')==load(WORK/'rosters.json')['roles_sha256'],'Role roster changed')
    ink = torch.as_tensor(np.unpackbits(bank,axis=1),device='cuda').bool()
    question = load(OUT/'preparation.json')['question_tiles'][0][0]
    results = streams(net,ink,[r['x'] for r in cases],question,start)
    mapping = {bytes(v):i for i,v in enumerate(bank)}
    features = visual_input(ink,net.landmarks)
    rows = []
    for r,(pics,stopped,_) in zip(cases,results,strict=True):
        ids = [mapping.get(bytes(v),-1) for v in pics]
        x = torch.tensor([r['x'],r['x']],device='cuda')
        target,lengths = tensor_pad([r['y'],r['other']],x.device)
        q = torch.tensor([question,question],device='cuda')
        action,glyph = net.score_grid(features[x],ink[target][:,:,net.landmarks],features[q],
                                      torch.full_like(lengths,len(question)))
        score = net.marginal(action,glyph,ink[x][:,:,net.landmarks],ink[target][:,:,net.landmarks],lengths)
        rows.append(dict(source=r['source'], seen_source=r['seen_source'], reference=r['target'], foil=r['foil'],
                         true_logp=float(score[0]),foil_logp=float(score[1]),stopped=stopped,
                         correct_distance=Levenshtein.distance(ids,r['y']),foil_distance=Levenshtein.distance(ids,r['other']),
                         exact=bool(stopped and ids==r['y']),pixels_sha256=hashlib.sha256(pics.tobytes()).hexdigest()))
    paired = sum(a['true_logp']+b['true_logp']>a['foil_logp']+b['foil_logp'] for a,b in zip(rows[::2],rows[1::2]))
    return dict(rows=rows,correct_assignment=paired,both_exact=sum(a['exact'] and b['exact'] for a,b in zip(rows[::2],rows[1::2])))


def run():
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('high')
    bank = tiles()
    mapping = {bytes(v):i for i,v in enumerate(bank)}
    cases = {name:list(frozen(name)) for name in COHORTS}
    require(not {r['g'] for r in cases[COHORTS[0]]}&{r['g'] for r in cases[COHORTS[1]]},'Chapter overlap')
    original = OUT/'train.jsonl'
    require(sha(original)==load(OUT/'preparation.json')['partitions']['train']['sha256'],'Parent data changed')
    program,targets = local_control(itertools.chain(records(original),frozen('train')),cases)
    for name in (*ARMS,'parent'):
        start = time.monotonic()
        guard(start)
        folder = WORK/name
        if name=='parent':
            net=original_editor(PARENT).eval()
            folder.mkdir()
        else:
            path=folder/'editor.safetensors'
            require(sha(path)==load(folder/'fit.json')['checkpoint_sha256'],'Fit changed')
            net=restore(path).eval()
            write_new(folder/'roles.json',roles(net,bank,start))
        for split,rows in cases.items():
            out = folder/split
            evaluate(split,net,bank,program,targets,start,out,records=rows)
            write_new(folder/(split+'_recount.json'),recount(out,rows,bank,mapping))
            write_new(folder/(split+'_spans.json'),inserted(net,bank,rows,out/'streams.zip',mapping,supplied=name!='parent'))
        guard(start)
    result = {}
    require(load(WORK/'fixed/fit.json')['schedule_sha256']==load(WORK/'marginal/fit.json')['schedule_sha256'],
            'Training schedules differ')
    for split in COHORTS:
        rows=load(WORK/'marginal'/split/'rows.json')
        for r,c in zip(rows,load(WORK/'fixed'/split/'rows.json'),strict=True):
            require(r['index']==c['index'] and r['local']==c['local'],'Unmatched comparison')
            r['fixed']=c['translation']
        interval=ci(rows,'fixed')
        spans=[load(WORK/a/(split+'_spans.json')) for a in ARMS]
        require(spans[0]['spans']==spans[1]['spans'],'Different aligned spans')
        result[split]=dict(improvement_over_fixed_pp=interval,passed=bool(interval[0]>0 and
            load(WORK/'marginal'/split/'report.json')['pass'] and spans[1]['autonomous_exact']>=spans[0]['autonomous_exact']
            and spans[1]['supplied_span_exact']>=spans[0]['supplied_span_exact']))
    write_new(WORK/'comparison.json',result)
    print(result,flush=True)


if __name__=='__main__':
    with (ROOT/'artifacts/ilm_gpu0.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run()
