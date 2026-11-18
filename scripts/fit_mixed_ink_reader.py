"""Fit and measure a question-conditioned source pointer and frozen ink writer."""
import fcntl
import time
from collections import defaultdict

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from torch.nn import functional as F

from ilm.visual_lm.idiom_ink import PicturedIdiom, visual_input
from ilm.visual_lm.mixed_ink_reader import MixedInkReader
from scripts.compare_passage_diversity import original, verified_split, load_initial, ROOT
from scripts.diagnose_visual_context_gradients import load, require, sha, write_new
from scripts.fit_pictured_idioms import guard, DATA
from scripts.prepare_visual_sequence_corpus import Rasterizer
from scripts.score_visual_writer_transfer import paired_interval
from scripts.train_visual_extractive_reader import gold_occurrences, load_cases, save_page

OUT = ROOT/'artifacts/mixed_ink_reader_v1'
FRESH = ROOT/'artifacts/glyph_planner_v1/fresh'
BASELINE = ROOT/'artifacts/visual_extractive_reader_v1'
PROMPTS = [('补写缺失成语','填入空缺成语','填补所缺成语'),
           ('抄写空前四字','写出空前四字','摘录空前四字'),
           ('抄写空后四字','写出空后四字','摘录空后四字')]
SEED = 4717


class PixelBank:
    def __init__(self, seed):
        self.tiles = list(map(bytes,seed))
        self.index = {v:i for i,v in enumerate(self.tiles)}
        require(len(self.index)==len(self.tiles) and not any(self.tiles[0]), 'Physical bank identity')

    def encode(self, tiles):
        result = []
        for tile in tiles:
            key = bytes(tile)
            i = self.index.get(key)
            if i is None:
                i = len(self.tiles)
                self.index[key] = i
                self.tiles.append(key)
            result.append(i)
        return np.asarray(result,np.int32)


def batches(rows, size, rng=None):
    order = np.arange(len(rows)) if rng is None else rng.permutation(len(rows))
    for block in np.array_split(order, max(1,len(order)//(size*16))):
        if rng is not None:
            block = sorted(block,key=lambda i:len(rows[i][0]))
        chunks = [block[j:j+size] for j in range(0,len(block),size)]
        if rng is not None:
            rng.shuffle(chunks)
        yield from chunks


def collate(rows, device):
    tensors = []
    for col in (0,1):
        seq = [r[col] for r in rows]
        lengths = np.array(list(map(len,seq)))
        array = np.zeros((len(seq),max(lengths)),np.int32)
        for i, values in enumerate(seq):
            array[i,:len(values)] = values
        tensors.extend(torch.as_tensor(x,device=device) for x in (array,lengths))
    p, lens, q, qlens = tensors
    return p,q,lens,qlens


def encoded_qa(cases,bank,training):
    passages, result = {}, []
    for c in cases:
        if c.group not in passages:
            passages[c.group] = bank.encode(c.passage)
        start = gold_occurrences(c.passage,c.answer)[0] if training else 0
        end = start + len(c.answer)-1 if training else 0
        result.append((passages[c.group],bank.encode(c.question),False,start,end))
    return result


def fit(net, features, rows, began, device):
    opt = torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.01)
    rng = np.random.default_rng(SEED)
    history = []
    for epoch in range(12):
        sums = np.zeros(4)
        for indices in batches(rows,32,rng):
            batch = [rows[i] for i in indices]
            p,q,lens,qlens = collate(batch,device)
            generate,a,b = torch.as_tensor([r[2:] for r in batch],device=device).T.long()
            gate,start,end,duration = net(features,p,q,lens,qlens)
            copy = ~generate.bool()
            loss = F.binary_cross_entropy_with_logits(gate,generate.float())
            if copy.any():
                stop_scores = net.end_scores(end[copy],duration[copy],a[copy,None],lens[copy])[:,0]
                loss = loss + F.cross_entropy(start[copy],a[copy]) + F.cross_entropy(stop_scores,b[copy])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(),2.)
            opt.step()
            sums += [len(batch),float(loss.detach())*len(batch),int(((gate>0)==generate).sum()),1]
        guard(began)
        history.append(dict(epoch=epoch+1,examples=int(sums[0]),updates=int(sums[3]),
                            loss=sums[1]/sums[0],routes_correct=int(sums[2])))
        print('epoch',epoch+1,history[-1],flush=True)
    return history


@torch.inference_mode()
def predict(net,features,rows):
    predicted = np.empty((len(rows),3),np.int32)
    for indices in batches(rows,64):
        p,q,lens,qlens = collate([rows[i] for i in indices],features.device)
        outputs = net(features,p,q,lens,qlens)
        predicted[indices] = torch.stack(net.decode(outputs,lens),dim=1).cpu().numpy()
    return predicted


def window(p):
    center = len(p)//2
    left, right = p[max(0,center-24):center], p[center:center+24]
    return np.r_[np.zeros(24-len(left),np.int32),left,right,np.zeros(24-len(right),np.int32)]


@torch.inference_mode()
def draw(writer,features,contexts,began):
    pictures, lengths, stops = [], [], []
    for i in range(0,len(contexts),128):
        guard(began)
        x = torch.as_tensor(contexts[i:i+128],device=features.device).long()
        ink,n,stopped = writer.generate(features[x])
        ink[np.arange(ink.shape[1])[None]>=n[:,None]] = 0
        pictures.append(ink)
        lengths.append(n)
        stops.append(stopped)
    return tuple(np.concatenate(v) for v in (pictures,lengths,stops))


def assessed_qa(net,writer,features,tiles,rows,cases,began):
    old = load(BASELINE/'rows.json')
    require(sha(BASELINE/'rows.json') == load(BASELINE/'report.json')['rows_sha256'],'Prior reader changed')
    require([r['source'] for r in old] == [c.label for c in cases], 'Changed QA order')
    groups = np.array([c.group for c in cases])
    buckets = defaultdict(list)
    for i,g in enumerate(groups):
        buckets[g].append(i)
    other = np.arange(len(cases))
    for group in buckets.values():
        other[group] = np.roll(group,-1)
    different = np.array([not np.array_equal(row[1],rows[j][1]) for row,j in zip(rows,other)])
    samples = []
    report, flags = {}, {}
    for name,queries in [('normal',[r[1] for r in rows]),('swapped',[rows[j][1] for j in other]),
                         ('blank',[np.zeros_like(r[1]) for r in rows])]:
        inputs = [(r[0],q) for r,q in zip(rows,queries)]
        prediction = predict(net,features,inputs)
        chosen = np.flatnonzero(prediction[:,0])
        generated = {}
        if len(chosen):
            ink,ns,stops = draw(writer,features,np.stack([window(rows[i][0]) for i in chosen]),began)
            generated = {i:(pic[:n],s) for i,pic,n,s in zip(chosen,ink,ns,stops)}
        outputs, hits, offsets = [], [], [0]
        for i,(_,a,b) in enumerate(prediction):
            cells,stop = generated.get(i,(tiles[rows[i][0][a:b]],True))
            outputs.append(cells)
            offsets.append(offsets[-1]+len(cells))
            hit = bool(stop and np.array_equal(cells,cases[i].answer))
            hits.append(hit)
            if name=='normal' and (i<3 or (hit and not any(s[0]=='success' for s in samples))):
                role = 'success' if hit else 'fixed'
                samples.append((role,i))
                for label,pic in [('question',cases[i].question),('actual',cells),('reference',cases[i].answer)]:
                    if len(pic):
                        save_page(OUT/f'{i}_{label}.png',pic)
        hits = np.array(hits)
        mask = different if name=='swapped' else np.ones(len(rows),bool)
        np.savez_compressed(OUT/(name+'.npz'),actions=prediction,offsets=offsets,pixels=np.concatenate(outputs),
                            stopped=[generated[i][1] if i in generated else True for i in range(len(rows))],exact=hits)
        report[name] = dict(cases=int(mask.sum()),exact=int(hits[mask].sum()),generate=int(prediction[mask,0].sum()))
        flags[name] = hits
    normal = flags['normal']
    baseline = np.array([r['predictions']['normal']['exact'] for r in old])
    long = np.array([len(c.answer)>8 for c in cases])
    report.update(baseline_exact=int(baseline.sum()),baseline_ci_pp=paired_interval(normal,baseline,groups),
                  swapped_ci_pp=paired_interval(normal[different],flags['swapped'][different],groups[different]),
                  long_cases=int(long.sum()),long_exact=int(normal[long].sum()),
                  normal_on_swappable=int(normal[different].sum()),sample_indices=samples)
    return report


def assessed_gaps(net,writer,features,tiles,split,prompts,began):
    x,y = split['context'],split['target']
    packed,n,stop = draw(writer,features,x,began)
    np.savez_compressed(OUT/'free.npz',packed=packed,length=n,stopped=stop)
    with np.load(FRESH/'released/generated.npz') as old:
        same = (packed==old['packed']).all((1,2)) & (n==old['length']) & (stop==old['stopped'])
    eligible = (x[:,20:28]!=0).all(1)
    absent = ~(y[:,:,None]==x[:,None,:]).any((1,2))
    report = dict(windows=len(x),eligible_neighbors=int(eligible.sum()),all_four_absent=int(absent.sum()),
                  frozen_replay_identical=int(same.sum()))
    for name,phrasing in [('familiar',0),('unseen',2)]:
        hits = []
        stats = {}
        for role,q in enumerate(prompts):
            pred = predict(net,features,[(p,q[phrasing]) for p in x])
            truth = tiles[(y,x[:,20:24],x[:,24:28])[role]]
            success,retained = [],[]
            for i,(gen,a,b) in enumerate(pred):
                free = packed[i,:n[i]]
                cells = free if gen else tiles[x[i,a:b]]
                stopped = stop[i] if gen else True
                success.append(bool(stopped and np.array_equal(cells,truth[i])))
                retained.append(bool(same[i] and stopped==stop[i] and np.array_equal(cells,free)))
            ok = np.array(success)
            keep = eligible if role else np.ones(len(x),bool)
            stats[('gap','left','right')[role]] = dict(exact=int(ok[keep].sum()),total=int(keep.sum()),
                       routed_generate=int(pred[keep,0].sum()),**({} if role else
                       dict(retained=int(np.sum(retained)),all_absent_exact=int(ok[absent].sum()))))
            hits.append(ok)
            np.savez_compressed(OUT/f'{name}_{role}.npz',actions=pred,exact=ok)
        stats['all_three'] = int((eligible & np.logical_and.reduce(hits)).sum())
        report[name] = stats
    return report


def main():
    began = time.monotonic()
    guard(began)
    OUT.mkdir(exist_ok=False)
    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    device = torch.device('cuda:0')
    train,test = load_cases(began)
    gaps = original()
    fresh = verified_split(FRESH)
    seed = np.load(FRESH/'pixels.npy')
    old_bank = np.load(DATA/'pixels.npy')
    require(np.array_equal(seed[:len(old_bank)],old_bank),'Changed glyph packing')
    bank = PixelBank(seed)
    train_rows = encoded_qa(train,bank,True)
    test_rows = encoded_qa(test,bank,False)
    render = Rasterizer()
    prompts = [[bank.encode([render.tile(c) for c in text]) for text in row] for row in PROMPTS]
    rng = np.random.default_rng(SEED)
    x = gaps['context']
    selected = rng.choice(np.flatnonzero((x[:,20:28]!=0).all(1)),2048,replace=False)
    for i,index in enumerate(selected):
        for role,offset in enumerate((0,20,24)):
            train_rows.append((x[index],prompts[role][i%2],role==0,offset,offset+3))
    np.save(OUT/'selected.npy',selected)
    tiles = np.frombuffer(b''.join(bank.tiles),np.uint8).reshape(-1,128)
    state = load_initial()
    writer = PicturedIdiom(state['landmarks']).to(device).eval().requires_grad_(False)
    writer.load_state_dict(state)
    features = visual_input(torch.from_numpy(np.unpackbits(tiles,axis=1)).to(device),writer.landmarks)
    net = MixedInkReader().to(device)
    t = time.monotonic()
    history = fit(net,features,train_rows,began,device)
    fit_seconds = time.monotonic()-t
    save_file({k:v.cpu().contiguous() for k,v in net.state_dict().items()},str(OUT/'reader.safetensors'))
    net.load_state_dict(load_file(str(OUT/'reader.safetensors')))
    net.eval()
    qa = assessed_qa(net,writer,features,tiles,test_rows,test,began)
    print('qa',qa,flush=True)
    gap = assessed_gaps(net,writer,features,tiles,fresh,prompts,began)
    checks = dict(qa_prior=qa['baseline_ci_pp'][0]>0, qa_question=qa['swapped_ci_pp'][0]>0)
    for kind in ('familiar','unseen'):
        checks[kind+'_retention'] = gap[kind]['gap']['retained'] >= .99*gap['windows']
        for role in ('left','right'):
            checks[kind+'_'+role] = gap[kind][role]['exact'] >= .90*gap['eligible_neighbors']
    stats = dict(natural_training=len(train),gap_training=len(selected),training=history,qa=qa,gap=gap,
              checks=checks,passed=all(checks.values()),fit_seconds=fit_seconds,
              elapsed_seconds=time.monotonic()-began,gpu=torch.cuda.get_device_name(),
              peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,
              parameters=sum(v.numel() for v in net.parameters()),writer_parameters=sum(v.numel() for v in writer.parameters()),
              output_sha256={p.name:sha(p) for p in sorted(OUT.iterdir()) if p.is_file()},
              protocol_sha256=sha(ROOT/'references/mixed_ink_reading_protocol.md'))
    guard(began)
    write_new(OUT/'report.json',stats)
    print('gap',gap,'checks',checks,flush=True)


if __name__=='__main__':
    with (ROOT/'artifacts/ilm_gpu0.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        main()
