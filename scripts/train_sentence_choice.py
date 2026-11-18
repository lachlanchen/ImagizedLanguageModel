"""Fit matched visual likelihood and sentence-assignment objectives on real Chinese passages."""
import copy
import fcntl
import hashlib
import json
import resource
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file
from torch.nn import functional as F

from ilm.visual_lm.conditional_ink import ConditionalInk
from ilm.visual_lm.idiom_ink import nibbles, visual_input
from ilm.visual_lm.local_ink import LocalInk, PhraseInk, signatures
from ilm.visual_lm.sentence_choice import SentenceBoundary
from scripts.evaluate_conditional_ink import DIRECTORY, ROOT
from scripts.infer_visual_phrase import MODELS
from scripts.prepare_sentence_choice import WORK
from scripts.train_local_ink import OUT, checkpoint, fields, load, require, sha, tensors, tiles, write_new

DEST = WORK/'fit'
INTERACTION = MODELS/'conditional_ink.safetensors'
ARMS = ('likelihood', 'assignment')
SCALES = (0., .25, .5, 1.)
SEED = 5819


def resources(started):
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
    free = next(line for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:'))
    require(rss < 6 and torch.cuda.max_memory_allocated() < 8 * 2**30, 'Single-GPU fit memory ceiling')
    require(int(free.split()[1]) >= 24 * 1024**2 and time.monotonic()-started < 2100, 'Host reserve or phase time')


class PairedObservations:
    """Cache frozen states for both source assignments; targets only enter their causal history."""
    def __init__(self, net, locations):
        self.net = net
        lengths = locations[:, 3] - locations[:, 2]
        require(np.array_equal(lengths[::2], lengths[1::2]) and lengths.min() >= 1, 'Unmatched answer lengths')
        self.lengths = lengths
        self.offsets = np.r_[0, np.cumsum(lengths)]
        ns, cells = len(lengths), int(self.offsets[-1])
        def empty(*shape, dtype=torch.float32):
            return torch.empty(shape, device='cuda', dtype=dtype)
        self.boundary = empty(ns, 128)
        self.initial = empty(ns, 128)
        self.history = empty(cells, 2, 2, 128)  # authentic/exchanged, before/after target
        self.pattern_bias = empty(cells, 12, 16)
        self.end_bias = empty(cells)
        self.target = empty(cells, 48, dtype=torch.bool)
        self.final = empty(cells)

    @torch.no_grad()
    def prepare(self, rows, locations, bank, interaction, stream, started):
        net = self.net
        ink = torch.as_tensor(np.unpackbits(bank, axis=1), device='cuda').bool()
        bits = ink[:, net.landmarks]
        features = visual_input(ink, net.landmarks)
        question = features[load(OUT/'preparation.json')['question_tiles'][0][0]][None]
        sig = signatures(bits.cpu().numpy())
        for start in range(0, len(locations), 96):
            positions = locations[start:start+96]
            # Every target and prefix appears twice, paired with each classical sentence.
            block, cursor = [], []
            for n,(i,c,j,k) in enumerate(positions):
                for donor in (n, n^1):
                    source, boundary = positions[donor, :2]
                    block.append(dict(x=rows[source]['x'], y=rows[i]['y'][:k]))
                    cursor.append(boundary)
            mem, initial, hist, y, _ = tensors(block, features, bits, net, question, False)
            axis = torch.arange(len(block), device='cuda')
            at = net.boundary(mem, axis, torch.as_tensor(cursor, device='cuda').long())
            end = start+len(positions)
            self.boundary[start:end] = at[::2]
            self.initial[start:end] = initial[::2]
            steps = [(2*n, c, t, k) for n,(_,c,j,k) in enumerate(positions) for t in range(j,k)]
            b, _, t, k = torch.as_tensor(steps, device='cuda').long().T
            raw = bits[y[b,t]]
            window = slice(self.offsets[start], self.offsets[end])
            self.target[window] = raw
            self.final[window] = (t+1 == k).float()
            for donor in (0,1):
                self.history[window,donor,0] = hist[b+donor,t]
                self.history[window,donor,1] = hist[b+donor,t+1]
            context = fields(block, sig, steps)
            local = fields(block, sig, [(b+1,cursor[b+1],t,k) for b,_,t,k in steps])
            require(np.array_equal(context, local), 'Crossed assignment changes a local observation')
            onset = np.array([t == positions[b//2,2] for b,c,t,k in steps])
            shifts = torch.empty((len(steps),12,16),device='cuda')
            for mask, model in ((onset, stream.onset),(~onset, stream.later)):
                if mask.any():
                    shifts[torch.as_tensor(mask,device='cuda')] = torch.as_tensor(model.bias(context[mask]),device='cuda').float()
            shifts[:,1:] += interaction.teacher(torch.as_tensor(context.astype('int64'),device='cuda'),
                                                torch.as_tensor(onset,device='cuda'), nibbles(raw))
            self.pattern_bias[window] = shifts
            self.end_bias[window] = torch.as_tensor(stream.update(context, raw.cpu().numpy()),device='cuda')
            if start % 9600 == 0:
                resources(started)
                print('cached sites',start,'/',len(locations),flush=True)

    def scores(self, model, pairs, crossed):
        # Reduce each complete phrase before taking the logistic assignment loss.
        sites = np.stack((2*pairs,2*pairs+1),1).reshape(-1)
        lens = self.lengths[sites]
        offsets = np.cumsum(lens) - lens
        group = np.repeat(np.arange(len(sites)), lens)
        positions = np.repeat(self.offsets[sites]-offsets, lens) + np.arange(lens.sum())
        axis, group, pos = (torch.as_tensor(v,device='cuda') for v in (sites,group,positions))
        ref = self.target[pos]
        codes = nibbles(ref)[:,:,None]
        at = model(self.boundary[axis],self.initial[axis])
        outputs = []
        for donor in range(2 if crossed else 1):
            boundary = at[group ^ donor]
            state = (boundary + self.history[pos,donor,0]).tanh()
            logits = self.net.draw.teacher(state, ref) + self.pattern_bias[pos]
            ll = logits.log_softmax(-1).gather(-1,codes).sum((1,2))
            end = self.net.end((boundary+self.history[pos,donor,1]).tanh())[:,0] + self.end_bias[pos]
            ll -= F.binary_cross_entropy_with_logits(end, self.final[pos], reduction='none')
            outputs.append(ll.new_zeros(len(sites)).index_add(0,group,ll).reshape(-1,2))
        return outputs, torch.as_tensor(lens[::2].copy(),device='cuda')


def objective(scores, lengths):
    own = scores[0].sum(1)
    normalized = -own / (2 * lengths)
    if len(scores) == 2:
        margin = own - scores[1].sum(1)
        normalized = normalized + F.softplus(-margin / (2 * lengths))
    return normalized.mean()


def run():
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('high')
    torch.cuda.reset_peak_memory_stats()
    began = time.monotonic()
    resources(began)
    spec = load(WORK/'preparation.json')
    require(sha(WORK/'rows.jsonl')==spec['rows_sha256'] and sha(WORK/'sites.npz')==spec['pairs_sha256'], 'Training data changed')
    DEST.mkdir()
    with np.load(WORK/'sites.npz') as data:
        locations = data['locations'][data['pairs'].ravel()]
        held = data['calibration']
    with (WORK/'rows.jsonl').open() as file:
        rows = [{k:r[k] for k in ('x','y')} for r in map(json.loads,file)]
    net = checkpoint('source').requires_grad_(False)
    model = ConditionalInk.restore(INTERACTION,sha(DIRECTORY/'manifest.json'),'cuda').requires_grad_(False)
    stream = PhraseInk(*(LocalInk.load(DIRECTORY/(n+'.npz')) for n in ('onset','later','end')))
    cache = PairedObservations(net, locations)
    cache.prepare(rows, locations, tiles(), model, stream, began)
    del rows, model, stream
    prep_seconds = time.monotonic()-began
    base = SentenceBoundary().cuda()
    writers = {arm:copy.deepcopy(base) for arm in ARMS}
    opts = {arm:torch.optim.AdamW(writer.parameters(),lr=.0005,weight_decay=.01) for arm,writer in writers.items()}
    rng = np.random.default_rng(SEED)
    schedule = hashlib.sha256()
    train = np.flatnonzero(~held)
    calibration = np.flatnonzero(held)
    history = {arm:[] for arm in ARMS}
    cost = dict.fromkeys(ARMS,0.)
    for epoch in range(3):
        order = rng.permutation(train)
        schedule.update(order.astype('<i4').tobytes())
        sums = dict.fromkeys(ARMS,0.)
        for start in range(0,len(order),256):
            batch = order[start:start+256]
            for arm,writer in writers.items():
                t = time.monotonic()
                scored, lens = cache.scores(writer,batch,arm=='assignment')
                loss = objective(scored,lens)
                opts[arm].zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(writer.parameters(),2.,error_if_nonfinite=True)
                opts[arm].step()
                sums[arm] += loss.detach().item()*len(batch)
                cost[arm] += time.monotonic()-t
            if start%(256*100)==0:
                resources(began)
        for arm in ARMS:
            history[arm].append(sums[arm]/len(train))
        print('epoch',epoch+1,history,flush=True)
    result = {}
    with torch.no_grad():
        for arm,writer in writers.items():
            grid, wins = [], []
            for scale in SCALES:
                writer.scale.fill_(scale)
                total = correct = 0
                for idx in range(0,len(calibration),256):
                    scores, lengths = cache.scores(writer,calibration[idx:idx+256],True)
                    total += (-scores[0].sum(1)/(2*lengths)).double().sum().item()
                    correct += (scores[0].sum(1)>scores[1].sum(1)).sum().item()
                grid.append(total/len(calibration))
                wins.append(correct)
            writer.scale.fill_(SCALES[int(np.argmin(grid))])
            path = DEST/(arm+'.safetensors')
            save_file({k:v.detach().cpu() for k,v in writer.state_dict().items()},str(path),
                      metadata=dict(format='ilm-sentence-boundary-v1',interaction_sha256=sha(INTERACTION)))
            result[arm]=dict(selected_scale=writer.scale.item(),calibration_nats=grid,calibration_assignment_wins=wins,
                             epoch_loss=history[arm],optimizer_seconds=cost[arm],sha256=sha(path))
    resources(began)
    write_new(DEST/'fit.json',dict(arms=result,scales=SCALES,schedule_sha256=schedule.hexdigest(),
        epochs=3,training_pairs=len(train),calibration_pairs=len(calibration),parameters=sum(p.numel() for p in base.parameters()),
        preparation_seconds=prep_seconds,seconds=time.monotonic()-began,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
        peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,interaction_sha256=sha(INTERACTION),
        data=spec,source_state_sites=len(locations),teacher_cells=int(cache.offsets[-1])))
    print(result,flush=True)


if __name__ == '__main__':
    with (ROOT/'artifacts/ilm_gpu0.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run()
