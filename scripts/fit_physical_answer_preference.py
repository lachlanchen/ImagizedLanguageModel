"""Train the visual context reader and image writer on complete competing trajectories."""
import fcntl
from pathlib import Path
import resource
import time

import numpy as np
import torch
from torch.nn import functional as F

from ilm.visual_lm.idiom_ink import GROUPS, PicturedIdiom, visual_input
from scripts.diagnose_visual_context_gradients import load, require, sha, write_new
from scripts.fit_local_teacher_writer import DATA, GUIDED, OUT as HINTS, ROOT, load_split
from scripts.fit_pictured_idioms import guard
from scripts.score_paired_physical_answers import physical_score
from scripts.score_visual_writer_transfer import matching_pairs

OUT = ROOT / 'artifacts/physical_answer_preference_v1'


def fit(arm, initial, x, y, foil, features, bits, feedback, start, *, epochs=8, destination=None, row_weights=None):
    model = PicturedIdiom(initial['landmarks']).cuda()
    model.load_state_dict(initial)
    model.reader.eval()  # Disable dropout identically while differentiating through the reader.
    model.motor.requires_grad_(False)
    trainable = [v for v in model.parameters() if v.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=.0002, weight_decay=.01)
    rng = np.random.default_rng(720)
    history = []
    begin = time.monotonic()
    for epoch in range(epochs):
        order = rng.permutation(len(y))
        likelihood, ranking, seen = 0., 0., 0
        for pos in range(0, len(y), 192):
            if pos % 3072 == 0:
                guard(start)
            ids = order[pos:pos + 192]
            targets = y[ids]
            if arm == 'preference':
                partner = foil[ids]
                paired = partner >= 0
            with torch.autocast('cuda', dtype=torch.bfloat16):
                state = model.read(features[x[ids]])
                positive = physical_score(model, state, bits[targets], feedback[targets])
                weighted = positive if row_weights is None else positive * row_weights[ids]
                reference_loss = -weighted.mean() / (4 * GROUPS + 1)
                extra = reference_loss.new_zeros(())
                if arm == 'preference' and bool(paired.any()):
                    competing = y[partner[paired]]
                    negative = physical_score(model, state[paired], bits[competing], feedback[competing])
                    extra = F.softplus((negative - positive[paired]) / 2).sum() / len(ids)
                loss = reference_loss + .08 * extra
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1., error_if_nonfinite=True)
            opt.step()
            likelihood += reference_loss.detach().item() * len(ids)
            ranking += extra.detach().item() * len(ids)
            seen += len(ids)
        history.append(dict(rows=seen, nll=likelihood/seen, preference=ranking/seen))
        print(arm, epoch + 1, history[-1], flush=True)
    torch.cuda.synchronize()
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    require(all(torch.equal(v, initial[k]) for k,v in state.items() if k.startswith('motor.') or k == 'landmarks'),
            'Physical renderer changed')
    path = OUT / arm if destination is None else destination
    path.mkdir()
    torch.save(state, path / 'weights.pt')
    write_new(path / 'fit.json', dict(fit_seconds=time.monotonic()-begin, epochs=history,
              trained_parameters=sum(p.numel() for p in trainable), weights_sha256=sha(path/'weights.pt')))


def main():
    start = time.monotonic()
    guard(start)
    require(torch.cuda.is_available() and '4090' in torch.cuda.get_device_name(), 'Expected 4090')
    diagnostic = load(HINTS/'paired_trajectories/result.json')
    require(diagnostic['guided']['new']['all']['assignment_correct'] == 1956, 'Expected completed pair diagnostic')
    torch.set_num_threads(2)
    torch.manual_seed(720)
    torch.set_float32_matmul_precision('high')
    OUT.mkdir()
    path = GUIDED/'distilled_targets/weights.pt'
    require(sha(path) == load(path.parent/'fit.json')['weights_sha256'], 'Guided writer changed')
    expected = load(ROOT/'references/evidence/pictured_idioms_20261118/preparation.json')
    require(sha(DATA/'train.npz') == expected['splits']['train']['sha256'] and
            sha(DATA/'pixels.npy') == expected['pixels_sha256'], 'Training data changed')
    initial = torch.load(path, weights_only=True, map_location='cpu')
    frozen = PicturedIdiom(initial['landmarks']).cuda().eval().requires_grad_(False)
    frozen.load_state_dict(initial)
    train = load_split(DATA/'train.npz')
    i, j = matching_pairs(train['context'], train['target'], train['group'])
    partner = np.full(len(train['group']), -1, np.int64)
    partner[i], partner[j] = j, i
    require(len(np.unique(np.r_[i, j])) == 2*len(i) and (train['group'][i] != train['group'][j]).all() and
            (train['target'][i] != train['target'][j]).any(1).all(), 'Invalid training foils')
    np.save(OUT/'training_partner.npy', partner)
    x, y = [torch.as_tensor(train[k].astype(np.int32), device='cuda') for k in ('context', 'target')]
    ink = torch.as_tensor(np.unpackbits(np.load(DATA/'pixels.npy'), axis=1), device='cuda').bool()
    features = visual_input(ink, frozen.landmarks)
    bits = ink[:, frozen.landmarks].float()
    feedback = torch.cat([(frozen.motor(v)[:, frozen.landmarks] > 0).float() for v in bits.split(128)])
    for arm in ('reference', 'preference'):
        fit(arm, initial, x, y, torch.as_tensor(partner,device='cuda'), features, bits, feedback, start)
    guard(start)
    write_new(OUT/'execution.json', dict(seconds=time.monotonic()-start, pairs=len(i), training_rows=len(x),
              initial_sha256=sha(path), partner_sha256=sha(OUT/'training_partner.npy'),
              program_sha256=sha(Path(__file__)), device=torch.cuda.get_device_name(),
              peak_allocated_bytes=torch.cuda.max_memory_allocated(),
              host_peak_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))


if __name__ == '__main__':
    with (OUT.parent/'ilm_gpu0.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        main()
