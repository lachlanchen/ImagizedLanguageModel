"""Build the paper table and select original free-writing images by fixed source order."""
from pathlib import Path
import shutil

import numpy as np

from scripts.check_pictured_idioms import raw_cases
from scripts.confirm_joint_reference_writer import OUT as CONFIRM
from scripts.diagnose_visual_context_gradients import load, require, sha, write_new
from scripts.fit_local_teacher_writer import DATA, GUIDED, OUT as HINTS, ROOT, load_split
from scripts.fit_physical_answer_preference import OUT as JOINT
from scripts.prepare_visual_sequence_corpus import Rasterizer
from scripts.train_visual_extractive_reader import save_page

DEST = ROOT/'references/evidence/expression_preference_20261118'


def examples():
    src = load_split(DATA/'validation.npz')
    px = np.load(DATA/'pixels.npy')
    truth = px[src['target']]
    states = [load_split(k/'generated.npz') for k in
              [GUIDED/'distilled_targets', JOINT/'evaluation/validation/reference']]
    hits = [(s['length'] == 4) & s['stopped'] & (s['packed'][:,:4] == truth).all((1,2)) for s in states]
    old, new = hits
    raw = raw_cases()
    raster = Rasterizer()
    possible = sorted({c for _,_,word in raw.values() for c in word})
    known = {}
    for c in possible:
        try:
            tile = raster.tile(c)
        except ValueError:
            continue
        known.setdefault(tile,[]).append(c)
    samples = {}
    for label, mask in [('gain',new & ~old),('loss', ~new & old)]:
        i = int(np.flatnonzero(mask)[0])
        before,after,word = raw[src['tag'][i]]
        out = states[1]['packed'][i,:states[1]['length'][i]]
        inputs = px[src['context'][i]]
        for name, images in [('before',inputs[12:24]),('after',inputs[24:36]),('output',out),('reference',truth[i])]:
            save_page(DEST/f'{label}_{name}.png', images)
        samples[label] = dict(i=i,before=before,after=after,reference=word,
            output_interpretations=[known.get(tile.tobytes(),[]) for tile in out],
            answer_image_sha256=sha(DEST/f'{label}_output.png'))
    return samples


def main():
    DEST.mkdir()
    sources = dict(hints=HINTS/'evaluation/result.json', local_fit=HINTS/'execution.json', local_hints=HINTS/'teacher.json',
        paired_diagnosis=HINTS/'paired_trajectories/result.json', joint=JOINT/'evaluation/result.json', joint_fit=JOINT/'execution.json',
        joint_pairs=JOINT/'paired_trajectories/result.json', confirmation=CONFIRM/'result.json',
        confirmation_sources=CONFIRM/'preparation.json', holdout_36=HINTS/'confirmation/preparation.json',
        holdout_40=JOINT/'confirmation/preparation.json')
    for arm, root in [('shuffled',HINTS),('aligned',HINTS),('reference',JOINT),('preference',JOINT)]:
        sources[arm+'_fit'] = root/arm/'fit.json'
    for name,path in sources.items():
        shutil.copyfile(path, DEST/(name+'.json'))
    hints, joint, last = (load(sources[k]) for k in ['hints','joint','confirmation'])
    require(hints['new']['n'] == joint['preceding']['n'], 'Earlier confirmation changed')
    sizes = [joint['validation']['n'], joint['preceding']['n'], joint['new']['n'], last['n']]
    table = [['Training', *[f'{n:,} gaps' for n in sizes]]]
    def add(name, key, family):
        scores = [family[k]['outcomes'][key]['exact'] for k in
                  (('validation','new') if family is hints else ('validation','preceding','new'))]
        if key in last['outcomes']:
            scores.append(last['outcomes'][key]['exact'])
        table.append([name,*map(str,scores),*(['–']*(4-len(scores)))])
    add('Published writer','guided',joint)
    add('Shuffled local hints','shuffled',hints)
    add('Aligned local hints','aligned',hints)
    add('Joint reference','reference',joint)
    add('Joint preference','preference',joint)
    table.append(['Local counts',*[str(s['local_exact']) for s in
                  [joint['validation'], joint['preceding'], joint['new'], last]]])
    write_new(DEST/'summary.json', dict(table=table,samples=examples()))


if __name__ == '__main__':
    main()
