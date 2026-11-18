"""Correct the training-only log odds for marginal sampling of complete glyphs."""
import fcntl
import json
import time

import numpy as np
import torch
from safetensors.torch import load_file

from ilm.visual_lm.visual_edit_writer import INSERT, REPLACE
from scripts.compare_visual_edit_contrast import PARENT, WORK, fit, restore, score
from scripts.evaluate_visual_edit_joint import teacher
from scripts.evaluate_visual_edit_translation import ci, evaluate, local_program
from scripts.train_visual_edit_translation import ROOT, examples, guard, inputs, load, require, sha, write_new


class DrawFrequency:
    def __init__(self, train, tiles, landmarks):
        counts = np.zeros(len(tiles), dtype=np.int64)
        for row in train:
            glyphs = [v for a,v in zip(row['a'],row['o']) if a in (INSERT,REPLACE)]
            if glyphs:
                np.add.at(counts,glyphs,1)
        active = counts > 0
        bits = np.unpackbits(tiles[active],axis=1)[:,landmarks]
        powers = 1 << np.arange(len(landmarks),dtype=np.int64)
        keys, inverse = np.unique(bits @ powers,return_inverse=True)
        totals = np.zeros(len(keys),np.int64)
        np.add.at(totals,inverse,counts[active])
        self.powers = torch.from_numpy(powers).cuda()
        self.keys = torch.from_numpy(keys).cuda()
        self.values = torch.tensor(np.log(totals/totals.sum()),dtype=torch.float32,device='cuda')
        self.observations = int(totals.sum())
        self.distinct = len(keys)

    def __call__(self, bits):
        keys = (bits.long()*self.powers).sum(-1)
        index = torch.searchsorted(self.keys,keys)
        require(torch.equal(self.keys[index], keys),'Target training code missing from drawing marginal')
        return self.values[index]


def roster(directory):
    prep = load(directory/'preparation.json')
    require(sha(directory/'pixels.npy') == prep['pixels_sha256'] and sha(directory/'fresh.jsonl') == prep['fresh_sha256'],
            'Roster inputs changed')
    return np.load(directory/'pixels.npy'), [json.loads(l) for l in (directory/'fresh.jsonl').read_text().splitlines()]


def main():
    begin = time.monotonic()
    guard(begin)
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('high')
    require('4090' in torch.cuda.get_device_name() and sha(PARENT)==load(WORK/'contrast/fit.json')['parent'],
            'Original model or GPU changed')
    train = inputs('train')
    tiles, fresh = roster(WORK)
    reserve_tiles, reserve = roster(WORK/'confirmation_roster')
    landmarks = load_file(str(PARENT))['landmarks'].numpy()
    log_frequency = DrawFrequency(train,tiles,landmarks)
    columns, lengths, source_lengths = examples(train)
    arm = 'noise_corrected'
    fit(arm,tiles,columns,lengths,source_lengths,log_frequency=log_frequency)
    require(load(WORK/arm/'fit.json')['schedule']==load(WORK/'likelihood/fit.json')['schedule'], 'Different schedule')
    del columns
    program = local_program(train)
    score(arm,tiles,train,program,fresh)
    for name in ('likelihood', arm):
        net = restore(WORK/name/'editor.safetensors').eval()
        folder = WORK/name
        write_new(folder/'reserved_teacher.json',teacher(net,reserve_tiles,records=reserve))
        evaluate('reserved',net,reserve_tiles,program,train,time.monotonic(),folder/'reserved',records=reserve)
    summary = dict(draw_exposures=log_frequency.observations, distinct_codes=log_frequency.distinct, splits={})
    for split in ('heldout','fresh','reserved'):
        rows = load(WORK/arm/split/'rows.json')
        controls = load(WORK/'likelihood'/split/'rows.json')
        for r,c in zip(rows,controls,strict=True):
            require(r['index']==c['index'] and r['group']==c['group'] and r['local']==c['local'],'Row mismatch')
            r['matched']=c['translation']
        interval = ci(rows,'matched')
        report = load(WORK/arm/split/'report.json')
        summary['splits'][split]=dict(matched_ci_pp=interval,local_ci_pp=report['improvement_ci_pp']['local'],
                                      qualified=bool(report['pass'] and interval[0]>0))
    write_new(WORK/'correction.json',summary)
    print(summary,flush=True)


if __name__ == '__main__':
    with (ROOT/'artifacts/ilm_gpu0.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        main()
