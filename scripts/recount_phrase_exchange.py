"""Recover time-aligned historical exchange scores and compare the original physical outputs."""
import fcntl
import json

import torch

from scripts.evaluate_sentence_choice import (DEST, DIRECTORY, INTERACTION, ROOT, SPLITS, WORK, ConditionalInk,
    LocalInk, PhraseInk, SentenceBoundary, checkpoint, exchange, load, require, roster, sha, write_new)
from scripts.train_first_emission import WORK as CURRENT


def run():
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('high')
    net = checkpoint('source')
    ink = ConditionalInk.restore(INTERACTION, sha(DIRECTORY/'manifest.json'), 'cuda')
    tables = PhraseInk(*(LocalInk.load(DIRECTORY/(key+'.npz')) for key in ('onset','later','end')))
    results = {}
    for split in SPLITS:
        rows, bank = roster(split)
        folder = WORK/'evaluation'/split
        results[split] = {}
        for arm in ('frozen','likelihood','assignment'):
            path = folder/(arm+'.npz')
            require(sha(path)==load(folder/'report.json')['arms'][arm]['pixels_sha256'], 'Changed original images')
            residual = None if arm=='frozen' else SentenceBoundary.restore(DEST/(arm+'.safetensors'),sha(INTERACTION),'cuda')
            old = load(folder/(arm+'_exchange.json'))
            new = exchange(rows,bank,net,tables,ink,path,residual)
            for field in ('sites','exact_authentic_exchanged','both_exact','changed_batch_sites','output_changed','first_changed'):
                require(old[field]==json.loads(json.dumps(new[field])), f'{split}/{arm}/{field} physical comparison changed')
            results[split][arm] = {key:new[key] for key in ('pairs','assignment_wins','ties','losses','exact_authentic_exchanged')}
            results[split][arm]['original_wins'] = old['assignment_wins']
        print(split,results[split],flush=True)
    write_new(CURRENT/'corrected_exchange.json', results)


if __name__ == '__main__':
    with (ROOT/'artifacts/ilm_gpu0.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run()
