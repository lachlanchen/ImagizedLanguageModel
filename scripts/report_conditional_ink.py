"""Recount complete pixel streams, export figure panels and verify the CPU writer."""
import json
import shutil

import numpy as np
import torch

from ilm.visual_lm.conditional_ink import ConditionalInk
from ilm.visual_lm.local_ink import LocalInk, PhraseInk
from scripts.evaluate_conditional_ink import COHORTS, DIRECTORY, WORK, control, inputs, sites, training_rows
from scripts.infer_visual_phrase import MODELS, restore_public
from scripts.report_visual_phrases import png, sites_independent
from scripts.train_local_ink import ROOT, load, require, sha, write_new

EVIDENCE = ROOT/'references/evidence/conditional_ink_20261118'


@torch.inference_mode()
def report():
    EVIDENCE.mkdir()
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('high')
    train = training_rows()
    seen = {tuple(train[i]['y'][j:k]) for i,c,j,k in sites(train)}
    del train
    fit = load(WORK/'fit.json')
    require(sha(WORK/'transitions.safetensors') == fit['checkpoint_sha256'], 'Fit changed')
    model = ConditionalInk.restore(WORK/'transitions.safetensors',sha(DIRECTORY/'manifest.json'),'cpu')
    net,q = restore_public(MODELS/'editor.safetensors',MODELS/'phrase_update.safetensors','cpu')
    stream = PhraseInk(*(LocalInk.load(DIRECTORY/(key+'.npz')) for key in ('onset','later','end')))
    paths = {'fit.json':WORK/'fit.json','confirmation_roster.json':WORK/'confirmation/receipt.json'}
    for key in ('result.json','binding.json','ratings.txt'):
        paths['reading_'+key] = WORK/'reading'/key
    table = [['Exact stopped phrases','Discovery','Replication','Previous holdout','New holdout']]
    cases, chart = {},[[],[]]
    counts = {a:[] for a in ('marginal','interaction','local')}
    sizes, audit, widths = [],{},{}
    for split in (*COHORTS,'fresh','confirmation'):
        folder = WORK/'results'/split
        paths[split+'.json'] = folder/'report.json'
        paths[split+'_exchange.json'] = folder/'exchange.json'
        doc = load(folder/'report.json')
        exchange = load(folder/'exchange.json')
        chart[0].append(exchange['assignment_wins'])
        chart[1].append(exchange['losses'])
        rows,bank = inputs(split)
        loc = list(sites_independent(rows))
        sizes.append(len(loc))
        drawings, exact = {},{}
        for name in counts:
            path = folder/'interaction.npz' if name=='interaction' else control(split,name)
            require(sha(path) == doc['arms'][name]['pixels_sha256'],'Stream changed')
            with np.load(path) as z:
                parts = np.split(z['pixels'],z['offsets'][1:-1])
                hits = np.array([st and np.array_equal(p,bank[rows[i]['y'][j:k]]) for p,st,(i,c,j,k)
                                 in zip(parts,z['stopped'],loc,strict=True)])
            require(int(hits.sum())==doc['arms'][name]['exact'],'Independent exact count differs')
            counts[name].append(str(hits.sum()))
            drawings[name],exact[name] = parts,hits
        glyphs = {v.tobytes() for v in bank}
        audit[split] = dict(emitted=sum(len(p) for p in drawings['interaction']),
                            known=sum(g.tobytes() in glyphs for p in drawings['interaction'] for g in p))
        if split not in ('fresh','confirmation'):
            continue
        labels = {bytes(bank[g]):ch for r in rows for key,ids in [('source','x'),('target','y')]
                  for ch,g in zip(r[key],r[ids])}
        for n,(i,c,j,k) in enumerate(loc):
            r=rows[i]
            target = r['y'][j:k]
            phrase = drawings['interaction'][n]
            if split=='fresh' and exact['interaction'][n] and tuple(target) not in seen:
                name = 'novel'
            elif (split=='confirmation' and j and 2<=len(target)<=5 and 2<=len(phrase)<=5
                  and not exact['interaction'][n] and exact['local'][n] and len(r['x'])<=20
                  and all('\u3400'<=ch<='\u9fff' for ch in r['target'][j:k])):
                name = 'failed'
            else:
                continue
            if name in cases:
                continue
            src,pre = bank[r['x']],bank[r['y'][:j]]
            for key,strip in dict(source=src,prefix=pre[-14:],prediction=phrase,
                                  marginal=drawings['marginal'][n],reference=bank[target]).items():
                png(EVIDENCE,name+'_'+key,strip)
                widths[name+'_'+key] = len(strip)
            sx,px = (torch.from_numpy(np.unpackbits(z,axis=1)).bool() for z in (src,pre))
            generated,stopped = net.generate_phrase(sx,c,px,q,stream=stream,interaction=model)
            require(np.array_equal(np.packbits(generated.numpy(),axis=1),phrase), 'CPU example differs from GPU')
            cases[name] = dict(split=split,site=n,cursor=c,source=r['source'],prefix=r['target'][:j],
                               prefix_shown=r['target'][max(0,j-14):j],stopped=stopped,
                               output=''.join(labels.get(bytes(g),'□') for g in phrase),reference=r['target'][j:k])
    require(len(cases)==2,'Missing figure example')
    phrase = cases['novel']['reference']
    train = training_rows()
    inserted = [row['target'][j:k] for row in train for _, _, j, k in sites([row])]
    write_new(EVIDENCE/'novel_sequence_exposure.json',dict(
        sequence=phrase, training_source_occurrences=sum(r['source'].count(phrase) for r in train),
        training_modern_occurrences=sum(r['target'].count(phrase) for r in train),
        training_sentences_with_modern_sequence=sum(phrase in r['target'] for r in train),
        exact_insertion_occurrences=inserted.count(phrase),
        longer_insertions_containing_sequence=sum(phrase in s and phrase!=s for s in inserted)))
    for name,label in [('marginal','Existing visual writer'),('interaction','Adjacent patterns'),('local','Whole-phrase counts')]:
        table.append([label,*counts[name]])
    table.append(['Insertion sites',*[str(n) for n in sizes]])
    for name,src in paths.items():
        shutil.copyfile(src,EVIDENCE/name)
    write_new(EVIDENCE/'figure.json',dict(table=table,exchange_counts=chart,examples=cases,widths=widths))
    write_new(EVIDENCE/'physical_recount.json',audit)
    print(json.dumps(cases,ensure_ascii=False,indent=2),audit)


if __name__=='__main__':
    report()
