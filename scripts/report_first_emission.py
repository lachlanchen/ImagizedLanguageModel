"""Independently recount complete physical predictions and locate the first lost glyph."""
import shutil

import numpy as np
import torch

from ilm.visual_lm.conditional_ink import ConditionalInk
from ilm.visual_lm.local_ink import LocalInk, PhraseInk
from ilm.visual_lm.first_emission import FirstEmission
from scripts.evaluate_sentence_choice import DIRECTORY, INTERACTION, ROOT, SPLITS, load, require, roster, sha, write_new
from scripts.evaluate_visual_phrases import chapter_interval
from scripts.train_first_emission import WORK
from scripts.infer_visual_phrase import MODELS, restore_public
from scripts.report_visual_phrases import png, sites_independent

EVIDENCE = ROOT/'references/evidence/first_emission_20261118'
NAMES = ('frozen','likelihood','preference','local')


from scripts.report_sentence_choice import first_loss


@torch.inference_mode()
def report():
    EVIDENCE.mkdir()
    torch.set_num_threads(2)
    net,prompt = restore_public(MODELS/'editor.safetensors',MODELS/'phrase_update.safetensors','cpu')
    ink = ConditionalInk.restore(INTERACTION,sha(DIRECTORY/'manifest.json'),'cpu')
    stream = PhraseInk(*(LocalInk.load(DIRECTORY/(key+'.npz')) for key in ('onset','later','end')))
    writer = FirstEmission.restore(WORK/'fit/preference.safetensors',sha(INTERACTION),'cpu')
    totals = {a:[] for a in NAMES}
    denominators, cases, widths, breakdown = [],{},{},{}
    for split in SPLITS:
        rows,bank=roster(split)
        folder=WORK/'evaluation'/split
        receipt=load(folder/'report.json')
        loc=list(sites_independent(rows))
        targets=[bank[rows[i]['y'][j:k]] for i,c,j,k in loc]
        lengths=np.array([len(t) for t in targets])
        multi = np.where(lengths>1)[0]
        denominators.append(len(loc))
        groups=[rows[i]['g'] for i,_,_,_ in loc]
        outputs,hits,analysis={},{},{}
        for arm in NAMES:
            path=folder/(arm+'.npz')
            require(sha(path)==receipt['arms'][arm]['pixels_sha256'],'Changed physical output')
            with np.load(path) as archive:
                parts=np.split(archive['pixels'],archive['offsets'][1:-1])
                stopped=archive['stopped']
            first=np.array([np.array_equal(p[0],t[0]) for p,t in zip(parts,targets,strict=True)])
            correct=np.array([s and np.array_equal(p,t) for p,t,s in zip(parts,targets,stopped,strict=True)])
            counts=receipt['arms'][arm]
            require((int(first.sum()),int(correct.sum()),int(correct[multi].sum()))==
                    (counts['first_glyph_exact'],counts['exact'],counts['multi'][0]),'Physical recount disagrees')
            if arm in ('likelihood','preference'):
                old_first = np.stack([p[0] for p in outputs['frozen']])
                old_hits = (old_first==np.stack([t[0] for t in targets])).all(1)
                changed = (old_first!=np.stack([p[0] for p in parts])).any(1)
                require([int(v.sum()) for v in (~old_hits & first, old_hits & ~first, changed & ~old_hits & ~first)] ==
                        [counts[k] for k in ('wrong_to_right','right_to_wrong','wrong_to_different_wrong')],
                        'First image transitions disagree')
            totals[arm].append(int(correct.sum()))
            a=dict(multi=first_loss([parts[i] for i in multi],stopped[multi],[targets[i] for i in multi]),
                   multi_first_correct=int(first[multi].sum()),single_exact=int(correct[lengths==1].sum()),
                   generated_single_on_multi=sum(len(parts[i])==1 for i in multi))
            if arm in ('likelihood','preference'):
                ex=load(folder/(arm+'_exchange.json'))
                ix=np.asarray(ex['sites'],dtype=int).reshape(-1,2)
                positive=ix[np.array(ex['pair_margins_nats'])>.0001].ravel()
                a.update(pairs=len(ix),assignment_wins=ex['assignment_wins'],
                         positive_pair_sites=len(positive),positive_pair_first=int(first[positive].sum()),
                         positive_pair_exact=int(correct[positive].sum()),authentic_exchanged=ex['exact_authentic_exchanged'],
                         genuine_rebatch_changed=len(ex['changed_batch_sites']))
            analysis[arm]=a
            outputs[arm]=parts
            hits[arm]=correct
        analysis['preference_minus_likelihood_pp']=chapter_interval(groups,hits['preference'].astype(int)-hits['likelihood'])
        breakdown[split]=analysis
        shutil.copyfile(folder/'report.json',EVIDENCE/(split+'.json'))
        for arm in ('likelihood','preference'):
            shutil.copyfile(folder/(arm+'_exchange.json'),EVIDENCE/(split+'_'+arm+'_exchange.json'))
        # First bounded, nonempty-prefix, multi-glyph win and error from a confirmation roster.
        # An error must be solved by the count control; neither example may be cropped within an answer.
        if split not in ('confirmation','reserved'):
            continue
        symbols={bytes(bank[g]):ch for row in rows for field,ids in [('source','x'),('target','y')]
                 for ch,g in zip(row[field],row[ids])}
        for n,(i,c,j,k) in enumerate(loc):
            row=rows[i]
            if not j or len(row['x'])>22 or not 2<=k-j<=6 or any(len(outputs[a][n])>12 for a in ('frozen','preference')):
                continue
            won=hits['preference'][n]
            if won and not hits['frozen'][n] and not np.array_equal(outputs['frozen'][n][0], targets[n][0]):
                name='gained'
            elif not won and hits['local'][n] and not np.array_equal(outputs['preference'][n][0], targets[n][0]):
                name='missed'
            else:
                continue
            if name in cases:
                continue
            x,prefix=bank[row['x']],bank[row['y'][:j]]
            for field,raw in dict(source=x,prefix=prefix[-14:],frozen=outputs['frozen'][n],
                                  prediction=outputs['preference'][n],reference=targets[n]).items():
                png(EVIDENCE,name+'_'+field,raw)
                widths[name+'_'+field]=len(raw)
            args=[torch.from_numpy(np.unpackbits(v,axis=1)).bool() for v in (x,prefix)]
            raw,end=net.generate_phrase(args[0],c,args[1],prompt,stream=stream,interaction=ink,first_emission=writer)
            require(np.array_equal(np.packbits(raw.numpy(),axis=1),outputs['preference'][n]),'Published CPU example differs')
            cases[name]=dict(split=split,site=n,source=row['source'],cursor=c,prefix=row['target'][:j],
                             target=row['target'][j:k],following=row['target'][k:],cpu_stopped=end,
                             frozen=''.join(symbols.get(bytes(v),'▧') for v in outputs['frozen'][n]),
                             prediction=''.join(symbols.get(bytes(v),'▧') for v in outputs['preference'][n]))
    require(len(cases)==2,'Missing success or error')
    table=[['Exact stopped phrases','Discovery','Replication','Holdout 1','Holdout 2','New chapters']]
    for arm,label in zip(NAMES,('Frozen writer','Likelihood','Preference','Whole-phrase counts')):
        table.append([label,*map(str,totals[arm])])
    table.append(['Insertion sites',*map(str,denominators)])
    chart = [[sum(load(WORK/'evaluation'/s/'report.json')['arms'][a][key] for s in SPLITS)
              for key in ('wrong_to_right','right_to_wrong','wrong_to_different_wrong')] for a in ('likelihood','preference')]
    write_new(EVIDENCE/'figure.json',dict(table=table,exchange_counts=chart,examples=cases,widths=widths))
    write_new(EVIDENCE/'first_error.json',breakdown)
    for src,dst in [(WORK/'fit/fit.json','fit.json'),(WORK/'preparation.json','preparation.json')]:
        shutil.copyfile(src,EVIDENCE/dst)
    print(cases,chart,flush=True)


if __name__=='__main__':
    report()
