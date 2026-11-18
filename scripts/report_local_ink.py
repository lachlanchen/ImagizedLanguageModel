"""Audit frozen streams and export original examples for the local ink comparison."""
import argparse
import shutil
from pathlib import Path
import numpy as np
import torch

from ilm.visual_lm.local_ink import LocalInk, PhraseInk
from scripts.infer_visual_phrase import MODELS, restore_public
from scripts.report_visual_phrases import png, sites_independent
from scripts.evaluate_stream_ink import COHORTS, PRIOR, ROOT, RUN, load, require, roster, sha, write_new


@torch.inference_mode()
def report(directory):
    directory.mkdir(parents=True)
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('high')
    for arm in ('source','prefix'):
        for prefix,folder in [('onset',RUN),('stream',RUN/'stream')]:
            shutil.copyfile(folder/(arm+'_fit.json'),directory/(prefix+'_'+arm+'_fit.json'))
    shutil.copyfile(RUN/'fresh_roster/receipt.json',directory/'fresh_roster.json')
    shutil.copyfile(MODELS/'local_ink/manifest.json',directory/'public_manifest.json')
    net,q = restore_public(MODELS/'editor.safetensors',MODELS/'phrase_update.safetensors','cpu')
    stream_model = PhraseInk(*(LocalInk.load(MODELS/'local_ink'/(part+'.npz')) for part in ('onset','later','end')))
    examples,widths,counts,audit = {},{},{},{}
    for split in (*COHORTS,'fresh'):
        old = split in COHORTS
        folder = RUN/'stream'/split
        stats = load(folder/'report.json')
        for src,dst in [(folder/'report.json',split+'_stream.json'),(folder/'length_audit.json',split+'_length.json')]:
            shutil.copyfile(src,directory/dst)
        if old:
            shutil.copyfile(RUN/split/'report.json',directory/(split+'_onset.json'))
        rows,bank = roster(split)
        loc = list(sites_independent(rows))
        counts[split] = stats['arms']
        outputs = {}
        for arm,record in stats['arms'].items():
            path = folder/(arm+'.npz')
            if old and arm in ('prior','local','onset'):
                path = (RUN if arm=='onset' else PRIOR)/split/(('local' if arm=='local' else 'source')+'.npz')
            require(sha(path)==record['pixels_sha256'],'Unexpected stream digest')
            with np.load(path) as z:
                parts = np.split(z['pixels'],z['offsets'][1:-1])
                stops = z['stopped']
            correct = np.asarray([st and p.tobytes()==bank[rows[i]['y'][j:k]].tobytes()
                                  for p,st,(i,c,j,k) in zip(parts,stops,loc,strict=True)])
            require(int(correct.sum())==record['exact'] and int(stops.sum())==record['stopped'],'Physical recount differs')
            outputs[arm] = (parts,stops,correct)
        exact = outputs['source'][2]
        audit[split] = dict(source_only_vs_local=int((exact&~outputs['local'][2]).sum()),
                            local_only_vs_source=int((~exact&outputs['local'][2]).sum()))
        if split!='fresh':
            continue
        charmap = {g:char for r in rows for text,ids in [('source','x'),('target','y')]
                   for char,g in zip(r[text],r[ids])}
        bitmap_char = {bytes(bank[g]):ch for g,ch in charmap.items()}
        for n,(i,c,j,k) in enumerate(loc):
            r = rows[i]
            ref = r['target'][j:k]
            predicted = outputs['source'][0][n]
            if (not j or len(r['x'])>23 or j>21 or len(predicted)>14 or k-j<2
                    or not all('\u3400'<=ch<='\u9fff' for ch in ref) or not set(ref).isdisjoint(r['source'])):
                continue
            key = ('success' if exact[n] and not outputs['prior'][2][n] else
                   'error' if not exact[n] and outputs['local'][2][n] else '')
            if not key or key in examples:
                continue
            pics = dict(source=bank[r['x']],prefix=bank[r['y'][:j]],prediction=predicted,
                        local=outputs['local'][0][n],reference=bank[r['y'][j:k]])
            for name,pic in pics.items():
                png(directory,key+'_'+name,pic)
                widths[key+'_'+name] = len(pic)
            source,prefix = (torch.from_numpy(np.unpackbits(pics[name],axis=1)).bool() for name in ('source','prefix'))
            actual,ended = net.generate_phrase(source,c,prefix,q,stream=stream_model)
            require(np.array_equal(np.packbits(actual.numpy(),axis=1),predicted) and ended==outputs['source'][1][n],
                    'Standalone compact writer differs')
            examples[key] = dict(index=i,site=n,boundary=c,source=r['source'],prefix=r['target'][:j],reference=ref,
                                 output=''.join(bitmap_char.get(bytes(p),'□') for p in predicted),stopped=ended)
        require(len(examples)==2,'Missing fresh illustrations')
    labels = [('prior','Original phrase writer'),('onset','Local evidence on first glyph'),('source','Local evidence throughout'),
              ('prefix','Modern prefix only, throughout'),('local','Local whole-phrase counts')]
    table = [['Exact stopped phrases','Discovery','Replication','Fresh sentences']]
    for arm,label in labels:
        table.append([label,*[str(counts[s][arm]['exact']) for s in counts]])
    chart = [[counts[s][arm]['multi'][0] for s in counts] for arm in ('onset','source','local')]
    write_new(directory/'figure.json',dict(table=table,multi_counts=chart,widths=widths,examples=examples))
    write_new(directory/'physical_recount.json',audit)
    print(examples,audit)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=RUN/'report')
    report(p.parse_args().output)
