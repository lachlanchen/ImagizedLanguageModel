"""Recount all autonomous sentences and publish a complete short output in fixed source order."""
from collections import Counter
import shutil

import numpy as np
from rapidfuzz.distance import Levenshtein
from scripts.report_first_emission import EVIDENCE, WORK, load, png, require, roster, sha, write_new


def report():
    folder=WORK/'sentences'
    rows,bank=roster('reserved')
    results=load(folder/'report.json')
    streams,summary={},{}
    symbols={bytes(bank[g]):ch for row in rows for field,ids in [('source','x'),('target','y')]
             for ch,g in zip(row[field],row[ids])}
    for arm,stats in results.items():
        path=folder/(arm+'.npz')
        require(sha(path)==stats['archive_sha256'],'Changed sentence images')
        with np.load(path) as z:
            parts=np.split(z['pixels'],z['offsets'][1:-1])
        for row,out,graded in zip(rows,parts,stats['rows'],strict=True):
            ref=list(map(bytes,bank[row['y']]))
            generated=list(map(bytes,out))
            require(Levenshtein.distance(generated,ref)==graded['distance'] and
                    (generated==ref and graded['end']=='stop')==graded['exact'], 'Sentence recount disagrees')
        summary[arm]=dict(exact=stats['exact'],distance=stats['distance'],reference=stats['reference'],
                          ends=Counter(r['end'] for r in stats['rows']),archive_sha256=stats['archive_sha256'],
                          insertions=sum(r['insertions'] for r in stats['rows']))
        streams[arm]=parts
        if arm!='copy':
            shutil.copyfile(path,EVIDENCE/('sentences_'+arm+'.npz'))
    index=next(i for i,r in enumerate(rows) if 1<=len(r['y'])<=25 and len(r['x'])<=25 and
               all(0<len(streams[arm][i])<=25 for arm in ('frozen','preference')))
    r=rows[index]
    widths={}
    for name,data in dict(source=bank[r['x']],reference=bank[r['y']],frozen=streams['frozen'][index],
                          preference=streams['preference'][index]).items():
        png(EVIDENCE,'sentence_'+name,data)
        widths['sentence_'+name]=len(data)
    sample=dict(index=index,source=r['source'],reference=r['target'],widths=widths,
                outputs={k:''.join(symbols.get(bytes(cell),'▧') for cell in streams[k][index]) for k in ('frozen','preference')})
    write_new(EVIDENCE/'sentences.json',dict(arms=summary,sample=sample,roster_size=len(rows)))
    shutil.copyfile(folder/'report.json',EVIDENCE/'sentence_records.json')


if __name__=='__main__':
    report()
