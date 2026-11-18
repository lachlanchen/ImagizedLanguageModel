"""Select real changed outputs and publish measurements for the neighborhood experiment."""
import shutil
import numpy as np
from scripts.train_ordered_neighborhood import OUT, ROOT, load, verified_split, write_new
from scripts.report_passage_diversity import changed_samples

DEST = ROOT/'references/evidence/ordered_neighborhood_20261118'


def main():
    files = {name+'.json':OUT/(name+'.json') for name in ('evaluation','fits','train_resources','evaluate_resources')}
    files.update(preparation=OUT/'fresh/preparation.json',transfer=OUT/'transfer/result.json')
    report = load(files['evaluation.json'])
    transfer = load(files['transfer'])
    DEST.mkdir()
    for name,path in files.items():
        shutil.copyfile(path,DEST/(name if name.endswith('.json') else name+'.json'))
    fresh = report['fresh']
    samples = changed_samples(fresh,verified_split(OUT/'fresh'),np.load(OUT/'fresh/pixels.npy'),
                              OUT/'evaluate_fresh',DEST,'uniform','aligned')
    table=[['Reader','All correct','Eight seen','Eight new','Count baseline missed']]
    for name,label in [('start','Starting writer'),('uniform','Original reader'),('shuffled','Shuffled local view'),('aligned','Aligned local view')]:
        a=fresh['arms'][name]
        table.append([label,*[str(a[k]) for k in ('exact','repeat_exact','new_exact','local_misses_correct')]])
    write_new(DEST/'figure.json',dict(table=table,samples=samples,distinct_donors=transfer['distinct_donors']))

if __name__=='__main__':
    main()
