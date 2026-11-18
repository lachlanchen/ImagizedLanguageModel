"""Locate where local training associations and freely drawn expressions disagree."""
from collections import Counter, defaultdict
import numpy as np

from scripts.compare_passage_diversity import corpora, verified_split, local_counts, load_split, load, require, sha, write_new
from scripts.fit_pictured_idiom_alignment import local_keys
from scripts.test_expression_frequency_weight import OUT


def main():
    _, train = corpora()
    test = verified_split(OUT/'fresh')
    px = np.load(OUT/'fresh/pixels.npy')
    words = Counter(map(bytes,train['target']))
    widths = (8,4,2,1,1)
    tables = [defaultdict(Counter) for _ in widths]
    for x,y in zip(train['context'],train['target']):
        for table,key in zip(tables,local_keys(x)):
            table[key][y.tobytes()] += 1
    fallback = words.most_common(1)[0][0]
    local, exact_window, strong = [],[],[]
    for x,y in zip(test['context'],test['target']):
        group = None
        width = 0
        for table,key,width in zip(tables,local_keys(x),widths):
            if key in table:
                group = table[key]
                break
        if group:
            guess,n = group.most_common(1)[0]
            supported = n>=2 and n/sum(group.values())>=.5
        else:
            width,guess,supported = 0,fallback,False
        local.append(guess==y.tobytes())
        exact_window.append(width)
        strong.append(supported)
    local = np.array(local)
    require(np.array_equal(local,local_counts(train,test)), 'Local baseline replay differs')
    encoded = np.array(exact_window)
    strong = np.array(strong)
    result = {}
    ref = load(OUT/'evaluation.json')['fresh']
    for arm in ('unweighted','weighted'):
        path = OUT/'fresh'/arm/'generated.npz'
        require(sha(path)==ref['arms'][arm]['archive_sha256'], 'Changed output')
        saved = load_split(path)
        correct = saved['stopped'] & (saved['length']==4) & (saved['packed'][:,:4]==px[test['target']]).all((1,2))
        data = {}
        for category, mask in [('all',np.ones_like(local)),('eight',encoded==8),('four',encoded==4),
                              ('two',encoded==2),('one',encoded==1),('fallback',encoded==0),('supported',strong)]:
            data[category]=dict(cases=int(mask.sum()),local=int((local&mask).sum()),model=int((correct&mask).sum()),
                               both=int((local&correct&mask).sum()),local_only=int((local&~correct&mask).sum()),
                               model_only=int((~local&correct&mask).sum()))
        result[arm] = data
    mass = np.array(list(words.values()))
    result['training_distribution'] = {'answers':len(words),'passages':int(mass.sum()),
                  'uniform_top10_mass':float(np.sort(mass)[-10:].sum()/mass.sum()),
                  'weighted_top10_mass':float(np.sort(np.sqrt(mass))[-10:].sum()/np.sqrt(mass).sum())}
    write_new(OUT/'local_errors.json',result)

if __name__=='__main__':
    main()
