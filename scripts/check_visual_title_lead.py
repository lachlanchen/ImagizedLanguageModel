"""Compare against a training-selected rule that copies the opening up to a visual delimiter."""
from collections import Counter
from zipfile import ZipFile

import numpy as np
from scripts.diagnose_visual_context_gradients import load, write_new
from scripts.train_visual_title_transducer import Examples, OUT


def physical(a):
    return [x.tobytes() for x in a]


def lead(p, delimiters, cap=32):
    stop = next((i for i,x in enumerate(p[:cap+1]) if x in delimiters), min(len(p),cap))
    return p[:stop]


def main():
    train, dev = Examples('train'), Examples('dev')
    boundaries=Counter(); pairs=[]
    for i in range(len(train.rows)):
        a,b=train.pair(i);p,y=physical(a),physical(b);pairs.append((p[:33],y))
        if p[:len(y)]==y and len(p)>len(y):
            boundaries[p[len(y)]]+=1
    ranked=[glyph for glyph,_ in boundaries.most_common()]
    candidates=[(sum(lead(p,set(ranked[:k]))==y for p,y in pairs),k) for k in (1,2,4,8,16,32)]
    train_exact,k=max(candidates,key=lambda x:(x[0],-x[1]))
    delimiters=set(ranked[:k]); rows=load(OUT/'rows.json'); diff=[];stats=Counter()
    with ZipFile(OUT/'outputs.zip') as z:
        for i,r in enumerate(rows):
            a,b=dev.pair(i);p,y=physical(a),physical(b)
            baseline=lead(p,delimiters)==y; full=r['normal']['exact']
            diff.append(int(full)-int(baseline))
            stats['exact_lead_rule']+=baseline
            stats['reference_is_article_prefix']+=p[:len(y)]==y
            stats['full_exact_is_article_prefix']+=full and p[:len(y)]==y
            raw=z.read(f'normal/{i}.bin')
            stats['nonempty_full_output_is_article_prefix']+=bool(raw) and a.tobytes().startswith(raw)
    rng=np.random.default_rng(417)
    interval=np.quantile(rng.choice(diff,(2500,len(diff))).mean(1),[.025,.975])
    write_new(OUT/'lead.json',dict(development_questions=len(rows),counts=dict(stats),training_exact=train_exact,
        delimiters_selected=k,candidate_training_exact=candidates,full_minus_lead_95=interval.tolist(),
        scope='Post hoc diagnostic, selection of stopping delimiters uses only training examples.'))


if __name__=='__main__':
    main()
