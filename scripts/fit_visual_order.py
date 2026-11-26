"""Estimate which mentioned field comes first using training-only image contexts."""
from collections import Counter
from pathlib import Path
from time import monotonic

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression
from ilm.visual_lm.origin_instruction import TRAIN, question
from ilm.visual_lm.visual_order import MODEL, WIDTH, features, skeleton
from scripts.prepare_region_writer import Ink
from scripts.train_origin_instruction import MODEL as READER_MODEL
from scripts.train_origin_value_reader import OUT, ROOT, RECEIPT, dump, read, rows, sha

PROTOCOL=ROOT/"references/origin_visual_order_protocol.md"
# Locked before fitting the order classifier. Both directions of each order are tested.
FRESH={
    True: ("From the page on {h}, copy {a} followed by {b}.",
           "I need {b} for {h}, but only after you have shown {a}."),
    False: ("查「{h}」，{a}在前，{b}在後。","查到「{h}」以後，{b}應放在{a}的後面。"),
}


def main():
    start=monotonic()
    MODEL.mkdir(parents=True,exist_ok=True)
    path=MODEL/"order.npz"
    source=[Path(__file__),ROOT/"ilm/visual_lm/visual_order.py",ROOT/"ilm/visual_lm/origin_instruction.py",
            PROTOCOL,RECEIPT,READER_MODEL/"varied/fit.json"]
    binding={str(p.relative_to(ROOT)):sha(p) for p in source}
    if (MODEL/"fit.json").exists():
        assert sha(path)==read(MODEL/"fit.json")["weights"] and read(MODEL/"fit.json")["binding"]==binding
        return
    ink=Ink(np.load(OUT/"tiles.npy"))
    counts=Counter()
    missed=0
    for row in rows("train"):
        s=row["supervision"]
        labels=[[ink.tiles[i] for i in row["page"][k]["label"]] for k in s["value_rows"]]
        all_queries=[row["question"]]+[ink.encode(question(row,t)[0]) for t in TRAIN[s["english"]]]
        for q in all_queries:
            item=skeleton([ink.tiles[i] for i in q],*labels)
            if item is None:
                missed+=1
            else:
                zones,direct=item
                counts[features(zones),int(direct)]+=1
    assert not ink.bad
    columns,ptr,y,w=[],[0],[],[]
    for (indices,label),n in counts.items():
        columns.extend(indices);ptr.append(len(columns));y.append(label);w.append(n)
    X=csr_matrix((np.ones(len(columns)),columns,ptr),shape=(len(y),WIDTH))
    # Cap repetitions of a single skeleton; many headwords share exactly the same syntax.
    weights=np.sqrt(np.asarray(w))
    est=LogisticRegression(C=1.,solver="liblinear",max_iter=200,random_state=401)
    est.fit(X,y,sample_weight=weights)
    np.savez_compressed(path,coef=est.coef_[0],bias=est.intercept_[0])
    dump(MODEL/"fit.json",dict(weights=sha(path),binding=binding,train_exposures=sum(w)+missed,
         aligned=sum(w),unresolved=missed,distinct_feature_label_pairs=len(y),active_coefficients=int(np.count_nonzero(est.coef_)),
         correct_training_weight=int(np.dot((est.predict(X)==y),w)),seconds=monotonic()-start))
    print(read(MODEL/"fit.json"),flush=True)


if __name__=="__main__":
    main()
