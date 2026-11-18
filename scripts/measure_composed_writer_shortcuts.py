"""Count passage-only baselines, paired outcomes and available positive writing examples."""
from collections import Counter
import numpy as np

from scripts.train_composed_answer_writer import DATA, OUT, Split
from scripts.prepare_pictured_entry_answers import Rasterizer, write_json


def images(path):
    with np.load(path) as z:
        return [(s.tobytes(),bool(stop)) for s,stop in zip(np.split(z["pixels"],z["offsets"][1:-1]),z["stopped"],strict=True)]


def measure():
    train,dev=Split("train"),Split("development")
    frequencies=Counter()
    for p,q,y in zip(train.p,train.q,train.y,strict=True):
        visible=set(np.concatenate((p,q)))
        frequencies.update(bytes(train.tiles[g]) for g in y if g not in visible)
    fit_memory=np.load(OUT/"memory.npy")
    histogram=Counter()
    punctuation={Rasterizer().tile(c) for c in "。！？!?；;"}
    original=images(OUT/"raster.npz")
    exchanged=images(OUT/"exchanged.npz")
    counts=Counter()
    for i,(p,q,y) in enumerate(zip(dev.p,dev.q,dev.y,strict=True)):
        passage=list(map(bytes,dev.tiles[p]))
        reference=list(map(bytes,dev.tiles[y]))
        visible=set(passage)|set(map(bytes,dev.tiles[q]))
        for g in reference:
            if g not in visible:
                n=frequencies[g]
                bucket="0" if n==0 else "1-3" if n<=3 else "4-15" if n<=15 else "16+"
                histogram[bucket]+=1
        exact=original[i]==(b"".join(reference),True)
        swapped=exchanged[i]==(b"".join(reference),True)
        same=original[i]==exchanged[i]
        end=next((k+1 for k,g in enumerate(passage) if g in punctuation),len(passage))
        whole=passage==reference
        first=passage[:end]==reference
        counts.update(dict(whole_passage_exact=int(whole),first_sentence_exact=int(first),
                           whole_passage_and_raster_exact=int(whole and exact),
                           first_sentence_and_raster_exact=int(first and exact),
                           identical_after_question_rotation=int(same),
                           both_exact=int(exact and swapped),only_original=int(exact and not swapped),
                           only_rotated=int(swapped and not exact)))
    return dict(positive_write_bitmaps=len(frequencies),memory_bitmaps=len(fit_memory),
                development_missing_positions_by_number_of_training_writes=dict(histogram),**counts)


if __name__=="__main__":
    result=measure()
    print(result)
    write_json(OUT/"shortcut_counts.json",result)
