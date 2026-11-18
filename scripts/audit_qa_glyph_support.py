"""Measure training landmark collisions and physical source-span availability for real QA."""
from collections import Counter, defaultdict
import time

import numpy as np
import torch

from scripts.train_glyph_planner import OUT, ROOT, DIVERSITY, corpora, load, require, sha, write_new
from scripts.train_visual_extractive_reader import load_cases


def inventory(tiles, occurrences, landmarks, counts):
    keys = np.packbits(np.unpackbits(tiles,axis=1)[:,landmarks],axis=1)
    for key, tile, n in zip(keys,tiles,occurrences):
        counts[key.tobytes()][tile.tobytes()] += int(n)


def score(cases, counts, landmarks):
    # Resolve equal-frequency alternatives deterministically using only training bitmaps.
    preferred = {code:min(options, key=lambda tile:(-options[tile],tile)) for code,options in counts.items()}
    total = Counter()
    for case in cases:
        answer = case.answer.reshape(-1,128)
        passage = np.ascontiguousarray(case.passage.reshape(-1,128))
        keys = np.packbits(np.unpackbits(answer,axis=1)[:,landmarks],axis=1)
        bitmap = [v.tobytes() for v in answer]
        codes = [k.tobytes() for k in keys]
        supported = [v in counts.get(k,{}) for v,k in zip(bitmap,codes)]
        correct = [v == preferred.get(k) for v,k in zip(bitmap,codes)]
        # Search only starts on physical cell boundaries.
        occurrences = np.flatnonzero((passage==answer[0]).all(1))
        contiguous = any(np.array_equal(passage[p:p+len(answer)],answer) for p in occurrences)
        visible = set(map(bytes,passage))
        total.update(cases=1,glyphs=len(bitmap),seen_target_cells=sum(supported),majority_correct_cells=sum(correct),
                     all_training_shapes=int(all(supported)),full_training_majority=int(all(correct)),
                     answer_in_one_source_span=int(contiguous),all_individually_visible=int(all(v in visible for v in bitmap)),
                     majority_miss_source_span=int(contiguous and not all(correct)),
                     majority_correct_within_eight=int(all(correct) and len(answer)<=8))
    return dict(total)


def main():
    train,test=load_cases(time.monotonic())
    _,idioms=corpora()
    bank=np.load(DIVERSITY/"pool/pixels.npy")
    weights=OUT/"aligned/weights.pt"
    require(sha(weights)==load(OUT/"aligned/fit.json")["weights_sha256"],"Changed landmark basis")
    lm=torch.load(weights,map_location="cpu",weights_only=True)["landmarks"].numpy()
    idx,n=np.unique(idioms["target"],return_counts=True)
    counts=defaultdict(Counter)
    inventory(bank[idx],n,lm,counts)
    report={}
    for name in ("idiom_only","idiom_and_qa_training"):
        if name=="idiom_and_qa_training":
            tiles,frequency=np.unique(np.concatenate([c.answer.reshape(-1,128) for c in train]),axis=0,return_counts=True)
            inventory(tiles,frequency,lm,counts)
        report[name]=dict(distinct_bitmaps=sum(map(len,counts.values())),distinct_codes=len(counts),
                          ambiguous_codes=sum(len(v)>1 for v in counts.values()),
                          conflicting_types=sum(len(v) for v in counts.values() if len(v)>1),
                          irreducible_training_occurrences=sum(sum(v.values())-max(v.values()) for v in counts.values()),
                          heldout=score(test,counts,lm))
    write_new(OUT/"qa_glyph_support.json",dict(report=report,program_sha256=sha(__file__),
              weights_sha256=sha(weights),scope="Training-majority reference lookup and source availability only; no learned answer selection."))
    print(report)


if __name__ == "__main__":
    main()
