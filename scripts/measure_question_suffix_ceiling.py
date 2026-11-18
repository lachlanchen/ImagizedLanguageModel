"""Exact finite-corpus ceiling on whole-answer identity after two-glyph prompt compression."""
from collections import Counter, defaultdict
import math
import os
from pathlib import Path
import time

import numpy as np

from ilm.visual_lm.layout_patch_language import LayoutPatchLanguage
from scripts.diagnose_visual_context_gradients import load, require, sha, write_new
from scripts.prepare_visual_sequence_corpus import Rasterizer
from scripts.train_visual_extractive_reader import ROOT, guard, load_cases

OUT = ROOT/'artifacts/question_suffix_ceiling_v1'


def summary(groups):
    counts = [sum(c.values()) for c in groups.values()]
    total = sum(counts)
    maxima = sum(max(c.values()) for c in groups.values())
    entropy = 0.0
    for outcomes, group_size in zip(groups.values(), counts):
        entropy += sum(n * math.log2(group_size/n) for n in outcomes.values())
    return dict(questions=total, distinct_states=len(groups), maximum_exact=maxima,
                maximum_rate=maxima/total, empirical_conditional_entropy_bits=entropy/total,
                questions_in_ambiguous_groups=sum(n for n,c in zip(counts, groups.values()) if len(c)>1))


def main():
    start=time.monotonic()
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','Use CPU')
    guard(start)
    OUT.mkdir(exist_ok=False)
    original=ROOT/'references/evidence/visual_extractive_reading_20261118/result.json'
    require(sha(ROOT/'artifacts/visual_extractive_reader_v1/rows.json')==load(original)['rows_sha256'],
            'Original reading cohort changed')
    train,test=load_cases(start)
    del train
    require(len(test)==1788 and len({r.group for r in test})==425,'Expected previous reading cohort')
    raster=Rasterizer()
    suffix=np.frombuffer(raster.tile('答')+raster.tile('：'),dtype=np.uint8).reshape(2,128)
    groups={name:defaultdict(Counter) for name in ['formatted','raw_question','passage_and_raw_question_suffix','full_question_and_passage']}
    def signature(packed):
        return LayoutPatchLanguage.history(np.unpackbits(packed[-2:],axis=1).reshape(-1,32,32))
    for r in test:
        answer=r.answer.tobytes()
        last=signature(np.concatenate((r.question[-2:],suffix)))
        raw=signature(r.question)
        groups['formatted'][last][answer]+=1
        groups['raw_question'][raw][answer]+=1
        groups['passage_and_raw_question_suffix'][r.group,raw][answer]+=1
        groups['full_question_and_passage'][r.group,r.question.tobytes()][answer]+=1
    report={n:summary(g) for n,g in groups.items()}
    require(report['formatted']['distinct_states']==1 and
            report['formatted']['maximum_exact']<=report['raw_question']['maximum_exact']<=
            report['passage_and_raw_question_suffix']['maximum_exact']<=
            report['full_question_and_passage']['maximum_exact'],'Expected information refinement')
    report.update(original_reader_trial_sha256=sha(original),program_sha256=sha(Path(__file__)),
                  elapsed_seconds=time.monotonic()-start,training_updates=0,generated_answers=0)
    guard(start)
    write_new(OUT/'result.json',report)
    print(report)


if __name__=='__main__':
    main()
