"""Recount blank and source probabilities from the measured patch writer and frozen images."""
import math
import os
from pathlib import Path
import time

import numpy as np
from ilm.visual_lm.patch_count_language import ESCAPE, PatchCountLanguage, tile_features
from scripts.diagnose_visual_context_gradients import load, require, sha, write_new
from scripts.measure_visual_successors import ROOT, SOURCE, SOURCE_HASH, digest, guard
from scripts.prepare_visual_sequence_corpus import Rasterizer
from scripts.train_patch_count_language import OUT as TRAIN


def complete_score(model, history, parts):
    """Log mass given continuation, including the uniform escape for any 64-bit patch."""
    nodes=[]
    total=0.
    for j, bits in enumerate(parts):
        up=nodes[j-4] if j>=4 else 0
        left=nodes[j-1] if j%4 else 0
        alpha=model.parts[j][0]
        idx=int(np.searchsorted(alpha,bits))
        known=idx<len(alpha) and alpha[idx]==bits
        mass=float(model.patch_probabilities(j,history,up*model.part_radix+left)[idx]) if known else 0.
        total+=math.log((1-ESCAPE)*mass + ESCAPE*2**-64)
        nodes.append(idx+1 if known else 0)
    return total


def main():
    start=time.monotonic()
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU only')
    report=load(TRAIN/'result.json')
    require(sha(SOURCE)==SOURCE_HASH and all(sha(TRAIN/p)==report['files'][p] for p in
            ('ordered/model.npz','ordered/successors.npz','articles.json')),'Saved evidence differs')
    output=ROOT/'artifacts/patch_blank_probability_v1'
    output.mkdir(exist_ok=False)
    model=PatchCountLanguage.load(TRAIN/'ordered/model.npz')
    articles={(digest(a['title']),digest(p['context'])):p['context'] for a in load(SOURCE)['data'] for p in a['paragraphs']}
    roster=load(TRAIN/'articles.json')
    with np.load(TRAIN/'ordered/successors.npz',allow_pickle=False) as z:
        masks=z['ink']; targets=z['targets']; stopped=z['stopped']
    require(not stopped.any(),'All observed contexts originally continued')
    _,source=tile_features(targets)
    _,drawn=tile_features(masks)
    blank=np.zeros(16,dtype=np.uint64)
    renderer=Rasterizer()
    scores=[]
    for i,(target,emitted) in enumerate(zip(source,drawn,strict=True)):
        row=roster[i//2]
        text=articles[row['article_sha256'],row['paragraph_sha256']]
        pos=row['offsets'][i%2]
        ink=np.unpackbits(np.frombuffer(b''.join(renderer.tile(c) for c in text[pos-2:pos]),dtype=np.uint8)).reshape(2,32,32)
        h=model.history(ink)
        require(np.packbits(targets[i]).tobytes()==renderer.tile(text[pos]),'Source character differs')
        scores.append([complete_score(model,h,s) for s in (target,emitted,blank)])
    s=np.asarray(scores)
    white=~masks.any(axis=(1,2)); require(np.isfinite(s).all() and np.all(s<=0) and np.array_equal(s[white,1],s[white,2]),'Mass replay')
    require(white.sum()==348 and int((masks==targets).all(axis=(1,2)).sum())==18,'Original pixel counts')
    np.savez_compressed(output/'masses.npz',log_mass=s,emitted_blank=white)
    diff=s[white,0]-s[white,2]
    result=dict(contexts=len(s),emitted_blank=int(white.sum()),source_preferred_to_blank_on_blank_outputs=int((diff>1e-8).sum()),
        blank_preferred_to_source_on_blank_outputs=int((diff< -1e-8).sum()),ties=int((abs(diff)<=1e-8).sum()),
        mean_blank_probability_when_blank=float(np.exp(s[white,2]).mean()),
        all_mean_blank_probability=float(np.exp(s[:,2]).mean()),
        source_minus_blank_log_mass_quartiles=np.quantile(diff,[.25,.5,.75]).tolist(),
        elapsed_seconds=time.monotonic()-start,score_file_sha256=sha(output/'masses.npz'),trial_sha256=sha(TRAIN/'result.json'),
        script_sha256=sha(Path(__file__)),optimizer_updates=0,new_generations=0)
    guard(start)
    write_new(output/'result.json',result)
    print(result)


if __name__=='__main__':
    main()
