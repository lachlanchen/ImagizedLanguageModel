#!/usr/bin/env python3
"""Post-hoc physical ink-count diagnosis; changes no original support scores."""
import json
import os
from pathlib import Path

from scripts import diagnose_visual_context_gradients as original

ROOT=original.ROOT
PUBLIC=ROOT/'references/evidence/occupancy_density_ink_mass_20261118.json'


def stats(values):
    import numpy as np
    return dict(positions=len(values),total=int(values.sum()),mean=float(values.mean()),median=float(np.median(values)),
                minimum=int(values.min()),maximum=int(values.max()),q10=float(np.quantile(values,.1)),q90=float(np.quantile(values,.9)))


def main():
    original.require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU saved-array analysis only')
    import numpy as np
    import torch
    source=ROOT/'references/evidence/occupancy_density_preflight_result_20261118.json';run=original.load(source)
    original.require(run['all_2208_generated_images_audited'] and not run['support'],'Use original complete failed comparison')
    cachepath=ROOT/'artifacts/layer_conditioned_writer_preflight_v1/conditional_training_cache.pt'
    cache=torch.load(cachepath,map_location='cpu',weights_only=True)
    target=np.concatenate([v['ink'].numpy().reshape(len(v['ink']),-1).sum(1,dtype=np.int64) for v in cache])
    inputs={str(source.relative_to(ROOT)):original.sha(source),str(cachepath.relative_to(ROOT)):original.sha(cachepath)}
    results=[]
    for arm in run['arms']:
        for stage in ['initial','final']:
            generated=[]
            for case in arm[stage]['cases']:
                p=ROOT/case['raw_file'];original.require(original.sha(p)==case['raw_sha256'],'Original generated array changed')
                inputs[str(p.relative_to(ROOT))]=original.sha(p)
                ink=np.load(p)['ink'];generated.append(ink.reshape(len(ink),-1).sum(1,dtype=np.int64))
            count=np.concatenate(generated);metrics=arm[stage]['pooled_pixels']
            original.require(len(count)==len(target)==552 and int(count.sum())==metrics['tp']+metrics['fp'] and int(target.sum())==metrics['tp']+metrics['fn'],'Independent pixel mass differs from original counts')
            results.append(dict(mode=arm['mode'],stage=stage,generated_ink=stats(count),
                paired_ink_count_absolute_error_mean=float(np.abs(count-target).mean()),
                nonempty_source_with_at_most_four_generated_pixels=int(((target>0)&(count<=4)).sum()),
                nonempty_source_with_more_than_twice_source_ink=int(((target>0)&(count>2*target)).sum()),
                legitimate_space_generated_ink=count[target==0].tolist(),original_pooled_f1=metrics['f1']))
    result=dict(schema='ilm-occupancy-ink-mass-diagnosis-v1',source_ink=stats(target),nonempty_source_positions=int((target>0).sum()),
        groups=results,input_sha256=inputs,analyzer_sha256=original.sha(Path(__file__)),
        post_hoc_descriptive_diagnosis=True,near_empty_and_double_mass_bins_are_not_acceptance_rules=True,
        original_support_and_all_scores_unchanged=True,neural_calls=0,optimizer_updates=0,
        interpretation='Opposite ink-mass errors coexist with correct probability replay. Count control alone would not prove glyph identity or language.',
        full_language_fit_authorized=False,goal_achieved=False)
    original.write_new(PUBLIC,result)
    print(json.dumps(dict(source=result['source_ink'],final=[v for v in results if v['stage']=='final'],sha256=original.sha(PUBLIC))),flush=True)


if __name__=='__main__':main()
