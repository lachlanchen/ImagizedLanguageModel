import itertools

import numpy as np

from scripts.analyze_v62_partial_transcript_bounds import envelope


def test_bounds_cover_every_missing_score_completion_with_grouped_resampling():
    lower=np.array([0.,0.,1.,.25]);upper=np.array([1.,1.,1.,.25])
    articles=["a","b","b","c"];groups=[np.array([0]),np.array([1,2]),np.array([3])]
    result=envelope(lower,upper,articles)
    assert result["identification_interval"]==[.3125,.8125]
    draws=np.random.default_rng(20266379).integers(0,3,(2000,3))
    resampled_indices=[np.concatenate([groups[i] for i in row]) for row in draws]
    for a,b in itertools.product((0.,.5,1.),repeat=2):
        scores=np.array([a,b,1.,.25])
        direct_means=np.array([scores[indices].mean() for indices in resampled_indices])
        low,high=np.quantile(direct_means,[.025,.975])
        outer=result["article_bootstrap_sensitivity_envelope"]
        assert outer[0]<=low+1e-14 and high<=outer[1]+1e-14
        assert result["identification_interval"][0]<=scores.mean()<=result["identification_interval"][1]
