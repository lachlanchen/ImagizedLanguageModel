import numpy as np
from collections import Counter
from scripts.sample_visual_library import cycle_sample, schedule


def rows(work, lengths):
    return [dict(work=work, cells=[[1]*en, [1]*zh]) for en,zh in lengths]


def test_multiple_works_parallel_edges_and_both_length_marginals():
    data = rows("a",[(3,4),(3,5),(6,4),(6,5)]*20) + rows("b",[(3,4)]*11+[(6,5)])
    replacements, result = schedule(data)
    assert 0 < result["cycles"] and result["distinct"] < len(data)
    for t,indices in enumerate(replacements):
        assert len(indices)==len(data)
        for i,j in enumerate(indices):
            assert data[i]["work"] == data[j]["work"]
            assert len(data[i]["cells"][t]) == len(data[j]["cells"][t])
        for side in (0,1):
            signature = lambda r:(r["work"],len(r["cells"][side]))
            assert Counter(map(signature,data))==Counter(signature(data[j]) for j in indices)
    assert Counter(replacements[0])==Counter(replacements[1])


def test_single_cycle_has_the_two_equal_probability_outcomes():
    class Choice:
        def __init__(self,k):
            self.k=k
        def permutation(self,values):
            return values
        def integers(self,bound):
            assert bound==2
            return self.k
    data=rows("a",[(3,4),(3,5),(6,4),(6,5)])
    a,n=cycle_sample(data,Choice(0),cap=2)
    b,m=cycle_sample(data,Choice(1),cap=2)
    assert n==m==1
    assert sorted(a)==[0,0,2,2] and np.all(a+b==2)
