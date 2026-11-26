"""Percentiles of the paired, equally weighted book bootstrap with exact contrasts."""
from fractions import Fraction
from itertools import product


def work_interval(works, candidate="prediction", control="regression"):
    contrasts=[Fraction(v[candidate]-v[control],v["n"]) for v in works.values()]
    samples=sorted(sum(d)/len(d) for d in product(contrasts,repeat=len(contrasts)))
    def percentile(q):
        rank=(len(samples)-1)*q
        lo=rank.numerator//rank.denominator
        hi=min(lo+1,len(samples)-1)
        return (1-(rank-lo))*samples[lo]+(rank-lo)*samples[hi]
    return [float(percentile(q)) for q in (Fraction(1,40),Fraction(39,40))]
