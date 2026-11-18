"""Evaluate the effect of supervised question/answer assignments on a real reader."""
from collections import defaultdict
from dataclasses import replace
import os
import resource
import time

import numpy as np
from threadpoolctl import threadpool_limits

from scripts.train_visual_extractive_reader import OUT, ROOT, guard, load, load_cases, require, sha, sha_bytes, train_reader, write_new

DEST=ROOT/'artifacts/visual_extraction_supervision_v1'


def main():
    began=time.monotonic()
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU only')
    guard(began,True)
    source=load(OUT/'report.json')
    require(sha(OUT/'rows.json')==source['rows_sha256'],'Original prediction file differs')
    for file,digest in source['implementation_sha256'].items():
        require(sha(ROOT/file)==digest,'Original training implementation changed')
    DEST.mkdir(exist_ok=False)
    with threadpool_limits(2):
        train,test=load_cases(began)
        by_group=defaultdict(list)
        for case in train:
            by_group[case.group].append(case)
        reassigned={}
        for group in by_group.values():
            ordered=sorted(group,key=lambda case:case.label)
            for original,other in zip(ordered,ordered[1:]+ordered[:1]):
                reassigned[original.label]=other.answer
        rotated=[replace(c,answer=reassigned[c.label]) for c in train]
        changed=sum(not np.array_equal(a.answer,b.answer) for a,b in zip(train,rotated))
        t=time.monotonic()
        model,training=train_reader(rotated,began)
        fit=time.monotonic()-t
        observations=[]
        stats=defaultdict(lambda:np.zeros(3,dtype=int))
        original=load(OUT/'rows.json')
        for i,(c,r) in enumerate(zip(test,original,strict=True)):
            if i%100==0:
                guard(began)
            require(c.label==r['source'] and c.group==r['group'] and
                    sha_bytes(c.answer.tobytes())==r['reference_sha256'],'Fixed test row mismatch')
            p=r['predictions']['normal']
            real=c.passage[p['start']:p['end']]
            matched=np.array_equal(real,c.answer)
            require(matched==p['exact'] and sha_bytes(real.tobytes())==p['sha256'],'Changed original output')
            start,end=model.bounds(c.passage,c.question)
            output=c.passage[start:end]
            control=np.array_equal(output,c.answer)
            stats[c.group]+=[1,int(matched),int(control)]
            observations.append(dict(source=c.label, group=c.group, start=start, end=end,
                                output_sha256=sha_bytes(output.tobytes()), matched=matched, rotated=control))
    matrix=np.asarray([stats[g] for g in sorted(stats)])
    n,m,c=map(int,matrix.sum(0))
    require(n==1788 and m==269 and len(matrix)==425,'Original cohort differs')
    rng=np.random.default_rng(202611212)
    estimates=[]
    for _ in range(100):
        a,b,d=matrix[rng.integers(0,len(matrix),(100,len(matrix)))].sum(1).T
        estimates.extend((b-d)/a)
    ci=np.quantile(estimates,[.025,.975]).tolist()
    write_new(DEST/'observations.json',observations)
    guard(began)
    result=dict(training=training,training_answers_changed=changed,fit_seconds=fit,questions=n,groups=len(matrix),
                original_exact=m,control_exact=c,original_only=sum(r['matched'] and not r['rotated'] for r in observations),
                control_only=sum(r['rotated'] and not r['matched'] for r in observations),difference_ci=ci,
                qualified=ci[0]>0,elapsed_seconds=time.monotonic()-began,
                peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,
                original_report_sha256=sha(OUT/'report.json'),observations_sha256=sha(DEST/'observations.json'),
                script_sha256=sha(ROOT/'scripts/check_visual_extraction_supervision.py'),
                protocol_sha256=sha(ROOT/'references/visual_extraction_supervision_control.md'))
    write_new(DEST/'report.json',result)
    print(result,flush=True)


if __name__=='__main__':
    main()
