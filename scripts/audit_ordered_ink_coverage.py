"""Bound new-ink generation by the exact physical inventory available to the decoder."""
from collections import Counter
import json
import numpy as np
from ilm.visual_lm.ordered_page_features import cells
from scripts.audit_ordered_page_answers import read_outputs
from scripts.cache_ordered_page_features import PUBLIC,WORK,sha
from scripts.compare_ordered_requests import OUT
from scripts.ordered_image_corpus import corpus,OPEN
from scripts.train_ordered_page_writer import dump


def subsequence(needle,haystack):
    stream=iter(haystack)
    return all(any(x==want for x in stream) for want in needle)


def run():
    rows=corpus("development")
    sources=json.loads((OPEN/"development/sources.json").read_text())
    receipt=json.loads((PUBLIC/"fit.json").read_text())
    assert sha(WORK/"writing_memory.npy")==receipt["memory_sha256"]
    memory=set(map(bytes,np.load(WORK/"writing_memory.npy")))
    study=json.loads((PUBLIC/"variation.json").read_text())
    report={}
    for arm,result in study["arms"].items():
        answers,stops=read_outputs(OUT,arm,result["output_sha256"])
        counts=Counter()
        for r,out,stop in zip(rows,answers[:len(rows)],stops[:len(rows)],strict=True):
            tiles=cells(out) if out else []
            if r.task=="single":
                counts["roles"]+=1
                counts["role_stopped"]+=int(stop)
                counts["role_expected_length"]+=int(len(out)==len(r.target))
                counts["role_contains_name"]+=int(any(out[i:i+len(r.target)]==r.target for i in range(0,len(out),128)))
            if r.task!="open" or sources[int(r.case)]["kind"]!="absent_content_ink":
                continue
            source=set(cells(r.passage))
            required=[t for t in cells(r.target) if t not in source]
            assert required
            emitted=[t for t in tiles if t not in source]
            assert set(emitted)<=memory
            counts["cases"]+=1
            feasible=set(required)<=memory
            counts["target_representable"]+=int(feasible)
            counts["missing_positions"]+=len(required)
            counts["representable_missing_positions"]+=sum(t in memory for t in required)
            counts["any_external_ink"]+=bool(emitted)
            counts["any_required_external_ink"]+=bool(set(emitted)&set(required))
            counts["all_required_external_in_order"]+=subsequence(required,emitted)
            counts["exact_if_feasible"]+=int(feasible and stop and out==r.target)
        report[arm]=dict(counts)
    dump(PUBLIC/"ink_coverage.json",report)
    print(report)


if __name__=="__main__":
    run()
