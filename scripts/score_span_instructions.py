"""Score frozen contrast translations and the physical reachability of each historical sequence."""
from collections import Counter
from zipfile import ZipFile
import json
from hashlib import sha256
from scripts.probe_span_instructions import ARCHIVE,PUB,read,sha,dump
from scripts.evaluate_optical_span_rollout import blind_votes,configure_cuda_cli,gpu_owner


def main():
    configure_cuda_cli()
    assert read(PUB/"word-origin-images.json")["archive"]==sha(ARCHIVE)
    review=read(PUB/"origin-review.json")
    assert review["archive"]==sha(ARCHIVE)
    credited={m:[] for m in ("single","span")}
    with ZipFile(ARCHIVE) as z:
        rows=json.loads(z.read("records.json"))
        origin_ids=[i for i,r in enumerate(rows) if r["kind"] in ("English origin","Chinese origin")]
        assert [v["i"] for v in review["decisions"]]==origin_ids
        for choice in review["decisions"]:
            i=choice["i"]
            assert choice["expected"]==rows[i]["reference"]
            for m,rating in choice["arms"].items():
                assert rating["sha256"]==sha256(z.read(f"{i:03d}/{m}.png")).hexdigest()
                if rating["credited"]: credited[m].append(i)
    receipt=PUB/"word-origin-scores.json"
    if receipt.exists():
        frozen=read(receipt)
        assert frozen["archive"]==sha(ARCHIVE)
        assert all(credited[m]==frozen["origin_review"][m+"_credited"] for m in credited)
        return
    translation=[r for r in rows if r["kind"]=="translation"]
    inputs=[dict(t=("en","zh").index(r["to"]),source=r["question"]) for r in translation]
    answers=[dict(span=r["span"],single=r["single"],reference=dict(text=r["reference"],stop=True)) for r in translation]
    with gpu_owner(): votes=blind_votes(inputs,answers)
    grades=[{m:bool(a["stop"] and votes[2*i][m] and votes[2*i+1][m]) for m,a in ref.items()} for i,ref in enumerate(answers)]
    groups=Counter()
    paired=Counter()
    for m in grades[0]:
        groups[m]=sum(g[m] for g in grades)
        paired[m]=sum(grades[i][m] and grades[i+2][m] for i in range(len(grades)) if i%4<2)
    historical=[r for r in rows if r["kind"]=="historical forms"]
    # All origin replies were read individually against the fixed source dictionary facts.
    origins=[i for i,r in enumerate(rows) if r["kind"] in ("Chinese origin","English origin")]
    dump(receipt,dict(archive=sha(ARCHIVE),votes=votes,contrast_grades=grades,contrast_totals=groups,
        both_correct_of_eight=paired,historical_correct={m:sum(r[m]["ordered_forms"] for r in historical) for m in ("span","single")},
        historical_requests=len(historical),reachable_target_instances=sum(sum(r["representable"]) for r in historical),
        requested_target_instances=sum(len(r["reference"]) for r in historical),
        origin_review=dict(cases=origins,span_credited=credited["span"],single_credited=credited["single"],
        method="Author-side bilingual review by the research agent against frozen entry facts. Each reply either asks a question, restates the instruction, repeats characters, or gives an unrelated answer. No independent human adjudication.")))
    print(groups,paired,flush=True)


if __name__=="__main__":
    main()
