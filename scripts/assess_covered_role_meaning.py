"""Distinguish free event meaning from verbatim glyph agreement on frozen answers."""
import json
from zipfile import ZipFile
from scripts.evaluate_covered_phrases import ARCHIVE,ARMS
from scripts.prepare_covered_phrases import PUB,ROOT,base
from scripts.evaluate_visual_library import blind_votes
from scripts.train_span_choice import gpu_owner
from scripts.verse_gpu_runtime import configure_cuda_cli


def main():
    configure_cuda_cli()
    if (PUB/"role-meaning.json").exists():
        assert base.read(PUB/"role-meaning.json")["archive"]==base.sha(ARCHIVE)
        return
    with ZipFile(ARCHIVE) as z:
        all_rows=json.loads(z.read("records.json"))
    indices=[i+j for i in range(96,len(all_rows),20) for j in range(4)]
    assert len(indices)==96
    rows=[all_rows[i] for i in indices]
    candidates=[]
    for k,r in enumerate(rows):
        controls=dict(reference=r["ref"],other_role=rows[k^2]["ref"],other_polarity=rows[k^1]["ref"])
        assert all(controls["reference"]!=controls[m] for m in ("other_role","other_polarity"))
        candidates.append(dict(r["answers"],**{m:dict(text=v,stop=True) for m,v in controls.items()}))
    with gpu_owner():
        votes=blind_votes(rows,candidates)
    stats={}
    grades=[]
    for k,c in enumerate(candidates):
        grades.append({m:bool(v["stop"] and votes[2*k][m] and votes[2*k+1][m]) for m,v in c.items()})
    for m in candidates[0]:
        marks=[r[m] for r in grades]
        stats[m]=dict(meaning=sum(marks),both=sum(all(marks[i:i+2]) for i in range(0,len(marks),2)),
                      four=sum(all(marks[i:i+4]) for i in range(0,len(marks),4)))
    base.dump(PUB/"role-meaning.json",dict(archive=base.sha(ARCHIVE),cases=indices,counts=stats,votes=votes,
                grades=grades,protocol=base.sha(ROOT/"references/covered_phrase_meaning_probe.md")))
    print(stats,flush=True)


if __name__=="__main__":
    main()
