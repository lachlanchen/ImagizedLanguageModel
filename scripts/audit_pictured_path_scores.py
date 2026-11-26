"""Count the finite cases appearing in the source-repair theorem and check physical output bindings."""
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from zipfile import ZipFile
from scripts.reserve_pictured_path_translation import PUB, ROOT, dump, read, sha

ARCHIVE=ROOT/"publication/ilm-nature/evidence/pictured_path_answers.zip"


def main():
    assessment=read(PUB/"assessment.json")
    replay=read(PUB/"replay.json")
    assert replay["archive"]==assessment["archive"]==sha(ARCHIVE)
    with ZipFile(ARCHIVE) as z:
        rows=json.loads(z.read("records.json"))
        grades=assessment["primary"]["records"]
        votes=assessment["votes"]
        counts=Counter()
        per_direction={"en":Counter(),"zh":Counter()}
        for i,(r,g,controls) in enumerate(zip(rows,grades,assessment["controls"],strict=True)):
            answers=r["answers"]
            for k,a in answers.items():
                assert sha256(z.read(f"{i:04d}/{k}.png")).hexdigest()==a["sha256"]
            for k,a in (answers|controls).items():
                assert g[k]==bool(a["stop"] and votes[2*i][k] and votes[2*i+1][k])
            path,native=answers["path"],answers["native"]
            equal=all(path[k]==native[k] for k in ("sha256","boundaries","stop","cells"))
            exact=path["native_path_equal"]
            for group in (counts,per_direction[r["direction"]]):
                group.update(n=1,exact=exact,identical=equal,identical_despite_other_path=equal and not exact,
                    exact_but_changed=exact and not equal,exact_score_disagreement=exact and g["native"]!=g["path"],
                    inferred_credits=g["path"],supplied_credits=g["native"],exact_uncredited=exact and not g["path"],
                    repaired=g["native"] and not g["path"],lost=g["path"] and not g["native"],
                    field_repaired=g["path"] and not g["field"],field_lost=g["field"] and not g["path"],
                    field_stopped=answers["field"]["stop"],path_stopped=path["stop"],native_stopped=native["stop"])
    for x in [counts,*per_direction.values()]:
        assert x["inferred_credits"]+x["repaired"]==x["supplied_credits"]+x["lost"]
        assert x["exact_uncredited"]<=x["n"]-x["supplied_credits"]+x["exact_score_disagreement"]
    dump(PUB/"repair-audit.json",dict(archive=sha(ARCHIVE),assessment=sha(PUB/"assessment.json"),
        proof=sha(ROOT/"formal/span_list/SourcePathRepair.lean"),counts=counts,directions=per_direction,
        program=sha(Path(__file__))))
    print(counts,flush=True)


if __name__=="__main__":
    main()
