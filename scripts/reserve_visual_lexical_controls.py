"""Reserve naturally occurring passage pairs lacking an answer-overlap cue."""
from collections import defaultdict, Counter
from itertools import combinations
from pathlib import Path
from scripts.prepare_sentence_context import (
    ROOT, WORK, PUB, prior_index, read, dump, sha, example, eligible,
    exam_rows, statements_from_qamr, phrase_key, digest,
)
from scripts.audit_question_overlap_assignment import dice, runs
from scripts.prepare_multiple_ink_runs import prepared_files

DEST = ROOT/"artifacts/visual_lexical_controls_v1"
PUBLIC = ROOT/"references/evidence/visual_lexical_controls_v1"


def reserve(folder):
    index = prior_index()
    for name in ("confirmation.json","long.json","additional.json"):
        path = WORK/name
        if sha(path) != read(PUB/"corpus.json")["files"][name]:
            raise ValueError("Changed fifth-cohort input")
        for r in read(path):
            index.add(*index.keys(r))
    result = []
    stats = {}
    for label,source in (("statements",statements_from_qamr()),("exam",exam_rows())):
        groups = defaultdict(list)
        optics = {}
        for row in source:
            keys = index.keys(row)
            if not eligible(row) or index.overlaps(keys):
                continue
            groups[row["family"]].append(row)
            for s in (row["q"], row["answers"][0]):
                if s not in optics:
                    seq = index.ink.encode(s)
                    optics[s] = [runs(seq,n) for n in (1,3)]
        families = sorted(groups,key=lambda f:digest("lexical-control-family"+f))
        used = set()
        sizes = Counter()
        for stratum,cap in (("overlap-resistant",48),("remaining",96)):
            for f in families:
                if f in used or sizes[stratum] >= cap:
                    continue
                ordered = sorted(groups[f],key=lambda r:digest("lexical-control-question"+r["q"]))
                for a,b in combinations(ordered,2):
                    if a["blocks"] != b["blocks"] or a["q"]==b["q"] or a["facts"]==b["facts"] or a["answers"][0]==b["answers"][0]:
                        continue
                    if any(index.overlaps(index.keys(r)) for r in (a,b)):
                        continue
                    if stratum=="overlap-resistant":
                        qs=[optics[r["q"]] for r in (a,b)]
                        ys=[optics[r["answers"][0]] for r in (a,b)]
                        if any(dice(qs[0][n],ys[0][n])+dice(qs[1][n],ys[1][n]) >
                               dice(qs[0][n],ys[1][n])+dice(qs[1][n],ys[0][n]) for n in (0,1)):
                            continue
                    used.add(f)
                    sizes[stratum]+=1
                    for r in (a,b):
                        result.append(dict(r,stratum=stratum))
                        index.add(*index.keys(r))
                    break
        stats[label]=dict(eligible_families=len(families),reserved=dict(sizes))
    file=folder/"confirmation.json"
    dump(file,result)
    return dict(files={file.name:sha(file)},families=stats,
                source=sha(WORK/"confirmation.json"),
                protocol=sha(ROOT/"references/visual_lexical_controls_protocol.md"),
                code=sha(Path(__file__)))


if __name__=="__main__":
    PUBLIC.mkdir(exist_ok=True,parents=True)
    prepared_files(DEST,PUBLIC/"reservation.json",{"confirmation.json":("files","confirmation.json")},reserve)
    print(read(PUBLIC/"reservation.json"))
