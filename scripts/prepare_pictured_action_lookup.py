"""Reserve matched, genuinely sourced clauses before fitting the pictured query reader."""
from collections import Counter,defaultdict
import hashlib
import re
from scripts.evaluate_visual_clause_roles import annotated_glosses
from scripts.prepare_visual_clause_roles import ROOT, WORK as SYNTAX_WORK, PUBLIC as SYNTAX, read,dump,sha
from scripts.train_typeface_writer import Face,FACES

PUBLIC=ROOT/"references/evidence/pictured_action_lookup_v1"
WORK=ROOT/"artifacts/pictured_action_lookup_v1"
SEED=37127


def key(r):
    return hashlib.sha256(r["page"].encode()).hexdigest()


def pair_pages(rows, face, same_head):
    groups=defaultdict(list)
    for i,r in enumerate(rows):
        head=face.cell(r["page"][r["focus"][0]]) if same_head else b""
        groups[r["book"],head].append(i)
    pairs=[]
    for ids in groups.values():
        sequence=sorted(ids,key=lambda i:key(rows[i]))
        while sequence:
            i=sequence.pop();a,b=(g[0] for g in rows[i]["gloss"])
            for k,j in enumerate(sequence):
                c,d=(g[0] for g in rows[j]["gloss"])
                if a!=c and b!=d and not {a,c}&{b,d}:
                    # The same two clauses and requests also run in inverse order.
                    pairs.append([i,j]);sequence.pop(k);break
    return pairs


def prepare():
    WORK.mkdir(exist_ok=True)
    reg=read(SYNTAX/"split.json")
    assert sha(SYNTAX_WORK/"train.json")==reg["train"]
    train=read(SYNTAX_WORK/"train.json");annotated_glosses(train)
    def valid(r):
        return all(r["gloss"]) and re.fullmatch("[a-z]+",r["gloss"][0][0]) and r["gloss"][0][0]!=r["gloss"][1][0]
    train=[r for r in train if valid(r)]
    counts=Counter(r["gloss"][0][0] for r in train)
    train=[r for r in train if counts[r["gloss"][0][0]]>=3]
    allrows=read(SYNTAX/"readings.json")
    held=[r for r in allrows if r["split"] in ("reserved_work","reserved_section") and valid(r) and counts[r["gloss"][0][0]]>=3]
    assert not {r["page"] for r in train}&{r["page"] for r in held}
    face=Face(FACES[0]);pairs=pair_pages(held,face,False);same=pair_pages(held,face,True)
    dump(WORK/"train.json",train);dump(PUBLIC/"held.json",held);dump(PUBLIC/"pairs.json",dict(overall=pairs,same_head=same))
    dump(PUBLIC/"split.json",dict(train_sha256=sha(WORK/"train.json"),held_sha256=sha(PUBLIC/"held.json"),pairs_sha256=sha(PUBLIC/"pairs.json"),
                                 training=len(train),actions=len({r["gloss"][0][0] for r in train}),held=len(held),study_pairs=len(pairs),same_head_pairs=len(same)))
    print(read(PUBLIC/"split.json"))


if __name__=="__main__":prepare()
