"""Prepare same-passage Chinese question pairs for which raw question/answer overlap cannot win."""
from collections import Counter, defaultdict
from itertools import combinations, product
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np

from scripts.prepare_pictured_entry_answers import load, sha, write_json
from scripts.train_paired_passage import ROOT, unpack_sequences

SOURCE = ROOT / "data/traditional_passage_evidence_v1"
DEST = ROOT / "data/shared_passage_overlap_control_v1"


def overlap(a, b):
    return len(a & b) / len(a | b)


def collect(name):
    path = SOURCE / name
    ps = unpack_sequences(path/"passages.npz")
    qs = unpack_sequences(path/"questions.npz")
    with np.load(path/"evidence.npz") as z:
        groups, offsets, spans = [z[k] for k in ("group","offsets","spans")]
    members = defaultdict(list)
    alternatives, selected = [], []
    for i, group in enumerate(groups):
        members[int(group)].append(i)
        seqs = {}
        for start, stop in spans[offsets[i]:offsets[i+1]]:
            glyphs = tuple(ps[group][start:stop])
            assert len(glyphs)>0
            seqs.setdefault(glyphs, (int(start),int(stop)))
        alternatives.append(seqs)
    stats = Counter()
    for group, questions in members.items():
        for i,j in combinations(questions,2):
            stats["pairs_considered"] += 1
            if np.array_equal(qs[i],qs[j]) or alternatives[i].keys() & alternatives[j].keys():
                continue
            # Require every combination of recorded answer variants to tie or favour the sibling
            # for BOTH questions. A score based only on shared shapes cannot identify both.
            qi, qj = set(qs[i]), set(qs[j])
            if any(overlap(qi,set(a))>overlap(qi,set(b)) or overlap(qj,set(b))>overlap(qj,set(a))
                   for a,b in product(alternatives[i],alternatives[j])):
                continue
            selected.append((group,i,j,*next(iter(alternatives[i].values())),*next(iter(alternatives[j].values()))))
    array = np.asarray(selected,np.int32).reshape(-1,7)
    stats.update(dict(records=len(array),passages=len(set(array[:,0].tolist())),
                 distinct_questions=len(set(array[:,1:3].ravel().tolist()))))
    return array,stats


def main():
    if DEST.exists():
        raise FileExistsError(DEST)
    receipt = load(SOURCE/"preparation.json")
    with TemporaryDirectory(dir=DEST.parent) as folder:
        out = Path(folder)/"pairs"
        out.mkdir()
        info = {}
        for split in ("train","development"):
            for name,expected in receipt[split]["files"].items():
                assert sha(SOURCE/split/name)==expected
            pairs, info[split] = collect(split)
            np.save(out/(split+".npy"),pairs)
        record = dict(groups=info,columns=["passage","question_a","question_b","begin_a","end_a","begin_b","end_b"],
                 parent_sha256=sha(SOURCE/"preparation.json"),source_sha256=sha(__file__),
                 files={p.name:sha(p) for p in out.glob("*.npy")})
        write_json(out/"receipt.json",record)
        out.rename(DEST)
        print(record)


if __name__ == "__main__":
    main()
