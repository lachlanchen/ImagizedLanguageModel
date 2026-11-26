"""Join trilingual book annotations to exact English and Chinese optical positions."""

from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

from scripts.prepare_visual_library import LIBRARY, RECEIPT, ROOT, read, sha
from scripts.sample_visual_library import library
from scripts.prepare_teacher_translation import dump

RUN = ROOT / "artifacts/library_roles_v1"
PUB = ROOT / "references/evidence/library_roles_v1"
LABELS = {"subject": 1, "predicate": 2, "object": 3}


def aligned_roles(units, expected):
    raw = "".join(part["t"] for part in units)
    if raw.strip() != expected:
        return None
    roles = [
        LABELS.get(part.get("g"), 0) if glyph.isalnum() else 0
        for part in units for glyph in part["t"]
    ]
    start = len(raw) - len(raw.lstrip())
    return roles[start:start + len(expected)]


def collect(split):
    rows = library(split)
    chunks = defaultdict(list)
    for i, row in enumerate(rows):
        chunks[row["work"], row["chunk"]].append((i, row))
    result = np.zeros((len(rows), 2, 96), dtype=np.uint8)
    counts = Counter()
    for (work, chunk), group in chunks.items():
        path = LIBRARY / work / "work/trilingual/interlinear/chunks" / chunk
        paras = {p["id"]: p["units"] for p in read(path)["paragraphs"]}
        for i, row in group:
            unit = paras[row["paragraph"]][row["unit"]]
            for t, language in enumerate(("en", "zh")):
                text = row["text"][t]
                if len(row["cells"][t]) != len(text):
                    raise ValueError("Cell count differs from original rendering")
                roles = aligned_roles(unit[language], text)
                if roles is None:
                    counts[language+"_unaligned"] += 1
                else:
                    result[i,t,:len(roles)] = roles
                    counts[language+"_exact_alignment"] += 1
    eligible = {}
    for t,lang in enumerate(("en","zh")):
        y=result[:,t]
        other=result[:,1-t]
        by_role={}
        mask=np.zeros_like(y,dtype=bool)
        for name,r in LABELS.items():
            matched=(y==r) & (other==r).any(axis=1)[:,None]
            mask |= matched
            by_role[name]=int(matched.sum())
        eligible[lang]=dict(positions=int(mask.sum()),sentences=int(mask.any(axis=1).sum()),roles=by_role)
    return result,dict(rows=len(rows),alignments=counts,eligible_target=eligible)


def prepare():
    RUN.mkdir(parents=True,exist_ok=True)
    PUB.mkdir(parents=True,exist_ok=True)
    metadata={}
    files={}
    for split in ("train","dev","test"):
        a,metadata[split]=collect(split)
        p=RUN/(split+".npy")
        np.save(p,a)
        files[p.name]=sha(p)
    dump(PUB/"alignment.json",dict(labels=LABELS,splits=metadata,files=files,
         corpus=sha(RECEIPT),program=sha(Path(__file__)),scope="Supplied grammatical annotations, no independent linguistic validation. Exact character alignment only. Zero marks unsupported, unknown or non-alphanumeric glyphs. Evaluation roles are diagnostic only."))
    print(metadata,flush=True)


if __name__=="__main__":
    prepare()
