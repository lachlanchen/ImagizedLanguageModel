"""Reserve eligible Chinese pages with human Chinese and English MLQA questions."""

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from zipfile import ZipFile
from scripts.interval_teacher_corpus import load_rows
from scripts.train_bilingual_novel_writer import novel_rows
from scripts.train_causal_glyph_writer import record, sha
from scripts.train_typeface_writer import Face, FACES, ROOT

WORK = ROOT / "artifacts/question_evidence_transfer_v1"
PUBLIC = ROOT / "references/evidence/question_evidence_transfer_20261124"
PACKET = WORK / "mlqa.json"
CHECK = PUBLIC / "reservation.json"
HASHES = dict(
    en="24a6514c9a86b07020a3a952738f2df57844960a300c0a5eb72b99ea7182db5e",
    zh="8fad23a8bb5618361209c04313218eff96139682006ff533f8533eb3f2691486",
)


def prepare():
    face = Face(FACES[0])
    archive = ROOT / "data/raw/mlqa/MLQA_V1.zip"
    questions = {}
    with ZipFile(archive) as z:
        for lang, digest in HASHES.items():
            raw = z.read(f"MLQA_V1/dev/dev-context-zh-question-{lang}.json")
            assert sha256(raw).hexdigest() == digest
            questions[lang] = {
                q["id"]: (p["context"], q)
                for a in json.loads(raw)["data"]
                for p in a["paragraphs"]
                for q in p["qas"]
            }
    assert questions["zh"].keys() == questions["en"].keys()
    excluded = Counter()
    pages = {}
    for key, (p, z) in questions["zh"].items():
        same, e = questions["en"][key]
        assert same == p and e["answers"] == z["answers"]
        refs = list(
            dict.fromkeys(
                a["text"]
                for a in z["answers"]
                if a["text"] in p and 0 < len(a["text"]) <= 48
            )
        )
        qs = [e["question"], z["question"]]
        if not refs:
            excluded["no_bounded_accepted_span"] += 1
        elif len(p) + max(map(len, qs)) + 3 > 512:
            excluded["context_length"] += 1
        elif any(
            ord(c) not in face.coverage or not any(face.cell(c))
            for s in (p, *qs, *refs)
            for c in set(s)
            if not c.isspace()
        ):
            excluded["font"] += 1
        else:
            pages.setdefault(p, []).append(dict(id=key, zh=qs[1], en=qs[0], refs=refs))
    # Index every 32-cell fresh window. Train sampling stride 16 covers any
    # aligned continuous match of at least 47 cells.
    width = 32 * 128
    index = {}
    for p in pages:
        bits = face(p)
        for i in range(0, len(bits) - width + 1, 128):
            index.setdefault(bits[i : i + width], set()).add(p)
    blocked = set()
    train = {r.p for r in load_rows() + novel_rows("bilingual")[0]}
    for p in train:
        for i in range(0, len(p) - width + 1, 16 * 128):
            if (s := p[i : i + width]) in index:
                blocked.update(index[s])
    for p in blocked:
        excluded["overlap"] += len(pages.pop(p))
    PUBLIC.mkdir(exist_ok=True)
    WORK.mkdir(exist_ok=True)
    entries = [dict(page=p, questions=qs) for p, qs in pages.items()]
    record(PACKET, entries)
    n = sum(len(q) for q in pages.values())
    assert n + sum(excluded.values()) == len(questions["zh"])
    record(
        CHECK,
        dict(
            archive_sha=sha(archive),
            files=HASHES,
            packet_sha=sha(PACKET),
            source_pairs=len(questions["zh"]),
            pages=len(pages),
            pairs=n,
            excluded=excluded,
            unique_fit_pages=len(train),
            ids=[q["id"] for b in entries for q in b["questions"]],
        ),
    )


if __name__ == "__main__":
    prepare()
