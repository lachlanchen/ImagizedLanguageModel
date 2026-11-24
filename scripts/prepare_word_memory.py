"""Reserve the third disjoint group of long CoQA stories for the word-image memory."""

import gzip
import json
from collections import Counter
from scripts.prepare_dialogue_ink import ROOT, dump, sha
from scripts.prepare_word_canvas import eligible_stories, reserved as preceding
from scripts.prepare_compose_dialogue import prepared, SOURCES, ARCHIVE_SHA, kind
from scripts.dialogue_teacher_stories import key

WORK = ROOT / "artifacts/word_memory_v1"
PUB = ROOT / "references/evidence/word_memory_v1"
MODEL = ROOT / "models/word_memory"


def write_reservation(start, end, old, work, pub):
    fresh, n = eligible_stories(start, end)
    if {key(p) for p in fresh["pages"].values()} & {
        key(p) for data in old for p in data["pages"].values()
    }:
        raise ValueError("Repeated story")
    work.mkdir(exist_ok=True, parents=True)
    pub.mkdir(exist_ok=True, parents=True)
    grouped = Counter(
        kind(r[5], (fresh["pages"][r[1]], r[3], r[4])) for r in fresh["rows"]
    )
    packed = gzip.compress(json.dumps(fresh, separators=(",", ":")).encode(), mtime=0)
    f = work / "reserved.json.gz"
    if f.exists() and f.read_bytes() != packed:
        raise ValueError("Reservation changed")
    f.write_bytes(packed)
    manifest = dict(
        source=SOURCES["train"],
        list_archive=ARCHIVE_SHA,
        eligible=n,
        range=[start, end],
        stories=list(fresh["pages"]),
        groups=dict(grouped),
        turns=len(fresh["rows"]),
        sha256=sha(f),
    )
    path = pub / "reservation.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError("Reservation manifest changed")
    dump(path, manifest)
    print(len(fresh["pages"]), grouped)


def reserve():
    write_reservation(400, 547, [preceding(), prepared()["fresh"]], WORK, PUB)


def reserved(work=WORK, pub=PUB):
    f = work / "reserved.json.gz"
    if sha(f) != json.loads((pub / "reservation.json").read_text())["sha256"]:
        raise ValueError("Wrong reserved cohort")
    return json.loads(gzip.decompress(f.read_bytes()))


if __name__ == "__main__":
    reserve()
