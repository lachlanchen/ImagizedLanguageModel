"""Reserve the next unused CoQA stories under the original global exclusion rule."""

import json
import gzip
import zipfile
from scripts.prepare_compose_dialogue import (
    ARCHIVE,
    ARCHIVE_SHA,
    ROOT,
    SOURCES,
    reserve_dialogue,
    kind,
    prepared,
)
from scripts.dialogue_teacher_stories import key
from scripts.dialogue_ink_data import read_data
from scripts.prepare_dialogue_ink import dump, sha
from scripts.score_dialogue_ink_references import official_index
from collections import Counter

WORK = ROOT / "artifacts/word_canvas_v1"
PUB = ROOT / "references/evidence/word_canvas_v1"
MODEL = ROOT / "models/word_canvas"


def eligible_stories(start, end):
    train = ROOT / "data/raw/coqa/train.parquet"
    if sha(train) != SOURCES["train"] or sha(ARCHIVE) != ARCHIVE_SHA:
        raise ValueError("Different source archive")
    reg = read_data()
    excluded = {key(reg["pages"][r[1]]) for r in reg["rows"]}
    official = official_index()[3]
    excluded.update(key(s["story"]) for s in json.loads(official.read_text())["data"])
    with zipfile.ZipFile(ARCHIVE) as z:
        for partition in ("train", "valid"):
            data = json.loads(z.read("MultiSpanQA_data/" + partition + ".json"))["data"]
            excluded.update(key(" ".join(r["context"])) for r in data)
    return reserve_dialogue(train, excluded, start, end)


def prepare():
    train = ROOT / "data/raw/coqa/train.parquet"
    fresh, n = eligible_stories(200, 400)
    earlier = prepared()["fresh"]
    if {key(s) for s in fresh["pages"].values()} & {
        key(s) for s in earlier["pages"].values()
    }:
        raise ValueError("Overlapping previous study")
    counts = Counter(
        kind(r[5], (fresh["pages"][r[1]], r[3], r[4])) for r in fresh["rows"]
    )
    WORK.mkdir(parents=True, exist_ok=True)
    PUB.mkdir(parents=True, exist_ok=True)
    f = WORK / "reserved.json.gz"
    f.write_bytes(
        gzip.compress(json.dumps(fresh, separators=(",", ":")).encode(), mtime=0)
    )
    manifest = dict(
        source=sha(train),
        list_archive=sha(ARCHIVE),
        stories=list(fresh["pages"]),
        turns=len(fresh["rows"]),
        eligible_pool=n,
        groups=dict(counts),
        sha256=sha(f),
    )
    dest = PUB / "reservation.json"
    if dest.exists() and json.loads(dest.read_text()) != manifest:
        raise ValueError("Different reservation")
    dump(dest, manifest)
    print(n, len(fresh["pages"]), manifest["turns"], counts, flush=True)


def reserved():
    f = WORK / "reserved.json.gz"
    if sha(f) != json.loads((PUB / "reservation.json").read_text())["sha256"]:
        raise ValueError("Wrong reserved cohort")
    return json.loads(gzip.decompress(f.read_bytes()))


if __name__ == "__main__":
    prepare()
