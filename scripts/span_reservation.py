"""Screen development/training records against the frozen DRCD confirmation reservation."""

from functools import lru_cache
import hashlib
import json
from pathlib import Path
import unicodedata

from opencc import OpenCC

ROOT = Path(__file__).resolve().parents[1]
POOL = ROOT / "artifacts/drcd_confirmation_reservation_v1/evaluation_only_pool.jsonl"
convert = OpenCC("t2s")


@lru_cache(maxsize=65536)
def normalized(s):
    return "".join(convert.convert(unicodedata.normalize("NFKC", s)).casefold().split())


class Reservation:
    def __init__(self):
        report = json.loads((POOL.parent / "report.json").read_text())
        raw = POOL.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest != report["pool_sha256"]:
            raise ValueError("Confirmation pool fingerprint changed")
        self.by_question = {}
        self.by_paragraph = {}
        self.titles = set()
        self.contexts = set()
        self.windows = set()
        for line in raw.splitlines():
            v = json.loads(line)
            p = v["paragraph_id"]
            self.by_paragraph[p] = v["article_id"]
            self.by_question.update({q["id"]: p for q in v["questions"]})
            self.titles.add(normalized(v["title"]))
            text = normalized(v["context"])
            self.contexts.add(text)
            self.windows.update(text[i : i + 64] for i in range(len(text) - 63))
        if len(self.by_question) != 3485 or len(self.by_paragraph) != 998:
            raise ValueError("Unexpected reservation dimensions")

        self.named_sources = {}
        for fname in ("DRCD_training.json", "DRCD_dev.json"):
            for a in json.loads((ROOT / "data/raw/drcd" / fname).read_text())["data"]:
                for p in a["paragraphs"]:
                    for q in p["qas"]:
                        self.named_sources[q["id"]] = (
                            fname,
                            normalized(a["title"]) in self.titles,
                        )

    @lru_cache(maxsize=65536)
    def overlaps(self, text):
        s = normalized(text)
        return s in self.contexts or any(
            s[i : i + 64] in self.windows for i in range(len(s) - 63)
        )

    def rejects(self, row):
        if row.get("group") == "drcd":
            key = row["key"]
            info = self.named_sources.get(key)
            if key in self.by_question or info is None or info[1]:
                return True
        return self.overlaps(row["p"])

    def filter_development(self, articles):
        for article in articles:
            if normalized(article["title"]) in self.titles:
                continue
            for p in article["paragraphs"]:
                if p["id"] in self.by_paragraph or self.overlaps(p["context"]):
                    continue
                if any(q["id"] in self.by_question for q in p["qas"]):
                    continue
                yield p


def development_path():
    return ROOT / "data/raw/drcd/DRCD_dev.json"


@lru_cache(maxsize=1)
def reservation():
    return Reservation()


def screened(rows, *, eligible_only=False):
    index = reservation()
    result = [row for row in rows if not index.rejects(row)]
    if not eligible_only and len(result) != len(rows):
        raise ValueError(
            "Archived records intersect the DRCD reservation. Use the screened development comparison."
        )
    return result
