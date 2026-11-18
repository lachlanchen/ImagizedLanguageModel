"""Screen real, chapter-held-out classical clauses for contextual span learning."""

import hashlib
from pathlib import Path
import tarfile
from collections import Counter, defaultdict

from scripts.prepare_contextual_phrase_realization import (
    BASE, EVENT, INITIAL, OLD, OUT as PREVIOUS, ROOT, Rasterizer, clauses, load,
    order, pairs, read_rows, records, sha, write_new,
)
from scripts.prepare_visual_boundary_candidates import OTHER, additional

OUT = ROOT / "artifacts/visual_phrase_boundary_v1"
EVIDENCE = ROOT / "references/evidence/phrase_boundary_20261119"
INVENTORY = ROOT / "references/evidence/visual_boundary_candidates_20261119.json"

# A reader examined the source and entire paired translation before any fit.
# Retain identifiable senses even where the edition supplies an omitted name.
REJECT = {
    834: "Bare 谢曰 supplies neither the person nor a distinguishing cue.",
    837: "The speaker and reporting verb do not distinguish the two readings.",
    842: "The speaker's name alone does not disambiguate the reported speech.",
    843: "Writing a letter to someone is consistent with either reading.",
    852: "The target name and wording of the salute are corrupt.",
    854: "The office and beneficiary of thanks are supplied from outside the clause.",
    856: "The object and the reference to water require the preceding clause.",
    858: "The damaged 鄂 gloss and addition of pretence are unreliable.",
    864: "Bare 谢之 allows either sense.",
    865: "The title and reporting verb allow either sense.",
    866: "本州岛 is corrupt and the agency of nomination in the target is unclear.",
    867: "Coming to thank or to apologize are both plausible.",
    870: "The choice of gratitude is not established by 陈 and 出.",
    871: "Taking someone's hand occurs with either speech act.",
    876: "The name and object pronoun do not disambiguate the speech act.",
    878: "Summoning Wei and speaking permits either sense.",
    880: "Broken source alignment and garbled personal names in the target.",
    883: "The omitted sword is supplied and the addressee's name is truncated.",
    884: "Respect and retirement do not determine gratitude.",
}


def prepare():
    OUT.mkdir()
    rows = load(INVENTORY)["records"]
    audit = []
    for i, r in enumerate(rows):
        if r["predicate"] in OTHER:
            audit.append(dict(index=i, source=r["source"], target=r["target"],
                              sense=r["sense"], admit=i not in REJECT, reason=REJECT.get(i)))
    assert len(audit) == 56 and {r["index"] for r in audit} >= REJECT.keys()
    write_new(OUT / "training_audit.json", audit)
    forbidden, chapters = set(), set()

    def remember(row):
        chapters.add(row.get("chapter") or row.get("group"))
        for key in ("source", "new", "old", "target", "reference"):
            if isinstance(row.get(key), str):
                forbidden.update(clauses(row[key]))
                if "a" in row and "b" in row:
                    forbidden.update(clauses(row[key].translate(str.maketrans(
                        {row["a"]: row["b"], row["b"]: row["a"]}))))
    for split in ("train", "heldout"):
        for row in read_rows(split):
            remember(row)
    for name, roster, _ in OLD:
        for row in load(ROOT / "artifacts" / name / (roster + ".json")):
            remember(row)
    for row in load(ROOT / "references/evidence/block_translation_20261118/calibration_mapping.json"):
        chapters.add(row["chapter"])
    for folder, names in [(EVENT, ("train", "roles", "broad")), (INITIAL, ("train", "proposed", "test")),
                          (PREVIOUS, ("train", "proposed", "test"))]:
        for name in names:
            for row in load(folder / (name + ".json")):
                remember(row)
    for row in rows:
        remember(row)
    render = Rasterizer()
    def digest(text):
        return hashlib.sha256(b"".join(map(render.tile, text))).digest()
    old_ink = set()
    for s in forbidden:
        if len(s) <= 55:
            try:
                old_ink.add(digest(s))
            except ValueError:
                continue
    archive = BASE / "archive.tar.gz"
    assert sha(archive) == load(INITIAL / "preparation.json")["archive"]
    pool = defaultdict(list)
    with tarfile.open(archive, "r|gz") as tar:
        for entry in tar:
            path = Path(entry.name)
            if len(path.parts) < 4 or path.parts[1] != "双语数据" or path.name != "bitext.txt":
                continue
            ch = "/".join(path.parts[2:-1])
            if ch in chapters:
                continue
            for n, (x, y) in enumerate(pairs(tar.extractfile(entry).read())):
                for row in [*records(x, y), *additional([dict(source=x, new=y, group=ch)])]:
                    s, t = row["source"], row["target"]
                    if not (3 <= len(s) <= 32 and 2 <= len(t) <= 55):
                        continue
                    try:
                        if {digest(s), digest(t)} & old_ink:
                            continue
                    except ValueError:
                        continue
                    pool[row["gloss"]].append(dict(row, chapter=ch, ordinal=n))
    candidates = []
    used_ch, used_ink = set(), set()
    for sense in sorted(pool, key=lambda g: (len(pool[g]), g)):
        taken = 0
        for row in sorted(pool[sense], key=lambda r: ("相去" not in r["source"],
                          order(f"boundary-1/{r['chapter']}/{r['ordinal']}/{r['source']}"))):
            keys = {digest(row[k]) for k in ("source", "target")}
            if row["chapter"] in used_ch or keys & used_ink:
                continue
            candidates.append(row)
            used_ch.add(row["chapter"])
            used_ink.update(keys)
            taken += 1
            if taken == 12:
                break
    write_new(OUT / "proposed.json", candidates)
    write_new(OUT / "screen.json", dict(inventory_sha=sha(INVENTORY), archive_sha=sha(archive),
        prior_chapters=len(chapters), prior_bitmaps=len(old_ink),
        eligible={g: len(v) for g, v in pool.items()},
        screened=dict(Counter(r["gloss"] for r in candidates)),
        training_audit_sha=sha(OUT / "training_audit.json"), proposed_sha=sha(OUT / "proposed.json")))
    for i, r in enumerate(candidates):
        print(i, r["gloss"], r["source"], "|", r["target"])


if __name__ == "__main__":
    prepare()
