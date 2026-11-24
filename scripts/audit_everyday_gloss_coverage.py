"""Count which reserved glosses were present for the same printed source glyph."""

from collections import Counter, defaultdict
from scripts.evaluate_everyday_gloss import DEST
from scripts.prepare_asked_gloss import ROOT, PRIOR, read, dump, sha
from scripts.train_typeface_writer import Face, FACES


def main():
    meta = read(ROOT / "references/evidence/visual_word_senses_20261124/split.json")
    train = PRIOR / "train.json"
    assert sha(train) == meta["data"]["train"]["meta"]
    face = Face(FACES[0])
    counts = defaultdict(Counter)
    for _, _, _, _, glyph, _, word in read(train):
        counts[face.cell(glyph)][word] += 1
    result, stats = [], Counter()
    for r in read(DEST / "positions.json"):
        if r["book"] not in {"zhuangzi","hanfeizi"}:
            continue
        row = []
        for k, i in enumerate(r["focus"]):
            glyph = r["page"][i]
            words = counts[face.cell(glyph)]
            allowed = r["gloss"][k]
            ink = r["words"][i]
            support = sum(words[w] for w in allowed)
            row.append(dict(glyph=glyph, accepted=allowed, output=ink, accepted_occurrences=support,
                            training_occurrences=words.total(), common=words.most_common(5)))
            stats["targets"] += 1
            stats["lexically_supported"] += bool(support)
            stats["correct_at_location"] += ink in allowed
            stats["correct_with_support"] += bool(support) and ink in allowed
        stats["pages"] += 1
        stats["both_supported"] += all(t["accepted_occurrences"] for t in row)
        result.append(dict(page=r["page"], targets=row))
    dump(DEST / "lexical_coverage.json", dict(summary=dict(stats), rows=result))


if __name__ == "__main__":
    main()
