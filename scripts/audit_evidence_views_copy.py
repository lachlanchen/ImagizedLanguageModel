"""Distinguish missing word-boundary phrases from contiguous source pixels."""

from collections import Counter
from hashlib import sha256

import numpy as np

from scripts.evaluate_evidence_views import groups
from scripts.evaluate_visual_discourse import unpack
from scripts.prepare_evidence_views import Alphabet, PUB, TRAIN, WORK, query, read, sha
from scripts.prepare_open_exam import dump
from scripts.score_dialogue_ink_references import official_index


def key(ids):
    return "".join(map(chr, ids))


def presence(text, ids, source, question, visible_ids, norm):
    normalized = norm(text)
    return (
        any(normalized in s for s in (norm(source), norm(question))),
        any(key(ids) in seq for seq in visible_ids),
    )


def train_counts(norm):
    receipt = read(PUB / "corpus.json")
    path = WORK / "train.npz"
    assert sha(path) == receipt["files"]["train.npz"]
    with np.load(path) as data:
        meta, tiles, p, q, y, pp, qp, yp = (
            data[k] for k in ("meta", "tiles", "p", "q", "y", "pptr", "qptr", "yptr")
        )
    alphabet = Alphabet(tiles)
    # Match every eligible example to the exact story, query, answer and stop used in training.
    english = int((meta[:, 3] == 0).sum())
    stories = {key(p[pp[i] : pp[i + 1]]): i for i in range(receipt["english_stories"])}
    accepted = Counter(
        (int(meta[i][0]), key(q[qp[i] : qp[i + 1]]), key(y[yp[i] : yp[i + 1]]))
        for i in range(english)
    )
    counts = Counter()
    for d in read(TRAIN)["data"]:
        story = d["story"]
        pixels = key(alphabet.encode(story))
        idx = stories.get(pixels)
        if idx is None:
            continue
        previous = ""
        normalized_page = norm(story)
        for qr, ar in zip(d["questions"], d["answers"], strict=True):
            raw, answer = qr["input_text"], ar["input_text"]
            current = query(raw, previous) if len(raw) <= 160 else raw
            previous = raw + " " + answer
            ask = key(alphabet.encode(current))
            ink = key(alphabet.encode(answer))
            row = (idx, ask, ink + chr(1))
            if not accepted[row]:
                continue
            accepted[row] -= 1
            counts["verified_turns"] += 1
            ref = norm(answer)
            if (
                len(ref.split()) < 2
                or " " + ref + " " in " " + norm(story + " " + current) + " "
            ):
                continue
            counts["word_boundary_absent"] += 1
            if (
                ref not in normalized_page
                and ref not in norm(current)
                and ink not in pixels
                and ink not in ask
            ):
                counts["both_substring_and_bitmap_absent"] += 1
    assert counts["verified_turns"] == english and not any(accepted.values())
    assert counts["word_boundary_absent"] == receipt["stats"]["absent_multiword"]
    return dict(counts)


def main():
    judge, _, _, dev, _ = official_index()
    norm = judge.normalize_answer
    data = groups(judge, dev)
    a = Alphabet()
    for rows in data.values():
        for row in rows:
            a.encode(row["question"])
            a.encode(row["page"])
    original_tiles = a.array()
    result = {}
    for arm in ("parent", "whole", "highlight"):
        outputs = {
            (v["group"], v["doc"], v["turn"]): v
            for v in unpack(WORK / (arm + "-responses.json.gz"))
        }
        result[arm] = {}
        for group in ("composition", "composition_evidence"):
            counts = Counter()
            hits = []
            for row in data["composition"]:
                v = outputs[group, row["doc"], row["turn"]]
                q, p = row["question"], row["page"]
                visible = [key(a.encode(s)) for s in (q, p)]
                flags = [
                    presence(t, a.encode(t), p, q, visible, norm) for t in row["refs"]
                ]
                strict = not any(x or y for x, y in flags)
                counts["questions"] += 1
                counts["strict_questions"] += strict
                if v["exact"]:
                    ids = np.asarray(v["ids"], int)
                    assert (
                        sha256(original_tiles[ids].tobytes()).hexdigest() == v["sha256"]
                    )
                    literal, pixel = presence(v["text"], ids, p, q, visible, norm)
                    counts["exact"] += 1
                    counts["strict_exact"] += strict
                    counts["exact_replies_copied_pixels"] += pixel
                    counts["exact_replies_present_normalized"] += literal
                    hits.append(
                        dict(
                            doc=row["doc"],
                            turn=row["turn"],
                            copied_pixels=pixel,
                            normalized_substring=literal,
                            strict_question=strict,
                        )
                    )
            result[arm][group] = dict(counts=counts, hits=hits)
    dump(
        PUB / "copy-audit.json",
        dict(
            training=train_counts(norm),
            scores=result,
            definition="Reject a turn if any human answer has a contiguous identical bitmap sequence or normalized substring anywhere in the original story or visible dialogue query. Cropped runs use the same original-story cohort.",
        ),
    )


if __name__ == "__main__":
    main()
