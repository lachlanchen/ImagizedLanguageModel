"""Count shared complete raster sequences between the new narrative tests and the actual fit."""

import numpy as np
from scripts.audit_sentence_context_split import key
from scripts.prepare_fact_channel import Ink
from scripts.prepare_story_answer_fit import WORK, PUB, TRAIN, PARENT, read, dump, sha


def main():
    indices = read(WORK / "indices.json")
    binding = read(PUB / "reservation.json")
    for name, checksum in binding["files"].items():
        if sha(WORK / name) != checksum:
            raise ValueError(name)
    if sha(TRAIN / "training.json") != binding["story_source"]:
        raise ValueError("Training stories changed")
    ink = Ink(np.load(PARENT / "tiles.npy"))
    train = read(TRAIN / "training.json")
    held = read(WORK / "confirmation.json")
    used = [train[i] for i in indices["stories"]]

    def images(r):
        def enc(s):
            return key(ink.encode(s))

        return (
            r["family"],
            enc(r["q"]),
            tuple(map(enc, r["blocks"])),
            set(map(enc, r["answers"])),
        )

    left = [images(r) for r in held]
    right = [images(r) for r in used]
    replay = read(PARENT / "writer.json")
    with np.load(PARENT / "strings.npz") as z:
        ptr, ids = z["ptr"], z["ids"]

    def img(i):
        return key(ids[ptr[i] : ptr[i + 1]])

    for group in indices["replay"].values():
        for i in group:
            q, blocks, _, answers, _, family = replay[i]
            right.append(
                (family, img(q), tuple(map(img, blocks)), set(map(img, answers)))
            )
    labels = ("family", "question", "page")
    overlap = {}
    for i, name in enumerate(labels):
        seen = {r[i] for r in right}
        overlap[name] = sum(r[i] in seen for r in left)
    targets = set.union(*(r[3] for r in right))
    overlap["any_complete_reference"] = sum(bool(r[3] & targets) for r in left)
    dump(
        PUB / "split-audit.json",
        dict(
            held=len(left),
            training=len(right),
            overlap=overlap,
            training_story_families=len({r["family"] for r in used}),
            held_families=len({r["family"] for r in held}),
        ),
    )
    print(overlap)


if __name__ == "__main__":
    main()
