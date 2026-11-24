"""Test the final form reader on fresh families and unseen question wordings."""

from collections import Counter, defaultdict
from hashlib import sha256
import numpy as np
import torch

from ilm.visual_lm.catalogue_ink import print_forms
from scripts.prepare_visual_story_writer import STAGES, history, dump, sha, MODEL
from scripts.prepare_fact_channel import Ink
from scripts.write_visual_story import StoryWriterAnswer
from scripts.train_balanced_visual_story import PUB, WORK


def questions(c, requested, other):
    zh, en = STAGES[requested]
    zh_alt, en_alt = STAGES[other]
    return {
        "plain-zh": f"请写出{c}的{zh}字形。",
        "plain-en": f"Show the {en} form of {c}.",
        "paraphrase-zh": f"从这一页找出{c}的{zh}写法。",
        "paraphrase-en": f"Which example on this page shows {c} in {en} writing?",
        "requested-first-zh": f"我要{c}的{zh}，请不要给我{zh_alt}。",
        "requested-first-en": f"Give me the {en} form of {c}, not the {en_alt} form.",
        "requested-last-zh": f"不要给我{zh_alt}，请写出{c}的{zh}。",
        "requested-last-en": f"Skip the {en_alt} form. Show me {c} in {en} writing.",
    }


def main():
    torch.set_num_threads(2)
    model = StoryWriterAnswer("balanced", "cpu")
    ink = Ink(model.palette)
    cases = print_forms(ink, history(ink, "sealed", 256))[128:]
    if len(cases) != 128:
        raise ValueError("Need 64 additional sealed families")
    requests = []
    for i in range(0, len(cases), 2):
        a, b = cases[i : i + 2]
        assert a["family"] == b["family"]
        for j, (r, other) in enumerate(((a, b), (b, a))):
            for name, prompt in questions(
                r["char"], r["stage"], other["stage"]
            ).items():
                q = ink.encode(prompt)
                for rev in (0, 1):
                    page = r["p"]
                    if rev:
                        page = [page[0], page[2], page[1]]
                    requests.append((r["family"], j, name, rev, q, page, r["y"][0][0]))
    if ink.bad:
        raise ValueError(f"Missing pictured characters: {sorted(ink.bad)}")
    tiles = ink.array()
    result = defaultdict(Counter)
    complete = defaultdict(list)
    detail = []
    for off in range(0, len(requests), 16):
        batch = requests[off : off + 16]
        pred, _ = model.answer(tiles, [r[4] for r in batch], [r[5] for r in batch])
        for (ids, stop), (family, j, name, rev, _, _, target) in zip(
            pred, batch, strict=True
        ):
            exact = bool(stop and ids == [target])
            complete[name, family].append(exact)
            result[name].update(requests=1, correct=int(exact))
            detail.append(
                dict(
                    family=family,
                    stage_index=j,
                    wording=name,
                    reversed=rev,
                    correct=exact,
                    stopped=stop,
                    target=sha256(tiles[target].tobytes()).hexdigest(),
                    answer=sha256(tiles[np.asarray(ids, int)].tobytes()).hexdigest(),
                )
            )
    for (name, _), bits in complete.items():
        assert len(bits) == 4
        result[name].update(families=1, all_4=int(all(bits)))
    dump(WORK / "paraphrases.json", detail)
    dump(
        PUB / "paraphrases.json",
        dict(
            groups=dict(result),
            weights=sha(MODEL / "balanced/weights.safetensors"),
            response_sha256=sha(WORK / "paraphrases.json"),
        ),
    )
    print(dict(result))


if __name__ == "__main__":
    main()
