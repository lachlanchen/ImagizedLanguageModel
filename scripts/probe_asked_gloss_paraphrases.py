"""Read frozen everyday paraphrases without supplying the trained grammar terms."""

from collections import defaultdict
import torch
from scripts.asked_gloss_questions import HELD
from scripts.prepare_asked_gloss import PUBLIC, WORK, read, dump, sha
from scripts.train_asked_gloss import packed_batch
from scripts.train_typeface_writer import Face, FACES
from scripts.write_visual_gloss import restore
from scripts.answer_visual_gloss import load_focus, RELEASE
from ilm.visual_lm.gloss_focus import cell_features, encoding_windows

PARAPHRASES = [
    ("句子說做了什麼，用英文答一個詞", "動作落在哪個人或東西上，用英文答一個詞"),
    (
        "What is being done here? Give one English word.",
        "Who or what is acted on here? Give one English word.",
    ),
    ("這句表達什麼行為", "這個行為涉及哪個對象"),
    ("Name the action in English", "Name what the action affects in English"),
]


@torch.inference_mode()
def probe():
    registry = read(WORK / "paraphrase_registration.json")
    assert sha(PUBLIC / "order_result.json") == registry["prior"] and registry[
        "questions"
    ] == [list(p) for p in PARAPHRASES]
    face = Face(FACES[0])
    model = load_focus(RELEASE, "cpu")
    writer = restore()
    rows = read(PUBLIC / "predictions.json")
    qp, ql = packed_batch(
        [s for pair in PARAPHRASES + [HELD[0]] for s in pair], face, "cpu"
    )
    q = model.read_questions(cell_features(qp), ql)
    result = defaultdict(list)
    saved = []
    for a in range(0, len(rows), 48):
        batch = rows[a : a + 48]
        pages, lens = packed_batch([r["page"] for r in batch], face, "cpu")
        ctx = encoding_windows(writer, cell_features(pages))
        pred = (
            model(ctx, lens, q.expand(len(batch), -1, -1))
            .argmax(-1)
            .reshape(len(batch), -1, 2)
            .tolist()
        )
        saved.extend(pred)
        for r, selections in zip(batch, pred, strict=True):
            scores = [
                pair == r["focus"]
                and all(r["all_words"][i] in r["gloss"][k] for k, i in enumerate(pair))
                for pair in selections
            ]
            for book in [
                r["book"],
                "combined",
                "object_first" if r["focus"][1] < r["focus"][0] else "verb_first",
            ]:
                result[book].append(scores)
    dump(
        PUBLIC / "paraphrases.json",
        dict(
            questions=PARAPHRASES,
            prior=registry["prior"],
            positions=saved,
            counts={
                k: dict(pages=len(v), both=[sum(col) for col in zip(*v, strict=True)])
                for k, v in result.items()
            },
        ),
    )


if __name__ == "__main__":
    torch.set_num_threads(3)
    probe()
