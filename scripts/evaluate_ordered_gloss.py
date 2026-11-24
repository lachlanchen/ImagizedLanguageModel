"""Score varied word order using the original physical streams at every source position."""

from collections import Counter
import numpy as np
import torch
from safetensors.torch import load_file

from ilm.visual_lm.gloss_focus import GlossFocus, cell_features, encoding_windows
from ilm.visual_lm.ordered_gloss_question import OrderedGlossFocus
from scripts.asked_gloss_questions import HELD
from scripts.prepare_asked_gloss import WORK, PUBLIC, QUESTIONS, dump, read, sha
from scripts.evaluate_asked_gloss import summary
from scripts.train_asked_gloss import packed_batch
from scripts.train_ordered_gloss import ARMS
from scripts.train_typeface_writer import Face, FACES
from scripts.write_visual_gloss import restore


@torch.inference_mode()
def evaluate():
    previous = read(PUBLIC / "result.json")
    for filename, key in [("emitted.npz", "ink"), ("predictions.json", "predictions")]:
        if sha(PUBLIC / filename) != previous[key]:
            raise ValueError("Changed source outputs")
    rows = read(PUBLIC / "predictions.json")
    face = Face(FACES[0])
    alpha = {face.cell(c): c for c in "abcdefghijklmnopqrstuvwxyz-"}
    with np.load(PUBLIC / "emitted.npz") as z:
        ink = z["ink"]
        offset = z["offsets"]
        end = z["stopped"]
        for r in rows:
            for j, w in zip(r["indices"], r["all_words"], strict=True):
                stream = ink[offset[j] : offset[j + 1]]
                actual = (
                    "".join(alpha.get(x.tobytes(), "�") for x in stream)
                    if end[j]
                    else "<unfinished>"
                )
                assert actual == w
    texts = [q for p in HELD + QUESTIONS for q in p]
    for pair in HELD:

        def local(s):
            cells = [b""] + [face.cell(c) for c in s] + [b""]
            return Counter(zip(cells, cells[1:], cells[2:]))

        assert local(pair[0]) == local(pair[1])
    qp, ql = packed_batch(texts, face, "cpu")
    writer = restore()
    result = {}
    predictions = {}
    for arm in ("answer", *ARMS):
        net = (OrderedGlossFocus if arm.startswith("ordered") else GlossFocus)()
        file = WORK / arm / "focus.safetensors"
        assert sha(file) == read(WORK / arm / "fit.json")["sha256"]
        net.load_state_dict(load_file(str(file)))
        net.eval()
        q = net.read_questions(cell_features(qp), ql)
        positions = []
        maxdiff = np.zeros(4)
        for a in range(0, len(rows), 48):
            batch = rows[a : a + 48]
            raw, lengths = packed_batch([r["page"] for r in batch], face, "cpu")
            ctx = encoding_windows(writer, cell_features(raw))
            logits = net(ctx, lengths, q.expand(len(batch), -1, -1))
            probs = logits.softmax(-1)
            gap = (probs[:, 0:8:2] - probs[:, 1:8:2]).abs().amax((0, 2)).numpy()
            maxdiff = np.maximum(gap, maxdiff)
            positions.extend(logits.argmax(-1).reshape(len(batch), -1, 2).tolist())
        groups = {}
        for book in sorted({r["book"] for r in rows} | {"combined"}):
            ids = [
                j for j, r in enumerate(rows) if book == "combined" or r["book"] == book
            ]
            group = [rows[j] for j in ids]
            score = []
            flags = []
            for k in range(len(texts) // 2):
                pos = [positions[j][k] for j in ids]
                out = [
                    [r["all_words"][i] for i in p]
                    for r, p in zip(group, pos, strict=True)
                ]
                score.append(summary(group, pos, out))
                flags.append(
                    np.array(
                        [
                            r["focus"] == p
                            and all(outcome[h] in r["gloss"][h] for h in range(2))
                            for r, p, outcome in zip(group, pos, out, strict=True)
                        ]
                    )
                )
            groups[book] = dict(
                pages=len(ids),
                by_wording=score,
                cn_both_orders=int((flags[0] & flags[1]).sum()),
                en_both_orders=int((flags[2] & flags[3]).sum()),
                both_languages_all_orders=int(np.logical_and.reduce(flags[:4]).sum()),
            )
        result[arm] = dict(
            books=groups, max_role_swap_probability_difference=maxdiff.tolist()
        )
        predictions[arm] = positions
    path = PUBLIC / "order_positions.npz"
    np.savez_compressed(
        path, **{k: np.asarray(v, dtype=np.int16) for k, v in predictions.items()}
    )
    dump(
        PUBLIC / "order_result.json",
        dict(
            arms=result,
            questions=HELD + QUESTIONS,
            positions=sha(path),
            previous_predictions=previous["predictions"],
            previous_ink=previous["ink"],
        ),
    )
    for arm, d in result.items():
        print(
            arm,
            {k: v for k, v in d["books"]["combined"].items() if k != "by_wording"},
            flush=True,
        )


if __name__ == "__main__":
    torch.set_num_threads(3)
    evaluate()
