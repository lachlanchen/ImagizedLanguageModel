"""Make Figure 120 from saved neural images and check its examples on the released CPU path."""

import numpy as np
import torch
from ilm.visual_lm.attested_gloss import AttestedGlosses
from ilm.visual_lm.causal_glyph_writer import write_strip
from scripts.asked_gloss_questions import HELD
from scripts.answer_visual_gloss import RELEASE, answer, load_focus
from scripts.prepare_asked_gloss import PUBLIC, read, dump, sha
from scripts.train_typeface_writer import Face, FACES
from scripts.write_visual_gloss import restore


def main():
    root = PUBLIC
    order = read(root / "order_result.json")["arms"]
    bank_result = read(root / "attested_result.json")
    raw = np.load(root / "emitted.npz")
    pos = np.load(root / "order_positions.npz")["ordered_contrast"]
    rows = read(root / "predictions.json")
    writer = restore()
    reader = load_focus(RELEASE, "cpu")
    face = Face(FACES[0])
    lex = AttestedGlosses(RELEASE / "attested.npz", writer.stop, "cpu")

    def pic(s):
        return np.frombuffer(face(s), np.uint8).copy().reshape(-1, 128)

    qs = HELD[0]
    for k, q in enumerate(qs):
        write_strip(pic(q), root / f"order_question{k}.png")
    examples = []
    changed = {(c["row"], c["cell"]): c for c in bank_result["changes"]}
    for name, page, roles in [
        ("hold", "執大象", [0, 1]),
        ("male", "知其雄", [1]),
        ("female", "守其雌", [1]),
    ]:
        j = next(
            i
            for i, r in enumerate(rows)
            if r["book"] == "Daodejing" and r["page"] == page
        )
        r = rows[j]
        source = pic(page)
        write_strip(source, root / f"{name}_page.png")
        for k in roles:
            i = int(pos[j, 0, k])
            idx = r["indices"][i]
            a, b = raw["offsets"][idx : idx + 2]
            original = raw["ink"][a:b]
            free, end, focus = answer(writer, reader, source, pic(qs[k]))
            assert (
                end == bool(raw["stopped"][idx])
                and focus == i
                and np.array_equal(free, original)
            )
            write_strip(free, root / f"{name}_free{k}.png")
            c = changed.get((j, i))
            values = dict(
                page=page,
                question=qs[k],
                cell=i,
                role=k,
                free=r["all_words"][i],
                reference=r["gloss"][k],
            )
            if c:
                actual, ended, focus = answer(writer, reader, source, pic(qs[k]), lex)
                assert (
                    ended and focus == i and actual.tobytes() == bytes.fromhex(c["ink"])
                )
                write_strip(actual, root / f"{name}_bank{k}.png")
                values["attested"] = c["new"]
            examples.append(values)
    books = [
        ("KR1h0004", "Analects"),
        ("KR4a0001", "Chu ci"),
        ("KR6f0082", "Amitābha"),
        ("KR6c0127", "Prajñā text"),
        ("Daodejing", "Laozi"),
    ]
    table = [
        ["Held work", "Pages", "Eight requests", "With gloss bank", "Supplied focus¹"]
    ]
    for book, name in books:
        d = bank_result["books"][book]
        table.append(
            [
                name,
                str(d["pages"]),
                str(d["free_all_eight"]),
                str(d["all_eight"]),
                str(d["known_at_focus"]),
            ]
        )
    series = []
    for arm in ["pooled_contrast", "ordered_contrast", "ordered_exchanged"]:
        d = order[arm]["books"]["combined"]
        series.append(
            [
                d[k]
                for k in (
                    "cn_both_orders",
                    "en_both_orders",
                    "both_languages_all_orders",
                )
            ]
        )
    dump(
        root / "figure.json",
        dict(
            table=table,
            series=series,
            examples=examples,
            free_and_bank_replay=True,
            order=sha(root / "order_result.json"),
        ),
    )
    raw.close()


if __name__ == "__main__":
    torch.set_num_threads(2)
    main()
