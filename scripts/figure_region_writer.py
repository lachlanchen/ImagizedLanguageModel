"""Plot identical pictured questions and the measured reply before and after a fact change."""

from hashlib import sha256
import numpy as np
from scripts.prepare_region_writer import MODEL, WORK, PUB, Ink
from scripts.prepare_open_exam import read, dump
from scripts.evaluate_visual_discourse import unpack, draw
from scripts.evaluate_region_writer import changed
from scripts.figure_recurrent_evidence import Plate, MUTED, TEAL


def main():
    original = read(WORK / "dev-source.json")
    edits = changed(original)
    replies = unpack(WORK / "selected-predictions.json.gz")
    choices = []
    for t in ("statements", "catalogue"):
        matches = []
        for i, r in enumerate(edits):
            if r["task"] != t:
                continue
            src = replies["original"][r["original"]]
            new = replies["edited"][i]
            passes = src["grounded_proxy"] and new["exact_old_new"]
            if passes == (t == "statements"):
                matches.append(i)
        choices.append(
            matches[0]
            if matches
            else next(i for i, r in enumerate(edits) if r["task"] == t)
        )
    assets = PUB / "plates"
    assets.mkdir(parents=True, exist_ok=True)
    plate = Plate("Figure_163_writing_after_a_fact_changes", assets)
    examples = []
    for j, i in enumerate(choices):
        e = edits[i]
        r = original[e["original"]]
        before = replies["original"][e["original"]]
        after = replies["edited"][i]
        ink = Ink(np.load(MODEL / "palette.npy"))
        case = MODEL / "examples" / str(j + 1)
        case.mkdir(parents=True, exist_ok=True)
        q = ink.encode(r["q"])
        f = ink.encode(e["old"] + "   →   " + e["new"])
        tiles = ink.array()
        for k, ids in (("question", q), ("facts", f)):
            draw(case, k, tiles[ids])
        for label, row in (("original", before), ("edited", after)):
            raw = bytes.fromhex(row["ink_hex"])
            assert sha256(raw).hexdigest() == row["sha256"]
            pix = np.frombuffer(raw, np.uint8).reshape(-1, 128)
            draw(case, label, pix)
        top = j * 135
        plate.label(4, top + 1, "ab"[j], 12, True)
        plate.label(
            23,
            top + 1,
            ("Natural reading question" if j == 0 else "Chinese catalogue question")
            + " with a changed source fact",
            10,
            True,
        )
        fields = [
            ("Question", "question", len(q)),
            ("Source edit", "facts", len(f)),
            ("First reply", "original", before["cells"]),
            ("After edit", "edited", after["cells"]),
        ]
        for k, (title, name, n) in enumerate(fields):
            yy = top + (21, 46, 69, 99)[k]
            plate.label(8, yy + 2, title, 8, True, TEAL if k == 3 else MUTED)
            if not n:
                plate.label(87, yy + 2, "[empty]", 8)
                continue
            pieces = 1 if n <= 75 else 2
            for z in range(pieces):
                plate.strip(
                    f"{j}-{name}-{z}",
                    case / (name + ".png"),
                    n,
                    87,
                    yy + 12 * z,
                    444,
                    16 if pieces == 1 else 12,
                    n * z // pieces,
                    n * (z + 1) // pieces,
                )
        rec = dict(
            task=e["task"],
            original_family=r["family"],
            original_index=e["original"],
            edit_index=i,
            old=e["old"],
            new=e["new"],
            question=r["q"],
            references=r["answers"],
            response=before["text"],
            after_edit=after["text"],
            complete=before["stop"],
            edited_complete=after["stop"],
            source_sha256=before["sha256"],
            edited_sha256=after["sha256"],
            selected=before["selected"],
            edited_selected=after["selected"],
        )
        dump(case / "case.json", rec)
        examples.append(rec)
        for tag, doc in (("first-input", r), ("edited-input", e)):
            folder = case / tag
            folder.mkdir(exist_ok=True)

            def emit(k, s):
                ids = ink.encode(s)
                draw(folder, k, ink.array()[ids])
                return dict(png=k + ".png", cells=len(ids))

            dump(
                folder / "input.json",
                dict(
                    question=emit("question", doc["q"]),
                    regions=[
                        emit("region-" + str(k), s) for k, s in enumerate(doc["blocks"])
                    ],
                ),
            )
    plate.rule(131)
    edit_panel = plate.render()
    complete = Plate("Figure_164_sentences_from_pictured_questions", assets)
    exact_cases = []
    for j, task in enumerate(("statements", "catalogue")):
        for i, (row, response) in enumerate(
            zip(original, replies["original"], strict=True)
        ):
            if row["task"] != task or not response["stop"] or response["visible_reply"]:
                continue
            ink = Ink(np.load(MODEL / "palette.npy"))

            def form(s):
                return "".join(map(chr, ink.encode(s)))

            sources = [form(v) for v in [row["q"], *row["blocks"]]]
            if any(form(a) in src for a in row["answers"] for src in sources):
                continue
            raw = bytes.fromhex(response["ink_hex"])
            assert sha256(raw).hexdigest() == response["sha256"]
            if not any(
                ink.array()[ink.encode(a)].tobytes() == raw for a in row["answers"]
            ):
                continue
            break
        else:
            raise ValueError("No complete held answer for " + task)
        folder = MODEL / "examples" / ("exact-" + task)
        folder.mkdir(parents=True, exist_ok=True)

        def grid(k, s):
            ids = ink.encode(s)
            draw(folder, k, ink.array()[ids])
            return dict(png=k + ".png", cells=len(ids))

        q = grid("question", row["q"])
        facts = grid("fact", next(iter(row["facts"].values())))
        dump(
            folder / "input.json",
            dict(
                question=q,
                regions=[
                    grid("region-" + str(k), s) for k, s in enumerate(row["blocks"])
                ],
            ),
        )
        pixels = np.frombuffer(raw, np.uint8).reshape(-1, 128)
        draw(folder, "answer", pixels)
        top = 135 * j
        complete.label(5, top + 1, "ab"[j], 12, True)
        complete.label(
            23,
            top + 1,
            (
                "Answering a question about a historical figure"
                if j == 0
                else "Giving the address of a hotel"
            ),
            10,
            True,
        )
        for k, title, n, y in (
            ("question", "Question", q["cells"], 22),
            ("fact", "Source fact", facts["cells"], 61),
            ("answer", "Written answer", len(pixels), 86),
        ):
            complete.label(
                8, top + y + 2, title, 8, True, TEAL if k == "answer" else MUTED
            )
            pieces = 1 if n <= 64 else 2
            for z in range(pieces):
                complete.strip(
                    str(j) + k + str(z),
                    folder / (k + ".png"),
                    n,
                    93,
                    top + y + 16 * z,
                    438,
                    16,
                    n * z // pieces,
                    n * (z + 1) // pieces,
                )
        exact_cases.append(
            dict(
                task=task,
                index=i,
                question=row["q"],
                answer=response["text"],
                sha256=response["sha256"],
                source_fact=row["facts"],
            )
        )
    complete.rule(131)
    dump(PUB / "exact-cases.json", exact_cases)
    dump(PUB / "plates.json", [complete.render(), edit_panel])
    dump(PUB / "figure-cases.json", examples)


if __name__ == "__main__":
    main()
