"""Audit complete contrasts and run the independently reserved alternate-name requests."""

from collections import defaultdict
import re
import numpy as np
from PIL import Image
import torch
from ilm.visual_lm.causal_glyph_writer import write_strip
from scripts.answer_pictured_action import answer, restore_lookup, read_lines
from scripts.answer_visual_gloss import load_focus
from scripts.pictured_action_paraphrases import PARAPHRASES
from scripts.pictured_action_questions import HELD_FRAMES
from scripts.prepare_pictured_action_lookup import (
    ALIGNED,
    ROOT,
    PUBLIC,
    dump,
    read,
    sha,
)
from scripts.train_typeface_writer import Face, FACES
from scripts.write_visual_gloss import restore, load_adapter


def alternate(actions, objects):
    if not all(a in PARAPHRASES for a in actions):
        return None
    terms = [PARAPHRASES[a] for a in actions]
    forbidden = set(actions + objects)
    if terms[0] == terms[1] or any(
        forbidden & set(re.findall(r"[a-z]+", s)) for s in terms
    ):
        return None
    return terms


def main():
    torch.set_num_threads(2)
    rows, line, pairs, result = [
        read(PUBLIC / (name + ".json"))
        for name in ("held", "line_outputs", "pairs", "result")
    ]
    assert result["lines_sha256"] == sha(PUBLIC / "line_outputs.json") and result[
        "margins_sha256"
    ] == sha(PUBLIC / "margins.npz")
    margins = np.load(PUBLIC / "margins.npz")
    for g, items in pairs.items():
        assert len(set(i for p in items for i in p)) == 2 * len(items)
        linewins = np.zeros(len(result["arms"]), int)
        written = linewins.copy()
        for k, (i, j) in enumerate(items):
            a, b = rows[i], rows[j]
            assert a["book"] == b["book"] and a["page"] != b["page"]
            correct = np.ones_like(linewins, dtype=bool)
            for side in range(2):
                for frame in range(2):
                    advantage = margins[g][k, side, frame]
                    # Actual first-row tie break under each physical order.
                    for rev in (False, True):
                        correct &= np.where(
                            advantage == 0, (side == int(rev)), advantage > 0
                        )
            linewins += correct
            if all(
                line[t]["arc"][1] == rows[t]["focus"][1]
                and line[t]["object"] in rows[t]["gloss"][1]
                for t in (i, j)
            ):
                written += correct
        assert (
            linewins.tolist() == result["groups"][g]["strict_line"]
            and written.tolist() == result["groups"][g]["strict_written"]
        )
    writer = restore()
    reader = load_focus(ROOT / "models/visual_clause_roles", "cpu")
    residual = load_adapter(ROOT / "models/pronoun_writer", "cpu")
    lookup = restore_lookup(ALIGNED, "cpu")
    face = Face(FACES[0])
    folder = PUBLIC / "samples"

    def pixels(s):
        return np.frombuffer(face(s), np.uint8).reshape(-1, 128).copy()

    letters = {face.cell(c): c for c in "abcdefghijklmnopqrstuvwxyz-"}
    records = []
    bitmap_hashes = {}
    groups = defaultdict(list)
    import hashlib

    for k, ix in enumerate(pairs["overall"]):
        actions = [rows[i]["gloss"][0][0] for i in ix]
        objects = [rows[i]["gloss"][1][0] for i in ix]
        terms = alternate(actions, objects)
        if not terms:
            continue
        lines = [pixels(rows[i]["page"]) for i in ix]
        if k == 77:
            for row, pix in enumerate(lines):
                write_strip(pix, folder / f"example-line-{row}.png")
            images = [
                Image.open(folder / f"example-line-{j}.png").convert("L")
                for j in (0, 1)
            ]
            page = Image.new("L", (max(im.width for im in images), 64), 255)
            for j, im in enumerate(images):
                page.paste(im, (0, 32 * j))
            page.save(folder / "example-page.png")
            lines = read_lines(folder / "example-page.png")
        strict = []
        for wording, names in (("literal", actions), ("alternate", terms)):
            hits = []
            written = []
            for rev in (False, True):
                source = lines[::-1] if rev else lines
                for side, i in enumerate(ix):
                    for f, frame in enumerate(HELD_FRAMES):
                        question = pixels(frame.format(names[side]))
                        ink, r = answer(
                            writer, reader, residual, lookup, source, question
                        )
                        word = "".join(letters.get(cell.tobytes(), "�") for cell in ink)
                        located = r["row"] - 1 == side ^ int(rev)
                        exact = (
                            located
                            and r["source_cell"] == rows[i]["focus"][1]
                            and r["stopped"]
                            and word in rows[i]["gloss"][1]
                        )
                        hits.append(located)
                        written.append(exact)
                        bitmap_hashes[word] = hashlib.sha256(ink.tobytes()).hexdigest()
                        records.append(
                            [
                                k,
                                wording,
                                int(rev),
                                side,
                                f,
                                r["row"],
                                r["source_cell"],
                                r["stopped"],
                                word,
                                located,
                                exact,
                            ]
                        )
                        if k == 77 and not rev and f == 0:
                            write_strip(
                                question,
                                folder / f"example-{wording}-question-{side}.png",
                            )
                            write_strip(
                                ink, folder / f"example-{wording}-answer-{side}.png"
                            )
            strict.append((all(hits), all(written)))
        assert strict[0][0] == bool((margins["overall"][k, :, :, 4] > 0).all())
        groups["all"].append(strict)
        groups[rows[ix[0]]["split"]].append(strict)
    dump(PUBLIC / "alternate_outputs.json", records)
    stats = {
        g: dict(
            pairs=len(x),
            line=np.array(x).sum(0)[:, 0].tolist(),
            written=np.array(x).sum(0)[:, 1].tolist(),
        )
        for g, x in groups.items()
    }
    dump(
        PUBLIC / "audit.json",
        dict(
            recounted=["overall", "same_head"],
            order=["literal", "alternate"],
            groups=stats,
            released_image_calls=len(records),
            outputs_sha256=sha(PUBLIC / "alternate_outputs.json"),
            ink_hashes=bitmap_hashes,
        ),
    )
    print(stats, len(records), flush=True)


if __name__ == "__main__":
    main()
