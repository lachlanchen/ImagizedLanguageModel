"""Examine the fixed-pointer constraint on the two actions in Li Bai's first line."""

from pathlib import Path
import argparse
import numpy as np
import torch

from scripts.answer_pictured_action import answer, restore_lookup
from scripts.answer_visual_gloss import load_focus
from scripts.prepare_visual_action_meaning import ROOT, dump, table
from scripts.pictured_action_questions import HELD_FRAMES
from scripts.train_typeface_writer import Face, FACES
from scripts.write_visual_gloss import restore, load_adapter, respond


@torch.inference_mode()
def probe(output):
    torch.set_num_threads(2)
    writer = restore()
    source = load_focus(ROOT / "models/visual_clause_roles", "cpu")
    residual = load_adapter(ROOT / "models/pronoun_writer", "cpu")
    lookup = restore_lookup(ROOT / "models/visual_action_meaning/aligned", "cpu")
    face = Face(FACES[0])

    def picture(s):
        return np.frombuffer(face(s), np.uint8).reshape(-1, 128).copy()

    alphabet = {face.cell(c): c for c in "abcdefghijklmnopqrstuvwxyz-"}

    def word(ink):
        return "".join(alphabet.get(x.tobytes(), "�") for x in ink)

    passage = "舉杯邀明月"
    distractor = "松下問童子"
    assert {passage, distractor} <= {r["line"] for r in table("tang.tsv")}
    definitions = {r["action"]: r["description"] for r in table("webster.tsv")}
    records = []
    supplied = {}
    for action, character in [("raise", "杯"), ("invite", "月")]:
        target = passage.index(character)
        ink, stop = respond(writer, picture(passage), target, context_adapter=residual)
        supplied[action] = dict(cell=target, word=word(ink), stopped=stop)
        for reverse in (False, True):
            order = [passage, distractor]
            if reverse:
                order.reverse()
            for expr in (action, definitions[action]):
                for frame in HELD_FRAMES:
                    q = frame.format(expr)
                    ink, rep = answer(
                        writer,
                        source,
                        residual,
                        lookup,
                        [picture(s) for s in order],
                        picture(q),
                    )
                    records.append(
                        dict(
                            action=action,
                            reverse=reverse,
                            question=q,
                            selected=order[rep["row"] - 1],
                            cell=rep["source_cell"],
                            word=word(ink),
                            stopped=rep["stopped"],
                        )
                    )
    pointers = {r["cell"] for r in records if r["selected"] == passage}
    assert len(pointers) <= 1
    dump(
        output,
        dict(
            source=passage,
            distractor=distractor,
            source_previously_seen=True,
            supplied=supplied,
            pointers_on_target=sorted(pointers),
            requests=records,
        ),
    )
    print("Selected cells in target line", pointers, "supplied objects", supplied)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    probe(parser.parse_args().output)
