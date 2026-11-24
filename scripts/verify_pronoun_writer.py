"""Reproduce the two-image question task with both released components on CPU."""

import numpy as np
import torch

from ilm.visual_lm.portable_writer import read_image
from scripts.answer_visual_gloss import answer, load_focus
from scripts.evaluate_everyday_gloss import DEST
from scripts.prepare_asked_gloss import ROOT, dump, read
from scripts.prepare_pronoun_writer import PUBLIC, window
from scripts.train_typeface_writer import Face, FACES
from scripts.write_visual_gloss import restore, load_adapter


@torch.inference_mode()
def main():
    torch.set_num_threads(2)
    face = Face(FACES[0])
    writer = restore()
    focus = load_focus(ROOT / "models/everyday_gloss", "cpu")
    adapter = load_adapter(ROOT / "models/pronoun_writer", "cpu")
    reference = read(PUBLIC / "transfer_words.json")["aligned"]
    correct = 0
    for r in read(DEST / "replay.json"):
        tag, j = r["image"].split("-q")
        page = read_image(DEST / "samples" / (tag + ".png"), len(r["page"]))
        question = read_image(
            DEST / "samples" / ("question" + j + ".png"), len(r["query"])
        )
        actual, ended, pos = answer(
            writer, focus, page, question, context_adapter=adapter
        )
        expected = reference[window(r["page"], r["position"])]
        assert pos == r["position"]
        if expected == "<unfinished>":
            assert not ended
        else:
            assert ended and np.array_equal(
                actual, np.frombuffer(face(expected), np.uint8).reshape(-1, 128)
            )
        correct += 1
    dump(
        PUBLIC / "release_replay.json",
        dict(
            identical_focus_and_replies=correct,
            device="cpu",
            parameters=sum(
                p.numel() for net in (writer, focus, adapter) for p in net.parameters()
            ),
        ),
    )


if __name__ == "__main__":
    main()
