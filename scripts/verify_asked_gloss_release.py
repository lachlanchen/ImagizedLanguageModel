"""Run every held contrast through the release, with both images and an autonomous stop."""

import numpy as np
import torch
from scripts.answer_visual_gloss import answer, load_focus, RELEASE
from scripts.asked_gloss_questions import HELD
from scripts.prepare_asked_gloss import PUBLIC, read, dump, sha
from scripts.train_typeface_writer import Face, FACES
from scripts.write_visual_gloss import restore
from ilm.visual_lm.attested_gloss import AttestedGlosses


@torch.inference_mode()
def verify():
    rows = read(PUBLIC / "predictions.json")
    positions = np.load(PUBLIC / "order_positions.npz")["ordered_contrast"]
    frozen = read(PUBLIC / "order_result.json")
    assert sha(PUBLIC / "predictions.json") == frozen["previous_predictions"]
    assert sha(PUBLIC / "emitted.npz") == frozen["previous_ink"]
    assert sha(PUBLIC / "order_positions.npz") == frozen["positions"]
    face = Face(FACES[0])

    def pic(s):
        return np.frombuffer(face(s), np.uint8).copy().reshape(-1, 128)

    qs = [pic(q) for p in HELD for q in p]
    writer = restore()
    reader = load_focus(RELEASE, "cpu")
    lex = AttestedGlosses(RELEASE / "attested.npz", writer.stop, "cpu")
    bank = read(PUBLIC / "attested_result.json")
    assert sha(RELEASE / "attested.npz") == bank["bank"]
    changes = {(c["row"], c["cell"]): bytes.fromhex(c["ink"]) for c in bank["changes"]}
    diffs = []
    tested = [0, 0]
    same = [0, 0]
    first_books = set()
    with np.load(PUBLIC / "emitted.npz") as archive:
        offsets, ink, stops = (archive[n] for n in ("offsets", "ink", "stopped"))
        for j, r in enumerate(rows):
            page = pic(r["page"])
            bank_check = r["book"] == "Daodejing" or r["book"] not in first_books
            first_books.add(r["book"])
            for n, q in enumerate(qs):
                ref = positions[j, n // 2, n % 2]
                ix = r["indices"][ref]
                free = ink[offsets[ix] : offsets[ix + 1]].tobytes()
                for known in range(1 + int(bank_check)):
                    actual, end, i = answer(
                        writer, reader, page, q, lex if known else None
                    )
                    expected = changes.get((j, ref), free) if known else free
                    expected_end = True if known else bool(stops[ix])
                    identical = (
                        i == ref
                        and end == expected_end
                        and actual.tobytes() == expected
                    )
                    if not identical:
                        diffs.append(
                            dict(
                                row=j,
                                request=n,
                                known=known,
                                position=i,
                                ended=end,
                                ink=actual.tobytes().hex(),
                            )
                        )
                    tested[known] += 1
                    same[known] += int(identical)
    dump(
        PUBLIC / "release_replay.json",
        dict(
            tested=tested,
            identical=same,
            differences=diffs,
            modes=[
                "free, all held pages",
                "attested, all Laozi pages and first page of every other work",
            ],
            focus=sha(RELEASE / "focus.safetensors"),
            bank=sha(RELEASE / "attested.npz"),
        ),
    )
    print(tested, same)


if __name__ == "__main__":
    torch.set_num_threads(2)
    verify()
