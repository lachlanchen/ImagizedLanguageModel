"""Measure transfer of the released visual gloss writer to a third source face."""

import json
import numpy as np
import torch
from scripts.evaluate_visual_word_senses import archive, load_replies
from scripts.prepare_visual_word_senses import PUBLIC, WORK, BOOKS, sha, save
from scripts.train_typeface_writer import Face, FONTS
from scripts.train_visual_word_senses import corpus
from scripts.train_span_choice import gpu_owner
from scripts.write_visual_gloss import restore
from ilm.visual_lm.visual_word_sense import retinal_features


@torch.inference_mode()
def test():
    chars = json.loads((WORK / "cells.json").read_text())
    file = FONTS[1]
    face = Face((file, 2))
    # Score only when each occupied input is present in this font.
    covered = np.array([c == " " or ord(c) in face.coverage for c in chars])
    cells = np.array(
        [
            np.frombuffer(face(c) if ok else bytes(128), np.uint8)
            for c, ok in zip(chars, covered)
        ]
    )
    feat = retinal_features(torch.from_numpy(np.unpackbits(cells, axis=-1)).cuda())
    net = restore(device="cuda")
    old = np.load(WORK / "pixels.npy")[0]
    pairs = json.loads((PUBLIC / "pairs.json").read_text())
    by = {}
    for book in BOOKS:
        x, y = corpus(book)
        outputs = []
        ends = []
        available = covered[x].all(1)
        for i in range(0, len(x), 256):
            codes, stopped = net.write(
                feat[torch.from_numpy(x[i : i + 256]).cuda().long()]
            )
            raw = np.packbits(net.draw(codes).cpu().numpy(), axis=-1)
            for row, stream in zip(codes.cpu().numpy(), raw):
                stop = np.flatnonzero(row == net.stop)
                outputs.append(stream[: stop[0] if len(stop) else len(row)].tobytes())
            ends.extend(stopped.cpu().tolist())
        path = PUBLIC / ("serif_" + book + ".npz")
        archive(path, outputs, ends)
        original = load_replies(PUBLIC / ("aligned_" + book + ".npz"))
        same = [(v, e) == r for v, e, r in zip(outputs, ends, original)]
        exact = np.array(
            [
                e and raw == old[row[row != 0]].tobytes()
                for e, raw, row in zip(ends, outputs, y)
            ]
        )
        pair = np.array(pairs[book])
        allowed = available[pair].all(1)
        by[book] = dict(
            exact=int(exact[available].sum()),
            covered=int(available.sum()),
            total=len(x),
            paired_exact=int(exact[pair[allowed]].all(1).sum()),
            covered_pairs=int(allowed.sum()),
            stopped=sum(ends),
            identical=sum(same),
            output=sha(path),
        )
    save(PUBLIC / "serif.json", dict(font=sha(file), index=2, books=by))


if __name__ == "__main__":
    with gpu_owner():
        torch.set_float32_matmul_precision("high")
        test()
