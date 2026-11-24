"""Fit a training-only ink exemplar bank and test the unchanged gloss writer."""

import json
from pathlib import Path
import numpy as np
import torch
from scripts.prepare_visual_word_senses import WORK, PUBLIC, BOOKS, sha, save
from scripts.train_visual_word_senses import corpus
from scripts.write_visual_gloss import RELEASE, restore
from scripts.evaluate_visual_word_senses import archive, load_replies
from scripts.train_typeface_writer import Face, FONTS
from scripts.train_span_choice import gpu_owner
from ilm.visual_lm.stroke_matching import StrokeMatcher
from ilm.visual_lm.visual_word_sense import retinal_features


@torch.inference_mode()
def evaluate():
    train, _ = corpus("train")
    ink = np.load(WORK / "pixels.npy")
    ids = np.unique(train)
    bank = np.unique(ink[:, ids].reshape(-1, 128), axis=0)
    path = RELEASE / "source_ink.npy"
    np.save(path, bank)
    matcher = StrokeMatcher(bank, "cuda")
    chars = json.loads((WORK / "cells.json").read_text())
    pairs = json.loads((PUBLIC / "pairs.json").read_text())
    model = restore(device="cuda")
    results = {}
    for name, fil, idx in [
        ("sans", FONTS[0], 2),
        ("serif", FONTS[1], 2),
        ("kai", Path("/usr/share/fonts/truetype/arphic/ukai.ttc"), 0),
    ]:
        face = Face((fil, idx))
        supported = np.array([c.isspace() or ord(c) in face.coverage for c in chars])
        raw = np.array(
            [
                np.frombuffer(face(c) if ok else bytes(128), np.uint8)
                for c, ok in zip(chars, supported)
            ]
        )
        mapped = matcher.match(raw)
        correct = ((ink == mapped[None, :, :]).all(-1)).any(0)
        known = np.isin(np.arange(len(chars)), ids)
        for mode, sources in [("matched", mapped), ("raw", raw)]:
            feat = retinal_features(
                torch.from_numpy(np.unpackbits(sources, axis=-1)).cuda()
            )
            if mode == "raw" and name in ("sans", "serif"):
                continue  # already bound; the existing replies are read below
            for book in BOOKS:
                x, y = corpus(book)
                outs = []
                ends = []
                for i in range(0, len(x), 256):
                    codes, end = model.write(
                        feat[torch.from_numpy(x[i : i + 256]).cuda().long()]
                    )
                    cells = np.packbits(model.draw(codes).cpu().numpy(), axis=-1)
                    for a, b in zip(codes.cpu().numpy(), cells):
                        at = np.flatnonzero(a == model.stop)
                        outs.append(b[: at[0] if len(at) else len(a)].tobytes())
                    ends.extend(end.cpu().tolist())
                out = PUBLIC / (name + "_" + mode + "_" + book + ".npz")
                archive(out, outs, ends)
                eligible = supported[x].all(1)
                pair = np.array(pairs[book])
                pc = eligible[pair].all(1)
                hit = np.array(
                    [
                        end and a == ink[0, yy[yy != 0]].tobytes()
                        for end, a, yy in zip(ends, outs, y)
                    ]
                )
                typ = np.unique(x[eligible])
                typ = typ[typ != 0]
                original = load_replies(PUBLIC / ("aligned_" + book + ".npz"))
                results[name + "/" + mode + "/" + book] = dict(
                    exact=int(hit[eligible].sum()),
                    covered=int(eligible.sum()),
                    pairs=int(pc.sum()),
                    complete=int(hit[pair[pc]].all(1).sum()),
                    optical_known=int(known[typ].sum()),
                    optical_right=int((known[typ] & correct[typ]).sum()),
                    unfamiliar=int((~known[typ]).sum()),
                    unfamiliar_right=int((~known[typ] & correct[typ]).sum()),
                    output=sha(out),
                    identical_to_sans=sum(
                        (a, e) == b for a, e, b in zip(outs, ends, original)
                    ),
                    font=sha(fil),
                )
    save(
        PUBLIC / "matching.json",
        dict(
            bank=sha(path),
            bank_cells=len(bank),
            stored_bytes=path.stat().st_size,
            results=results,
        ),
    )


if __name__ == "__main__":
    with gpu_owner():
        torch.set_float32_matmul_precision("high")
        evaluate()
