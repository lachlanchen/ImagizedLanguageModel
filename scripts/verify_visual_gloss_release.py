"""Verify saved GPU physical replies against batched CPU inference from release weights."""

import json
import numpy as np
import torch
from scripts.prepare_visual_word_senses import WORK, PUBLIC, BOOKS, sha, save
from scripts.evaluate_visual_word_senses import load_replies
from scripts.train_visual_word_senses import corpus
from scripts.write_visual_gloss import restore, RELEASE
from ilm.visual_lm.visual_word_sense import retinal_features


@torch.inference_mode()
def run():
    net = restore()
    pixels = np.load(WORK / "pixels.npy")[0]
    feats = retinal_features(torch.from_numpy(np.unpackbits(pixels, axis=-1)))
    stats = {}
    for book in BOOKS:
        x, y = corpus(book)
        path = PUBLIC / ("aligned_" + book + ".npz")
        assert (
            sha(path)
            == json.loads((PUBLIC / "aligned_outputs.json").read_text())["books"][book][
                "outputs"
            ]
        )
        original = load_replies(path)
        diffs = []
        exact = 0
        for start in range(0, len(x), 96):
            pred, end = net.write(feats[torch.from_numpy(x[start : start + 96]).long()])
            raw = np.packbits(net.draw(pred).numpy(), axis=-1)
            for j, (cells, state, e) in enumerate(zip(raw, pred.numpy(), end.tolist())):
                i = start + j
                out = cells[state != net.stop].tobytes()
                ok = e and out == pixels[y[i, y[i] != 0]].tobytes()
                exact += ok
                if (out, e) != original[i]:
                    diffs.append(dict(i=i, reply=out.hex(), stopped=e, correct=ok))
        stats[book] = dict(
            cases=len(x),
            identical=len(x) - len(diffs),
            cpu_exact=exact,
            differences=diffs,
        )
    save(
        PUBLIC / "cpu_replay.json",
        dict(weights=sha(RELEASE / "writer.safetensors"), books=stats),
    )


if __name__ == "__main__":
    torch.set_num_threads(2)
    run()
