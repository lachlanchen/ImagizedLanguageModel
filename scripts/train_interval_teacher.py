"""Train three matched independent image writers under endpoint supervision."""

import argparse
import json
import numpy as np
import torch

from ilm.visual_lm.interval_teacher import path_loss
from scripts.interval_teacher_corpus import WORK, PARENT, SEED, load_rows, NATURAL
from scripts.train_causal_glyph_writer import sha
from scripts.train_mixed_visual_answers import fit_rows
from scripts.train_span_choice import gpu_owner


class Supervision:
    def __init__(self, rows, distill):
        self.natural = {i for i, r in enumerate(rows) if r.task in NATURAL}
        self.logits = {}
        self.distill = distill
        if distill:
            if (
                sha(WORK / "teacher.npz")
                != json.loads((WORK / "teacher.json").read_text())["sha256"]
            ):
                raise ValueError("Teacher archive changed")
            with np.load(WORK / "teacher.npz") as data:
                x = data["logits"]
                off = data["offsets"]
                self.logits = {
                    i: torch.from_numpy(x[a:b])
                    for i, a, b in zip(data["indices"], off[:-1], off[1:], strict=True)
                }
            if self.logits.keys() != self.natural:
                raise ValueError("Missing natural training rows")

    def __call__(self, ids):
        keep = [i for i, j in enumerate(ids) if j in self.natural]
        if not keep:
            return None

        def extra(pointer, gate, stop, source, valid, targets, mask):
            t = int(mask[keep].sum(1).max())
            p = int(valid[keep].sum(1).max())
            teacher = None
            if self.distill:
                ts = [self.logits[ids[i]] for i in keep]
                teacher = torch.nn.utils.rnn.pad_sequence(ts, batch_first=True).to(
                    source.device
                )
                assert teacher.size(1) == p
            return 0.5 * path_loss(
                pointer[keep, :t, :p],
                gate[keep, :t],
                stop[keep, :t],
                source[keep, :p],
                valid[keep, :p],
                targets[keep, :t],
                mask[keep, :t],
                teacher,
                0.5 if self.distill else 0.0,
            )

        return extra


def run(arms):
    torch.set_num_threads(2)
    rows = load_rows()
    parent = json.loads((PARENT / "page_fit.json").read_text())["weights"]
    for arm in arms:
        out = WORK / arm
        out.mkdir(exist_ok=False)
        print("training", arm, len(rows), flush=True)
        supervision = Supervision(rows, arm == "distilled") if arm != "replay" else None
        fit_rows(out, PARENT / "page", parent, rows, SEED, supervision)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("arms", choices=["replay", "aligned", "distilled"], nargs="+")
    with gpu_owner():
        run(ap.parse_args().arms)
