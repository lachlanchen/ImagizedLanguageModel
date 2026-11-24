"""Test ordered reference on authentic prose and retain the original cast tasks."""

import gzip
import json
import numpy as np
import torch
from ilm.visual_lm.reference_order import cue
from scripts.ordered_reference_data import natural
from scripts.train_reference_order import load_order, sides
from scripts.train_story_cast import load_cast, prepare, partitions
from scripts.prepare_story_mentions import PUBLIC, dump
from scripts.check_mention_transfer import datasets, score
from scripts.story_mention_options import select
from scripts.train_semantic_page import gpu_owner


@torch.inference_mode()
def predict(rows):
    out = {}
    for mode in ("ink", "byte"):
        obs, (ids, slots, bounds, sizes, _), meta = prepare(rows, mode, "cuda")
        parent = load_cast(mode, "cuda")
        nets = {
            arm: load_order(arm, "cuda")
            for arm in (("ink", "erased") if mode == "ink" else ("byte",))
        }
        for a in range(0, len(rows), 48):
            ix = slice(a, a + 48)
            x = obs[ids[ix].long()]
            s = slots[ix]
            q = bounds[ix]
            n = sizes[ix]
            original = parent(x, s, q, n).cpu().numpy()
            delta = {
                arm: net(cue(x, q, s, n, arm == "erased")).cpu().numpy()
                for arm, net in nets.items()
            }
            for key, shift in [(mode + "-base", None), *delta.items()]:
                preds = out.setdefault(key, [])
                for j, (v, (tags, spans, marks)) in enumerate(zip(original, meta[ix])):
                    line = v.copy()
                    order = sides(tags, marks[0])
                    if order is not None and shift is not None:
                        for col, k in enumerate(order):
                            tag = np.array(tags)
                            line[: len(tags)] += np.where(tag == k, shift[j, col], 0.0)
                    preds.append(select(line, tags, spans, marks[0]))
    return out


def main():
    groups, screen = natural()
    official, _ = datasets()
    parts, _ = partitions()
    outcomes = {}
    with gpu_owner():
        for key, rows in [
            *groups.items(),
            *[(s + "-retained", r) for s, r in official.items()],
            *[(s + "-fresh", r[1]) for s, r in parts.items()],
        ]:
            choices = predict(rows)
            m = score(rows, choices, key.startswith("clue"))
            outcomes[key] = m
            print(key, {a: s["correct"] for a, s in m.items()}, flush=True)
            (PUBLIC / ("order-" + key + "-decisions.json.gz")).write_bytes(
                gzip.compress(
                    json.dumps(choices, separators=(",", ":")).encode(), mtime=0
                )
            )
    dump(PUBLIC / "order-scores.json", dict(screen=screen, groups=outcomes))


if __name__ == "__main__":
    from scripts.verse_gpu_runtime import configure_cuda_cli

    configure_cuda_cli()
    main()
