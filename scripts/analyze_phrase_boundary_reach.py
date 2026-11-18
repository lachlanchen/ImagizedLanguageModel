"""Count complete archived references reachable by one ideal visual phrase replacement."""

from collections import Counter
import time

from scripts.prepare_phrase_boundary_trial import (
    EVIDENCE,
    EVENT,
    OUT,
    PREVIOUS,
    Rasterizer,
    load,
    sha,
    write_new,
)
from scripts.train_visual_event_roles import PARENT, raster
from scripts.train_paired_student import restore
from scripts.train_pictured_revision import composer
from scripts.evaluate_paired_student import read_streams
from ilm.visual_lm.contextual_phrase_realizer import ContextualPhraseRealizer
from ilm.visual_lm.visual_phrase_boundary import VisualPhraseBoundary
import torch


def run():
    torch.set_num_threads(2)
    start = time.monotonic()
    n = len(load(EVIDENCE / "test.json"))
    broad = load(EVENT / "broad.json")
    data = load(EVIDENCE / "evaluation.json")
    for k in ("source", "reference", "composer", "joint", "single"):
        assert sha(OUT / (k + ".npz")) == data["streams"][k + ".npz"]
    x, t, original, joint, single = [
        read_streams(k, OUT)[n:]
        for k in ("source", "reference", "composer", "joint", "single")
    ]
    assert len(x) == len(broad) == 1024
    draft = composer()
    render = Rasterizer()
    senses = ContextualPhraseRealizer.restore(
        OUT / "senses.msgpack.gz", restore(PARENT), draft
    )
    model = VisualPhraseBoundary.restore(OUT / "boundaries.msgpack.gz", senses)
    rows = []
    for s, ref, base, y, z in zip(x, t, original, joint, single, strict=True):
        target = ref.tobytes()
        # Output length is unconstrained. A perfect fill exists iff the two
        # generated exteriors match the reference ends without overlapping.
        prefixes = {0: b""}
        suffixes = {len(s): b""}
        for i in range(1, len(s)):
            u = draft.generate(s[:i]).tobytes()
            v = draft.generate(s[i:]).tobytes()
            if target.startswith(u):
                prefixes[i] = u
            if target.endswith(v):
                suffixes[i] = v
        matches = [
            (b - a, a, b)
            for a, u in prefixes.items()
            for b, v in suffixes.items()
            if b > a and len(u) + len(v) <= len(target)
        ]
        width, l, r = min(matches)
        assert (
            target
            == prefixes[l]
            + target[len(prefixes[l]) : len(target) - len(suffixes[r])]
            + suffixes[r]
        )
        rows.append(
            dict(
                source_cells=len(s),
                target_cells=len(ref),
                minimum_width=width,
                witness=[l, r],
                baseline_exact=base.tobytes() == target,
                joint_exact=y.tobytes() == target,
                heads=sum(bytes(v) in senses.heads for v in s),
                joint_differs=y.tobytes() != z.tobytes(),
            )
        )
    known = []
    for r in load(PREVIOUS / "test.json"):
        if "相去" not in r["source"]:
            continue
        seq = raster(r["source"] + "。", render)
        outputs = [a.generate(seq).tobytes() for a in (senses, model)]
        chars = {
            render.tile(c): c
            for c in r["source"]
            + r["target"]
            + "相隔相距距离与前事情正一年里各几十数五百两。"
        }
        texts = [
            "".join(chars.get(out[i : i + 128], "▧") for i in range(0, len(out), 128))
            for out in outputs
        ]
        known.append(
            dict(
                source=r["source"],
                single=texts[0],
                joint=texts[1],
                single_bad=raster("相距离", render).tobytes() in outputs[0],
                joint_bad=raster("相距离", render).tobytes() in outputs[1],
            )
        )
    assert len(known) == 3
    write_new(
        EVIDENCE / "reach.json",
        dict(
            seconds=time.monotonic() - start,
            counts={
                k: sum(r["baseline_exact"] or r["minimum_width"] <= k for r in rows)
                for k in (1, 3, 6)
            },
            exact=sum(r["baseline_exact"] for r in rows),
            centres=sum(bool(r["heads"]) for r in rows),
            joint_changes=sum(r["joint_differs"] for r in rows),
            known=known,
            rows=rows,
        ),
    )
    print("Known cases", known)


if __name__ == "__main__":
    run()
