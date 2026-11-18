"""Measure two pictured completions under identical answer history and changed source."""

import numpy as np
import torch

from ilm.visual_lm.idiom_ink import visual_input
from ilm.visual_lm.causal_glyph_translator import COPY, EXEMPLAR
from scripts.prepare_predicate_likelihood import (
    OUT,
    PREDICATES,
    Rasterizer,
    load,
    sha,
    write_new,
)
from scripts.train_paired_student import restore
from scripts.train_visual_event_roles import PARENT, raster


@torch.inference_mode()
def measure():
    torch.set_num_threads(2)
    render = Rasterizer()
    rows = load(OUT / "test.json")
    fit = load(OUT / "fit.json")
    assert sha(OUT / "test.json") == load(OUT / "admission.json")["test_sha"]
    results = {}
    for arm, weights in (
        ("parent", PARENT),
        ("replay", OUT / "replay.safetensors"),
        ("context", OUT / "context.safetensors"),
    ):
        assert sha(weights) == (
            fit["parent"] if arm == "parent" else fit["arms"][arm]["sha256"]
        )
        model = restore(weights).eval()
        image_keys = model.image_keys()
        samples = []

        def bits(s):
            return torch.from_numpy(np.unpackbits(raster(s, render), axis=1)).bool()

        for i, r in enumerate(rows):
            g = r["gloss"]
            (other,) = set(PREDICATES[r["predicate"]]) - {g}
            # Compare exactly the same physical emission choices after the same
            # decoder history across all three source interventions.
            choices = bits(g[0] + other[0])
            seen = (model.exemplars[:, None] == choices).all(-1)
            assert (seen.sum(0) == 1).all()
            pref = r["target"][: r["target"].index(g)]
            decoder = model.start.reshape(1, 1, -1)
            if pref:
                decoder = torch.cat(
                    (
                        decoder,
                        model.embed(visual_input(bits(pref), model.landmarks))[None],
                    ),
                    1,
                )
            scores = {}
            for label, src in (
                ("actual", r["source"] + "。"),
                ("opposite", rows[i ^ 1]["source"] + "。"),
                ("isolated", r["predicate"]),
            ):
                x = bits(src)
                n = torch.tensor([len(x)])
                mem, keys, mask, h = model.read(
                    visual_input(x, model.landmarks)[None], n
                )
                q, _ = model.history(decoder, h)
                st, att, _ = model.plan_steps(
                    q, mem, keys, mask, n, torch.tensor([-0.65])
                )
                state = st[:, -1]
                pointer = att[0, -1].exp()
                route = model.route(state)[0].softmax(-1)
                proto = model.prototype_scores(state, image_keys)[0].softmax(-1)
                prob = route[COPY] * (
                    pointer[:, None] * (x[:, None] == choices).all(-1)
                ).sum(0) + route[EXEMPLAR] * (proto[:, None] * seen).sum(0)
                scores[label] = [float(z) for z in prob]
            samples.append(scores)
        results[arm] = samples
    doc = {}
    for a, vals in results.items():
        m = {}
        for label in ("actual", "opposite", "isolated"):
            p = np.array([s[label] for s in vals])
            wins = p[:, 0] > p[:, 1]
            m[label] = dict(
                wins=int(wins.sum()),
                both=int((wins[::2] & wins[1::2]).sum()),
                median_combined_mass=float(np.median(p.sum(1))),
                median_correct_mass=float(np.median(p[:, 0])),
            )
        odds = lambda lab: np.array([np.log(s[lab][0] / s[lab][1]) for s in vals])
        gain = odds("actual") - odds("opposite")
        m["source_gain"] = dict(
            positive=int((gain > 0).sum()),
            both_positive=int(((gain[::2] > 0) & (gain[1::2] > 0)).sum()),
            mean_log_odds=float(gain.mean()),
        )
        doc[a] = m
    write_new(
        OUT / "forced.json",
        dict(
            summary=doc,
            probabilities=results,
            rows=len(rows),
            roster=sha(OUT / "test.json"),
        ),
    )


if __name__ == "__main__":
    measure()
