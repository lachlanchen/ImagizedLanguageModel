"""Measure reference coverage as more spans are retained, without retraining."""

from collections import Counter
import json

import torch

from ilm.visual_lm.bound_fact_reader import best_bounds, features_batch
from ilm.visual_lm.ordered_page_features import contextualize
from ilm.visual_lm.role_conditioned_span import span_scores
from scripts.prepare_span_choice import PUBLIC
from scripts.prepare_visual_facts import Face, FONTS, write_json
from scripts.train_span_choice import gpu_owner, model, records


@torch.inference_mode()
def evaluate():
    dest = PUBLIC / "coverage.json"
    if dest.exists():
        raise FileExistsError(dest)
    reader, face = model(), Face((FONTS[0], 2))
    rows = records("fresh")
    counts, by_case = Counter(), []
    for start in range(0, len(rows), 24):
        batch = rows[start : start + 24]
        pairs = [(face(v["p"]), face(v["q"])) for v in batch]
        data = features_batch(contextualize(reader.base.reader, pairs), "cuda")
        roles, bounds = reader.base.head(*data)
        sb = best_bounds(bounds[:, :, :2]).tolist()
        role = torch.tensor(
            [
                reader.channel.choose(q, p[128 * a : 128 * b], r)[0]
                for (p, q), (a, b), r in zip(
                    pairs, sb, roles.argmax(1).tolist(), strict=True
                )
            ],
            device="cuda",
        )
        prior = bounds[:, :, 2:]
        ends = [data[1][:, :, 768:770], prior] + [
            h(data[1], data[2], role, prior) for h in reader.heads
        ]
        rankings = []
        for logits in ends:
            scores = span_scores(logits, data[2])
            flat = scores.flatten(1)
            top = flat.argsort(dim=1, descending=True, stable=True)[:, :8]
            assert torch.equal(top[:, 0], flat.argmax(1))
            rankings.append((top // scores.shape[2], top % scores.shape[2] + 1))
        starts = torch.stack([v[0] for v in rankings], 1).tolist()
        lengths = torch.stack([v[1] for v in rankings], 1).tolist()
        for j, ((p, _), v) in enumerate(zip(pairs, batch, strict=True)):
            refs = {face(a) for a in v["refs"]}
            first = [
                next(
                    (
                        i
                        for i, (a, n) in enumerate(zip(aa, nn, strict=True), 1)
                        if (a + n) * 128 <= len(p)
                        and p[a * 128 : (a + n) * 128] in refs
                    ),
                    9,
                )
                for aa, nn in zip(starts[j], lengths[j], strict=True)
            ]
            level = min(first)
            counts.update({k: level <= k for k in (1, 2, 4, 8)})
            by_case.append(
                dict(
                    source=v["group"],
                    key=v["key"],
                    first_rank_if_at_most_eight=level if level <= 8 else None,
                )
            )
    expected = json.loads((PUBLIC / "results.json").read_text())["direct"]["fresh"][
        "oracle"
    ]
    assert counts[1] == expected
    write_json(
        dest,
        dict(
            n=len(rows),
            top_k=counts,
            recovered=counts[8] - counts[1],
            missing=len(rows) - counts[1],
            criterion=2 * (counts[8] - counts[1]) >= len(rows) - counts[1],
            cases=by_case,
        ),
    )
    print("coverage", dict(counts), flush=True)


if __name__ == "__main__":
    with gpu_owner():
        evaluate()
