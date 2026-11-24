"""Grade the inclusion of authentic closed-cast supervision on old and fresh questions."""

import gzip
import json
import torch
from scripts.prepare_story_mentions import PUBLIC, RUN, read, dump
from scripts.train_story_cast import partitions, story_rows, prepare
from scripts.check_mention_transfer import datasets, score
from scripts.train_semantic_page import gpu_owner
from scripts.story_mention_options import select
from scripts.evaluate_story_mentions import grade, metrics, switches
from scripts.write_story_mentions import tiles
from scripts.prepare_visual_sequence_corpus import Rasterizer


from scripts.train_story_cast import load_cast


@torch.inference_mode()
def predict(rows):
    output = {}
    for mode in ("ink", "byte"):
        obs, (ids, slots, bounds, sizes, _), meta = prepare(rows, mode, "cuda")
        for arm in ["control", "ink", "geometry"] if mode == "ink" else ["byte"]:
            net = load_cast(arm, "cuda")
            pred = []
            for a in range(0, len(rows), 48):
                ix = slice(a, a + 48)
                feat = obs[ids[ix].long()]
                logits = (
                    net(
                        torch.zeros_like(feat) if arm == "geometry" else feat,
                        slots[ix],
                        bounds[ix],
                        sizes[ix],
                    )
                    .cpu()
                    .numpy()
                )
                for v, (slot, spans, q) in zip(logits, meta[ix]):
                    pred.append(select(v, slot, spans, q[0]))
            output[arm] = pred
        if mode == "ink":
            near = []
            for slot, spans, q in meta:
                options = [s for j, s in enumerate(spans[: q[0]]) if slot[j] >= 0]
                near.append(options[-1] if options else None)
            output["nearest"] = near
    return output


def main():
    parts, pin = partitions()
    if pin != read(PUBLIC / "cast-partitions.json"):
        raise ValueError("Changed reserved groups")
    official, _ = datasets()
    raster = Rasterizer()
    with gpu_owner():
        for lang in ("gap", "clue"):
            for split, rows in [
                ("official", official[lang]),
                ("fresh", parts[lang][1]),
            ]:
                choices = predict(rows)
                choices["always-null"] = [None] * len(rows)
                choices["always-first"] = [r["cast"][0] for r in rows]
                m = score(rows, choices, lang == "clue")
                write(lang + "-" + split, m, choices)
        for split in ("dev", "test"):
            original = read(RUN / (split + ".json"))
            for lang, g in story_rows(split).items():
                rows = [r for r in original if r["lang"] == lang]
                choices = predict(g)
                pairs = switches(rows, range(len(rows)))
                m = {}
                for arm, dec in choices.items():
                    grades = [
                        grade(r, tiles(r["text"], raster), span)
                        for r, span in zip(rows, dec)
                    ]
                    m[arm] = metrics(grades)
                    for k, subset in (
                        ("switched", pairs),
                        ("same_nearest", [p for p in pairs if p[2]]),
                    ):
                        m[arm][k] = [
                            sum(
                                grades[i]["ink"] and grades[j]["ink"]
                                for i, j, _ in subset
                            ),
                            len(subset),
                        ]
                write(lang + "-" + split, m, choices)


def write(key, metrics, decisions):
    dump(PUBLIC / ("cast-" + key + "-scores.json"), metrics)
    (PUBLIC / ("cast-" + key + "-decisions.json.gz")).write_bytes(
        gzip.compress(json.dumps(decisions, separators=(",", ":")).encode(), mtime=0)
    )
    print(key, {k: v["correct"] for k, v in metrics.items()}, flush=True)


if __name__ == "__main__":
    from scripts.verse_gpu_runtime import configure_cuda_cli

    configure_cuda_cli()
    main()
