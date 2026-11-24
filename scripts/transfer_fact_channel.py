"""Use the deployed pictured-page pipeline to continue books and return historical specimens."""

from collections import defaultdict
from hashlib import sha256
from scripts.evaluate_visual_discourse import draw
import numpy as np
import torch
from scripts.prepare_fact_channel import MODEL, PUB, Ink, dump, read
from scripts.train_region_writer import gpu_owner
from scripts.write_fact_channel import VisualAnswer, bitmap_ids
from scripts.evaluate_recurrent_history import historical_pages
from scripts.train_visual_discourse import Corpus
from scripts.evaluate_visual_discourse import cohorts
from scripts.verse_gpu_runtime import configure_cuda_cli


def main(models=None, public=PUB, output_model=MODEL):
    ink = Ink(np.load(MODEL / "palette.npy"))
    specs = []
    labels, images, targets = historical_pages()
    for i, r in enumerate(labels):
        q, p = (bitmap_ids(ink, img) for img in images[2 * i : 2 * i + 2])
        specs.append(
            (
                r["language"] + "-period",
                q,
                [p[j : j + 256] for j in range(0, len(p), 256)],
                bitmap_ids(ink, targets[i]),
            )
        )
    corpus = Corpus()
    for t, rows in cohorts(corpus)[1].items():
        if not t.startswith("book-"):
            continue
        q = ink.encode(
            "请续写下一行。" if t.endswith("zh") else "Continue with the next line."
        )
        for r in rows:
            page = bitmap_ids(ink, corpus.tiles[r["x"]])
            specs.append(
                (
                    t,
                    q,
                    [page[j : j + 256] for j in range(0, len(page), 256)],
                    bitmap_ids(ink, corpus.tiles[r["y"][:8]]),
                )
            )
    tiles = ink.array()
    scores = {}
    stage_evidence = {}
    yan = []
    with gpu_owner(), torch.inference_mode():
        for arm in models or ("direct", "human", "predicted"):
            net = models[arm]() if models else VisualAnswer(arm, "cuda")
            result = defaultdict(list)
            stages = defaultdict(lambda: defaultdict(int))
            for j in range(0, len(specs), 12):
                batch = specs[j : j + 12]
                values, spans = net.answer(
                    tiles, [q for t, q, p, y in batch], [p for t, q, p, y in batch]
                )
                for k, ((t, q, p, y), (ids, end), span) in enumerate(
                    zip(batch, values, spans, strict=True)
                ):
                    if t.endswith("-period"):
                        r = labels[j + k]
                        st = stages[t]
                        st["n"] += 1
                        st["answer_contains_specimen"] += y[0] in ids
                        if span is not None:
                            fact = span[2].tolist()
                            st["fact_contains_specimen"] += y[0] in fact
                            st["fact_only_specimen"] += fact == y
                        if r["headword"] == "言":
                            folder = (
                                output_model
                                / "examples"
                                / ("yan-" + r["language"] + "-" + r["stage"])
                            )
                            folder.mkdir(exist_ok=True, parents=True)
                            draw(folder, "question", tiles[q])
                            draw(folder, "reference", tiles[y])
                            for m, b in enumerate(p):
                                draw(folder, "source-" + str(m), tiles[b])
                            dump(
                                folder / "input.json",
                                dict(
                                    question=dict(png="question.png", cells=len(q)),
                                    regions=[
                                        dict(
                                            png="source-" + str(m) + ".png",
                                            cells=len(b),
                                        )
                                        for m, b in enumerate(p)
                                    ],
                                ),
                            )
                            raw = tiles[np.asarray(ids, int)]
                            draw(folder, arm, raw)
                            if span is not None:
                                draw(folder, "read-fact", tiles[span[2]])
                            yan.append(
                                dict(
                                    arm=arm,
                                    language=r["language"],
                                    stage=r["stage"],
                                    label=r["label"],
                                    expected=r["expected_sha256"],
                                    answer_hash=sha256(raw.tobytes()).hexdigest(),
                                    stopped=end,
                                    cells=len(ids),
                                    correct=end and ids == y,
                                )
                            )

                    result[t].append(
                        ids[:8] == y if t.startswith("book-") else end and ids == y
                    )
            stage_evidence[arm] = stages
            scores[arm] = {k: dict(n=len(v), correct=sum(v)) for k, v in result.items()}
    dump(public / "transfer.json", scores)
    dump(public / "historical-reading.json", dict(stages=stage_evidence, yan=yan))
    print(scores)


if __name__ == "__main__":
    configure_cuda_cli()
    main()
