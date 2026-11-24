"""Cross the positions of historical labels and exemplars on the same bilingual cards."""

from hashlib import sha256
import numpy as np
import torch
from ilm.visual_lm.book_prefix import observations
from scripts.evaluate_recurrent_history import historical_pages
from scripts.replay_paired_evidence import STAGES
from scripts.prepare_visual_sequence_corpus import Rasterizer
from scripts.prepare_open_exam import MODEL, PUB, dump, read
from scripts.train_paired_evidence import gpu_owner
from scripts.write_open_exam import load
from scripts.write_recurrent_evidence import generate
from scripts.write_visual_discourse import Writer, encode_pages


def main(
    arm="matched",
    model_dir=MODEL,
    report=PUB,
    baseline_path=PUB / "matched-history.json",
):
    rows, pages, marks = historical_pages()
    previous = read(baseline_path) if baseline_path is not None else None
    raster = Rasterizer()

    def ink(s):
        return np.stack([np.frombuffer(raster.tile(c), np.uint8) for c in s])

    conditioned = []
    records = []
    for k in range(0, len(rows), 2):
        pair = rows[k : k + 2]
        assert (
            pair[0]["headword"] == pair[1]["headword"]
            and pair[0]["stage"] != pair[1]["stage"]
        )
        en = pair[0]["language"] == "en"
        word = pair[0]["headword"]
        for labels, images in ((0, 0), (1, 0), (0, 1), (1, 1)):
            a, b = [STAGES[pair[j ^ labels]["stage"]][int(en)] for j in range(2)]
            text = (
                f"In this dictionary, the {a} form of {word} is §. Its {b} form is §."
                if en
                else f"字书记载，{word}的{a}字形是§，{b}字形是§。"
            ).split("§")
            page = np.concatenate(
                [
                    ink(text[0]),
                    marks[k + images],
                    ink(text[1]),
                    marks[k + (1 ^ images)],
                    ink(text[2]),
                ]
            )
            if not (images or labels):
                assert np.array_equal(page, pages[2 * k + 1])
            for j in range(2):
                row = pair[j]
                conditioned.extend([pages[2 * (k + j)], page])
                records.append(
                    dict(
                        index=k + j,
                        lang=row["language"],
                        example=row["example"],
                        labels=labels,
                        images=images,
                        position=1 + (j ^ labels),
                        expected=pair[j ^ images ^ labels]["expected_sha256"],
                    )
                )
    with gpu_owner(), torch.inference_mode():
        writer = Writer(device="cuda")
        net = load(arm, "cuda", model_dir)
        shapes, seq = encode_pages(writer.palette, conditioned)
        x = torch.from_numpy(observations({"tiles": shapes}, "ink")).cuda()
        for k in range(0, len(records), 16):
            q, p = seq[2 * k : 2 * (k + 16) : 2], seq[2 * k + 1 : 2 * (k + 16) : 2]
            result = generate(
                writer.model,
                net,
                x,
                len(writer.palette),
                q,
                p,
                [a + b for a, b in zip(p, q, strict=True)],
                64,
            )
            for r, (ids, end) in zip(records[k : k + 16], result, strict=True):
                h = sha256(shapes[np.asarray(ids, int)].tobytes()).hexdigest()
                r.update(output=h, stop=end, correct=end and h == r["expected"])
                if (
                    previous is not None
                    and not (r["images"] or r["labels"])
                    and (h, end)
                    != (
                        previous[r["index"]]["sha256"],
                        previous[r["index"]]["stopped"],
                    )
                ):
                    raise ValueError("Original historical card did not replay")
    summary = {}
    for lang in ("zh", "en"):
        scored = [r for r in records if r["lang"] == lang and not r["example"]]
        groups = {
            i: [r for r in scored if r["index"] == i]
            for i in {r["index"] for r in scored}
        }
        summary[lang] = dict(
            questions=len(groups),
            all_four=sum(all(r["correct"] for r in group) for group in groups.values()),
            correct_by_position=[
                sum(r["correct"] for r in scored if r["position"] == pos)
                for pos in (1, 2)
            ],
            by_layout={
                f"{l}{s}": sum(
                    r["correct"] for r in scored if (r["labels"], r["images"]) == (l, s)
                )
                for l in (0, 1)
                for s in (0, 1)
            },
        )
    dump(report / "stage-order.json", dict(summary=summary, rows=records))


if __name__ == "__main__":
    from scripts.verse_gpu_runtime import configure_cuda_cli

    configure_cuda_cli()
    main()
