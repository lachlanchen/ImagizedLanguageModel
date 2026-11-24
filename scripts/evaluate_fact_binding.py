"""Evaluate supplied pair membership and autonomous image answers on disjoint public pages."""

from collections import defaultdict
from hashlib import sha256
import torch
from ilm.visual_lm.book_prefix import observations
from ilm.visual_lm.fact_binding import decode, pack_paths, score_paths
from scripts.train_fact_binding import corpus
from scripts.write_fact_binding import load, read_batch
from scripts.prepare_fact_binding import PUB, MODEL, dump
from scripts.prepare_multiple_ink_runs import multi
from scripts.evaluate_complete_fact_sets import setup
from scripts.evaluate_multiple_ink_runs import grade
from scripts.evaluate_visual_discourse import packed, draw
from scripts.train_paired_evidence import gpu_owner
from scripts.write_multiple_ink_runs import write


def exchanged(rows):
    order = sorted(
        range(len(rows)),
        key=lambda i: (rows[i]["lang"], len(rows[i]["qi"]), rows[i]["id"]),
    )
    groups = defaultdict(list)
    for i in order:
        groups[rows[i]["lang"]].append(i)
    replacements = {}
    for indices in groups.values():
        for i, j in zip(indices, indices[1:] + indices[:1]):
            if rows[i]["qi"] == rows[j]["qi"]:
                raise ValueError("Question exchange left the image unchanged")
            replacements[i] = j
    return replacements


@torch.inference_mode()
def main():
    datasets = {"witness": corpus("valid"), "confirmation": setup()}
    summary = {}
    with gpu_owner():
        for arm in ("parent", "linear", "mixture"):
            reader, head = load(arm, "cuda")
            summary[arm] = {}
            for split, (rows, tiles) in datasets.items():
                bank = reader.ink(
                    torch.from_numpy(observations({"tiles": tiles}, "ink")).cuda()
                )
                swaps = exchanged(rows)
                details = []
                for change in (False, True):
                    stats = defaultdict(
                        lambda: dict(n=0, exact=0, matched=0, emitted=0, total=0)
                    )
                    for i in range(0, len(rows), 8):
                        batch = rows[i : i + 8]
                        qs = [
                            rows[swaps[j]] if change else rows[j]
                            for j in range(i, i + len(batch))
                        ]
                        inputs = [
                            dict(qi=q["qi"], pi=r["pi"])
                            for r, q in zip(batch, qs, strict=True)
                        ]
                        role, count, h = read_batch(reader, head, bank, inputs)
                        if split == "witness":
                            bounds = torch.from_numpy(
                                pack_paths([r["paths"] for r in batch])
                            )
                            scores, _ = score_paths(
                                role, count, h, bounds, arm == "mixture"
                            )
                        for j, r in enumerate(batch):
                            spans, value = decode(
                                role[j, : len(r["pi"])].numpy(),
                                count[j].numpy(),
                                h[j, : len(r["pi"])].numpy(),
                                arm == "mixture",
                            )
                            ink = [tiles[r["pi"][a:b]] for a, b in spans]
                            parts = [r["p"][a:b] for a, b in spans]
                            g = grade(parts, r["refs"])
                            group = (
                                "witness"
                                if split == "witness"
                                else r["lang"] + ("-multi" if multi(r) else "-single")
                            )
                            z = stats[group]
                            z["n"] += 1
                            z["exact"] += int(g["exact"])
                            fixed = grade(parts, r["refs"][:1])
                            for k in ("matched", "emitted"):
                                z[k] += fixed[k]
                            z["total"] += fixed["expected"]
                            rec = dict(
                                id=r["id"],
                                question_id=qs[j]["id"],
                                exchanged=change,
                                group=group,
                                spans=spans,
                                parts=parts,
                                exact=g["exact"],
                                path_utility=value,
                                ink=[sha256(v.tobytes()).hexdigest() for v in ink],
                            )
                            if split == "witness":
                                margins = (scores[j, :2] - scores[j, 2:]).tolist()
                                skip = int(role[j, : len(r["pi"]), 0].sum())
                                gap = value - skip - int(scores[j, :2].max())
                                if gap < 0:
                                    raise ValueError(
                                        "Free optimum below an admissible annotated path"
                                    )
                                rec.update(
                                    margins=margins,
                                    scores=scores[j].tolist(),
                                    free_over_best_supplied=gap,
                                )
                                for k, v in (
                                    ("both", all(d > 0 for d in margins)),
                                    ("one_or_more", any(d > 0 for d in margins)),
                                    ("ties", sum(d == 0 for d in margins)),
                                    ("strict", sum(d > 0 for d in margins)),
                                ):
                                    z[k] = z.get(k, 0) + int(v)
                                if not change and i + j == 0:
                                    dest = MODEL / "examples" / r["id"]
                                    dest.mkdir(parents=True, exist_ok=True)
                                    draw(dest, "page", tiles[r["pi"]])
                                    draw(dest, "question", tiles[r["qi"]])
                                    n = write(ink, dest / arm)
                                    dump(
                                        dest / arm / "receipt.json",
                                        dict(
                                            **rec,
                                            page_cells=len(r["pi"]),
                                            question_cells=len(r["qi"]),
                                            answer_cells=n,
                                        ),
                                    )
                            details.append(rec)
                    summary[arm][split + ("-exchanged" if change else "")] = dict(stats)
                packed(PUB / (arm + "-" + split + ".json.gz"), details)
    dump(PUB / "scores.json", summary)
    print(summary)


if __name__ == "__main__":
    from scripts.verse_gpu_runtime import configure_cuda_cli

    configure_cuda_cli()
    main()
