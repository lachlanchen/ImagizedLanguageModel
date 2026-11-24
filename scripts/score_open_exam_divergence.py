"""Compare two next-cell masses at the first different target after an identical ink prefix."""

from collections import defaultdict
import torch
from ilm.visual_lm.book_prefix import observations
from scripts.evaluate_visual_discourse import unpack
from scripts.prepare_open_exam import PUB, WORK, dump
from scripts.train_open_exam import corpus
from scripts.train_paired_evidence import gpu_owner
from scripts.write_open_exam import load
from scripts.write_recurrent_evidence import generate
from scripts.write_visual_discourse import Writer


def main():
    rows, tiles, n = corpus("held")
    groups = defaultdict(list)
    for row in rows:
        if row["cohort"] == "paired":
            groups[row["doc"]].append(row)
    output = {}
    with gpu_owner(), torch.inference_mode():
        writer = Writer(device="cuda")
        x = torch.from_numpy(observations({"tiles": tiles}, "ink")).cuda()
        for arm in ("exchanged", "matched", "parent"):
            net = load(arm, "cuda")
            details = []
            records = {
                r["id"]: r
                for r in unpack(WORK / (arm + "-responses.json.gz"))
                if r["condition"] == "intact"
            }
            for pair in groups.values():
                k = next(
                    i
                    for i, (a, b) in enumerate(zip(pair[0]["y"], pair[1]["y"]))
                    if a != b
                )
                targets = [r["y"][: k + 1] for r in pair]
                q = [r["qi"] for r in pair for _ in pair]
                s = [r["pi"] for r in pair for _ in pair]
                assert s[0] == s[2] and targets[0][:-1] == targets[1][:-1]
                masses = generate(
                    writer.model,
                    net,
                    x,
                    n,
                    q,
                    s,
                    [p + z for p, z in zip(s, q)],
                    k + 1,
                    forced=torch.tensor(targets * 2, device=x.device),
                )[:, -1].tolist()
                margins = [masses[0] - masses[1], masses[3] - masses[2]]
                emitted = [
                    records[r["id"]]["ids"] + ([1] if records[r["id"]]["stop"] else [])
                    for r in pair
                ]
                details.append(
                    dict(
                        doc=pair[0]["doc"],
                        index=k,
                        margins=margins,
                        both_preferred=min(margins) > 0,
                        both_free_prefix=all(
                            a[: k + 1] == b for a, b in zip(emitted, targets)
                        ),
                        both_free_complete=all(records[r["id"]]["exact"] for r in pair),
                    )
                )
            output[arm] = dict(
                pairs=len(details),
                both_preferred=sum(r["both_preferred"] for r in details),
                both_free_prefix=sum(r["both_free_prefix"] for r in details),
                both_free_complete=sum(r["both_free_complete"] for r in details),
                first_cell_disagreement=sum(r["index"] == 0 for r in details),
                rows=details,
            )
    dump(PUB / "divergence.json", output)


if __name__ == "__main__":
    from scripts.verse_gpu_runtime import configure_cuda_cli

    configure_cuda_cli()
    main()
