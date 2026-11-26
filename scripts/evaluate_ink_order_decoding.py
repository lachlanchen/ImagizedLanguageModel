"""Check source-order contrast on human twins and ordinary optical translation."""

import numpy as np
import torch
from collections import Counter, defaultdict
from scripts.prepare_paired_ink import ROOT, BASE, PUB, WORK, sha, read, dump, key
from scripts.train_paired_ink import material
from scripts.train_ordered_translation import pack
from scripts.evaluate_proposition_translation import semantic
from scripts.train_span_choice import gpu_owner
from scripts.write_ordered_translation import restore, stopped
from scripts.verse_gpu_runtime import configure_cuda_cli
from ilm.visual_lm.order_decoding import write_with_order, bitmap_ranks


@torch.inference_mode()
def generate():
    old = read(WORK / "generated.json")
    rows = old["rows"]
    answers = [{"greedy": o["assignment"].copy()} for o in old["out"]]
    for row in rows:
        row["group"] = "paws"
    held = read(WORK / "held.json")
    lookup = read(BASE / "readback.json")
    sources = [h["cells"][v][1 - t] for h in held for t in (0, 1) for v in (0, 1)]
    targets = [h["cells"][v][t] for h in held for t in (0, 1) for v in (0, 1)]
    excluded = [set() for _ in range(2)]
    for r in read(WORK / "train.json") + held:
        for z in r["cells"]:
            for t in (0, 1):
                excluded[t].add(key("".join(lookup[str(c)] for c in z[t])))
    bank = read(ROOT / "artifacts/bilingual_ink_v1/held.json")
    natural = [
        r
        for r in bank[32:]
        if not any(key(s) in excluded[t] for t, s in enumerate(r["text"]))
    ][:48]
    if len(natural) != 48:
        raise ValueError("OPUS size")
    for j, r in enumerate(natural):
        for t in (0, 1):
            rows.append(
                dict(
                    group="opus",
                    id=f"opus-{j}/{t}",
                    t=t,
                    source=r["text"][1 - t],
                    ref=r["text"][t],
                    foil=natural[(j + 24) % 48]["text"][t],
                )
            )
            sources.append(r["cells"][1 - t])
            targets.append(r["cells"][t])
            answers.append({})
    net = restore("paired-assignment", "cuda")
    f, inv, _ = material("cuda")
    rank = bitmap_ranks(np.load(BASE / "palette.npy"), "cuda")
    for t in (0, 1):
        ids = [j for j, r in enumerate(rows) if r["t"] == t]
        for k in range(0, len(ids), 24):
            group = ids[k : k + 24]
            x = pack([sources[j] for j in group], "cuda")
            for m, pred in (
                ("greedy", net.generate(f, x, inv[t], t)),
                ("order", write_with_order(net, f, x, inv[t], t, rank)),
            ):
                for j, y in zip(group, pred.cpu(), strict=True):
                    cells, stop = stopped(y)
                    if m in answers[j] and cells != answers[j][m]["cells"]:
                        raise ValueError("Greedy replay changed")
                    answers[j][m] = dict(
                        cells=cells,
                        stop=stop,
                        raw_exact=stop and cells == targets[j],
                        text="".join(lookup[str(c)] for c in cells),
                    )
    return rows, answers


def main():
    configure_cuda_cli()
    with gpu_owner():
        rows, ans = generate()
        dump(WORK / "order-generated.json", dict(rows=rows, out=ans))
        torch.cuda.empty_cache()
        report = semantic(rows, ans, ("greedy", "order"))
    stats = defaultdict(Counter)
    groups = defaultdict(lambda: defaultdict(list))
    for i, (r, a) in enumerate(zip(rows, ans, strict=True)):
        group = r["group"]
        g = stats[group]
        g["n"] += 1
        g["changed"] += a["order"]["cells"] != a["greedy"]["cells"]
        success = []
        for m in ("greedy", "order"):
            g[m + "_exact"] += a[m]["raw_exact"]
            g[m + "_stop"] += a[m]["stop"]
            success.append(
                int(a[m]["stop"] and all(report["votes"][2 * i + k][m] for k in (0, 1)))
            )
        groups[group][r["id"].split("/")[0]].append(success[1] - success[0])
    ci = {}
    rng = np.random.default_rng(27259)
    for g, families in groups.items():
        v = np.array([np.mean(v) for v in families.values()])
        ci[g] = np.quantile(
            v[rng.integers(len(v), size=(10000, len(v)))].mean(axis=1), [0.025, 0.975]
        ).tolist()
    dump(WORK / "order-graded.json", dict(rows=rows, out=ans))
    dump(
        PUB / "order_decoding.json",
        dict(
            free=stats,
            paired_family_difference_ci=ci,
            semantic=report,
            rows_sha256=sha(WORK / "order-graded.json"),
        ),
    )
    print(report["groups"], flush=True)


if __name__ == "__main__":
    main()
