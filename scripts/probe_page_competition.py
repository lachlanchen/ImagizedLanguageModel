"""Compare free, best-supported and fixed parent outputs, including actual training pages."""

from collections import defaultdict
from hashlib import sha256
from statistics import median
import numpy as np
import torch
from ilm.visual_lm.book_prefix import observations
from ilm.visual_lm.multiple_ink_runs import best_path
from ilm.visual_lm.page_competition import best_reference, spans
from scripts.audit_visible_answer_sets import placement
from scripts.evaluate_page_competition import setup, ARMS, release
from scripts.prepare_page_competition import PUB, WORK, read, dump, sha
from scripts.prepare_multiple_ink_runs import normalize, multi
from scripts.write_multiple_ink_runs import load_reader, forward
from scripts.evaluate_multiple_ink_runs import grade
from scripts.evaluate_visual_discourse import packed


def options(r):
    cache = {}
    result = []
    for ref in r["refs"]:
        words = list({normalize(w): w for w in reversed(ref)}.values())
        if placement(r["p"], words, cache) == "placed":
            result.append([[(i, i + len(w)) for i in cache[w]] for w in words])
    return result


def group(r):
    return r["lang"] + ("-multi" if multi(r) else "-single")


def training():
    src = WORK / "training.json"
    if sha(src) != read(PUB / "corpus.json")["files"][src.name]:
        raise ValueError("Modified training archive")
    rows = read(src)
    out = []
    for g, n in (("en-multi", 48), ("zh-multi", 112), ("zh-single", 32)):
        pool = sorted(
            (r for r in rows if group(r) == g),
            key=lambda r: sha256(
                ("page-competition-training-control " + r["id"]).encode()
            ).digest(),
        )
        out += pool[:n]
    return out, np.load(WORK / "training-tiles.npy")


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    held, palette = setup()
    for r in held:
        r["options"] = options(r)
    result = {}
    for name, (rows, tiles) in (("held", (held, palette)), ("training", training())):
        parent = {}
        summ = {}
        detail = []
        for arm in ARMS:
            net = load_reader("cpu", True, release(arm))
            bank = net.ink(torch.from_numpy(observations({"tiles": tiles}, "ink")))
            groups = defaultdict(list)
            for i in range(0, len(rows), 8):
                batch = rows[i : i + 8]
                role, count = forward(
                    net, bank, [r["qi"] for r in batch], [r["pi"] for r in batch], True
                )
                for r, u, c in zip(batch, role, count, strict=True):
                    u = np.rint(1000 * u[: len(r["pi"])].astype(np.float64)).astype(
                        np.int64
                    )
                    c = np.rint(1000 * c.astype(np.float64)).astype(np.int64)
                    tags, free = best_path(u, c)
                    parts = [r["p"][a:b] for a, b in spans(tags)]
                    exact = grade(parts, r["refs"])["exact"]
                    if arm == "parent":
                        parent[r["id"]] = (tags, exact)
                    old, ok = parent[r["id"]]
                    old_score = int(
                        u[np.arange(len(old)), old].sum() + c[(old == 1).sum()]
                    )
                    d = dict(
                        id=r["id"],
                        arm=arm,
                        group=group(r),
                        exact=exact,
                        parent_exact=ok,
                    )
                    if r["options"]:
                        _, gold = best_reference(u, c, r["options"])
                        d.update(
                            gap=(free - gold) / 1000,
                            gold_minus_parent=(gold - old_score) / 1000,
                        )
                    groups[group(r)].append(d)
                    detail.append(d)
            summ[arm] = {}
            for g, ds in groups.items():
                visible = [d for d in ds if "gap" in d]
                errors = [d for d in visible if not d["parent_exact"]]
                summ[arm][g] = dict(
                    n=len(ds),
                    exact=sum(d["exact"] for d in ds),
                    aligned=len(visible),
                    median_gap=median(d["gap"] for d in visible),
                    parent_errors=len(errors),
                    reference_beats_parent_path=sum(
                        d["gold_minus_parent"] > 0 for d in errors
                    ),
                    errors_fixed=sum(d["exact"] for d in errors),
                )
        result[name] = summ
        packed(PUB / (name + "-fixed-paths.json.gz"), detail)
    dump(PUB / "fixed-paths.json", result)


if __name__ == "__main__":
    main()
