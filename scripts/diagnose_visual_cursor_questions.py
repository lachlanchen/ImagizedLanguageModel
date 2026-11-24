"""Contrast two complete human answers under two questions with identical passage ink."""

from collections import defaultdict
import torch

from ilm.visual_lm.book_prefix import observations
from scripts.prepare_visual_cursor import (
    WORK,
    PUBLIC,
    FILES,
    ROOT,
    hash_text,
    records,
    page,
    read,
    dump,
    sha,
    LIMIT,
)
from scripts.prepare_visual_sequence_corpus import Rasterizer
from scripts.train_visual_cursor import load_data, base_batch, pad
from scripts.train_visual_discourse import Corpus, load_model
from scripts.write_visual_cursor import load_cursor
from scripts.evaluate_visual_discourse import packed
from scripts.train_span_choice import gpu_owner


@torch.inference_mode()
def main():
    from scripts.verse_gpu_runtime import configure_cuda_cli

    configure_cuda_cli()
    data = Corpus()
    pages = load_data()
    held = read(WORK / "held.json")
    tests = []
    notes = []
    raster = Rasterizer()
    by = {bytes(g): i for i, g in enumerate(pages["tiles"]) if i >= 2}
    docs = {}
    for k, files in FILES.items():
        path, digest = files[1]
        src = ROOT / "data/raw" / path
        if sha(src) != digest:
            raise ValueError(k)
        for passage, q, refs, hist in records(k, src):
            docs[hash_text(passage)] = passage
    for i in range(0, len(held), 2):
        left, right = held[i : i + 2]
        assert left["doc"] == right["doc"]
        k = left["kind"]
        p = docs[left["doc"]]
        prompts = [page(k, r["question"], "", "") for r in (left, right)]
        common = p[: LIMIT - max(map(len, prompts))]
        queries = [page(k, r["question"], "", common) for r in (left, right)]
        ys = [left["y"], right["y"]]
        notes.append(
            dict(
                doc=left["doc"],
                kind=k,
                clipped=len(common) < len(p),
                visible_passage_cells=len(common),
                both_in_passage=all(
                    any(x.casefold() in common.casefold() for x in r["references"])
                    for r in (left, right)
                ),
            )
        )
        for q in queries:
            try:
                x = [by[raster.tile(c)] for c in q]
            except (KeyError, ValueError):
                tests.extend([None, None])
                continue
            for y in ys:
                tests.append(None if len(y) > 112 else (x, y))
    with gpu_owner():
        base = load_model("ink-pair", "cuda")
        features = torch.from_numpy(
            observations({"tiles": pages["tiles"]}, "ink")
        ).cuda()
        basebank = base.encode(features)
        valid = [i for i, v in enumerate(tests) if v is not None]
        for arm in ("base", "independent", "persistent"):
            model = None if arm == "base" else load_cursor(arm, "cuda")
            score = {}
            for j in range(0, len(valid), 24):
                idx = valid[j : j + 24]
                x, y = zip(*(tests[i] for i in idx))
                ids, t = (torch.as_tensor(pad(v), device="cuda") for v in (x, y))
                bi, bp = (torch.as_tensor(a, device="cuda") for a in base_batch(x, y))
                lp = base.log_prob(basebank, bi, data.n, bp)
                if model is not None:
                    lp = model.likelihood(
                        features, ids, t, lp.exp(), data.n, arm == "persistent"
                    )
                for i, s in zip(idx, ((lp * (t > 0)).sum(1)).tolist(), strict=True):
                    score[i] = s
            summary = defaultdict(list)
            details = []
            for j, info in enumerate(notes):
                if not all(4 * j + n in score for n in range(4)):
                    continue
                aa, ab, ba, bb = [score[4 * j + n] for n in range(4)]
                d = dict(
                    **info,
                    true_a=aa,
                    other_a=ab,
                    other_b=ba,
                    true_b=bb,
                    correct_assignment=(aa + bb > ab + ba + 1e-6),
                    tie=abs(aa + bb - ab - ba) <= 1e-6,
                    both_rank=aa > ab and bb > ba,
                    both_prefer_own_question=aa > ba and bb > ab,
                )
                summary[info["kind"]].append(d)
                details.append(d)
            result = {
                k: dict(
                    pairs=len(v),
                    both_rank=sum(d["both_rank"] for d in v),
                    correct_assignment=sum(d["correct_assignment"] for d in v),
                    ties=sum(d["tie"] for d in v),
                    both_question=sum(d["both_prefer_own_question"] for d in v),
                    clipped=sum(d["clipped"] for d in v),
                    both_in_passage=sum(d["both_in_passage"] for d in v),
                )
                for k, v in summary.items()
            }
            packed(PUBLIC / (arm + "-matched-page-scores.json.gz"), details)
            dump(PUBLIC / (arm + "-matched-page.json"), result)
            print(arm, result, flush=True)


if __name__ == "__main__":
    main()
