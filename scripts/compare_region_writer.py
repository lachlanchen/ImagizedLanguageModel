"""Run the previous optical QA2D pipeline and a local Q4 vision reader on the held inputs."""

from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from time import monotonic
import numpy as np
import torch

from ilm.visual_lm.book_prefix import observations
from scripts.benchmark_competing_qwen import MODELS, STORE, png, server
from scripts.benchmark_open_exam import VLM, ask
from scripts.evaluate_region_writer import e2e_scorer, score_rows, summarize
from scripts.score_dialogue_ink_references import official_index
from scripts.prepare_evidence_views import Alphabet
from scripts.prepare_region_writer import MODEL, WORK, PUB, Ink
from scripts.prepare_open_exam import read, dump, sha
from scripts.write_recurrent_evidence import generate
from scripts.write_visual_statements import editor, fact_page
from scripts.write_open_exam import load
from scripts.write_visual_discourse import Writer
from scripts.train_paired_evidence import gpu_owner
from scripts.measure_writer_energy import snapshot, consumed
from scripts.write_region_writer import model, answer


def compare(rows=None, public=PUB, phrase_checkpoint=None):
    if rows is None:
        rows = read(WORK / "dev-source.json")
    # Reassemble the prior writer's page with visible gaps for region separation.
    a = Alphabet()
    sep = a.encode("  ")
    space = sep[0]
    qs = [a.encode(r["q"]) for r in rows]
    pages = [a.encode("  ".join(r["blocks"])) for r in rows]
    tiles = a.array()
    measure = e2e_scorer()
    norm = official_index()[0].normalize_answer
    with gpu_owner(), torch.inference_mode():
        w = Writer(device="cuda")
        prior = load("whole", "cuda", MODEL.parent / "evidence_views")
        dec = editor("cuda")
        x = torch.from_numpy(observations({"tiles": tiles}, "ink")).cuda()
        old = []
        oldfacts = []
        for j in range(0, len(rows), 8):
            q, p = qs[j : j + 8], pages[j : j + 8]
            facts = generate(
                w.model,
                prior,
                x,
                a.n,
                q,
                p,
                [s + t for s, t in zip(p, q)],
                64,
                geometry=(320, 4096),
            )
            records = [fact_page(t, f, space) for t, (f, end) in zip(q, facts)]
            written = generate(
                w.model, dec, x, a.n, q, records, records, 256, geometry=(320, 4096)
            )
            for target, vals in ((old, written), (oldfacts, facts)):
                for ids, end in vals:
                    target.append(
                        (
                            "".join(a.chars[i] for i in ids),
                            ids,
                            end,
                            None,
                            sha256(tiles[np.asarray(ids, int)].tobytes()).hexdigest(),
                        )
                    )

        def count(values, subset):
            scoring_rows = [
                {**s, "task": "catalogue"} if s["task"] == "exam" else s for s in subset
            ]
            r = score_rows(values, scoring_rows, measure, norm)
            for v, (t, *_), s in zip(r, values, subset):
                v["task"] = s["task"]
                v["visible_reply"] = bool(t) and any(
                    t in b for b in [s["q"], *s["blocks"]]
                )
            return summarize(r)

        out = {
            "previous_full_story": count(oldfacts, rows),
            "previous_story_plus_statement_writer": count(old, rows),
        }
    # First eight items in the sealed order per domain. Use identical ink in two grids.
    subset = [
        r
        for t in ("statements", "catalogue", "restaurant", "exam")
        for r in [z for z in rows if z["task"] == t][:8]
    ]
    ink = Ink(np.load(MODEL / "palette.npy"))
    qb = [ink.encode(r["q"]) for r in subset]
    pages = [[ink.encode(s) for s in r["blocks"]] for r in subset]
    native = ink.array()
    book = read(MODEL / "readback.json")
    chars = dict(enumerate(book))
    for c, i in ink.chars.items():
        chars.setdefault(i, c)

    # Separate the ordered regions visually with a blank row on a 32-column grid.
    def page_grid(parts):
        grid = []
        for i, part in enumerate(parts):
            if i:
                grid.extend([0] * ((-len(grid)) % 32 + 32))
            grid.extend(part)
        return grid

    msgs = [
        dict(
            role="user",
            content="The first image is the question. The second is the source page. Blank rows separate regions. Answer the question in one complete natural sentence in its original language. For descriptions, include all listed facts. Write only the answer.",
            images=[png(native, q), png(native, page_grid(p))],
        )
        for q, p in zip(qb, pages)
    ]
    if sha(STORE / "blobs" / ("sha256-" + MODELS[VLM])) != MODELS[VLM]:
        raise ValueError("Changed vision model")
    costs = {}
    paired = {}
    with gpu_owner(), torch.inference_mode():
        selector = model("selector", "cuda")
        loaders = [
            (a, a, lambda arm=a: model(arm, "cuda")) for a in ("direct", "selected")
        ]
        if phrase_checkpoint is not None:
            from scripts.write_region_writer import phrase_model

            loaders.append(
                ("phrase", "selected", lambda: phrase_model(phrase_checkpoint, "cuda"))
            )
        for label, arm, load_net in loaders:
            net = load_net()
            vals = []
            answer(net, selector, arm, native, len(book), [qb[0]], [pages[0]], "cuda")
            torch.cuda.synchronize()
            timer = monotonic()
            begin = snapshot()
            for q, p in zip(qb, pages):
                res, _ = answer(net, selector, arm, native, len(book), [q], [p], "cuda")
                ids, stop = res[0]
                vals.append(("".join(chars[i] for i in ids), ids, stop, None, ""))
            torch.cuda.synchronize()
            costs[label] = dict(
                seconds=monotonic() - timer,
                joules=consumed(begin, snapshot())["joules"],
            )
            paired[label] = count(vals, subset)
        with server() as url:
            ask(url, VLM, dict(role="user", content="只回答好。"))
            timer = monotonic()
            begin = snapshot()
            with ThreadPoolExecutor(max_workers=2) as pool:
                replies = list(pool.map(lambda m: ask(url, VLM, m), msgs))
            costs["qwen"] = dict(
                seconds=monotonic() - timer,
                joules=consumed(begin, snapshot())["joules"],
            )
        paired["qwen"] = count([(t, list(t), e, None, "") for t, e in replies], subset)
    dump(
        public / "comparators.json",
        dict(
            full=out,
            image_subset=paired,
            cost=costs,
            vision_hash=MODELS[VLM],
            ids=[dict(task=r["task"], family=r["family"]) for r in subset],
            qwen=[dict(text=t, stop=e) for t, e in replies],
        ),
    )


if __name__ == "__main__":
    from scripts.verse_gpu_runtime import configure_cuda_cli

    configure_cuda_cli()
    compare()
