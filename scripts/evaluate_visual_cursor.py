"""Score free responses, both questions on a passage and retained general writing."""

from collections import Counter, defaultdict
import hashlib
import numpy as np
import torch
from ilm.visual_lm.book_prefix import observations
from scripts.prepare_visual_cursor import WORK, MODEL, PUBLIC, read, dump, sha
from scripts.train_visual_cursor import load_data
from scripts.train_visual_discourse import Corpus, load_model
from scripts.write_visual_discourse import generate as original
from scripts.write_visual_cursor import load_cursor, generate
from scripts.evaluate_visual_discourse import cohorts, draw, packed, transcript
from scripts.train_span_choice import gpu_owner


def normalize(s):
    return "".join(s.casefold().split()).strip(".,!。！？")


def overlap(a, b):
    a, b = normalize(a), normalize(b)
    return 2 * sum((Counter(a) & Counter(b)).values()) / max(1, len(a) + len(b))


@torch.inference_mode()
def main():
    from scripts.verse_gpu_runtime import configure_cuda_cli

    configure_cuda_cli()
    data = Corpus()
    pages = load_data()
    if sha(WORK / "held.json") != read(PUBLIC / "corpus.json")["test"]:
        raise ValueError("Changed test")
    natural = read(WORK / "held.json")
    refs, blocks = cohorts(data)
    ref = [r for r in refs if r["split"] == "winograd"]
    tests = []
    for task, rs in blocks.items():
        order = sorted(range(len(rs)), key=lambda i: len(rs[i]["x"]))
        if len(order) % 2:
            order.pop()
        tests.extend(dict(r, task=task) for i, r in enumerate(rs) if i in order)
    blank = next(i for i, s in enumerate(pages["chars"]) if s == " ")
    erased = []
    for r in natural:
        p = r["prompt"].index(r["question"][:190])
        erased.append(
            list(r["x"][:p])
            + [blank] * len(r["question"][:190])
            + list(r["x"][p + len(r["question"][:190]) :])
        )
    with gpu_owner():
        base = load_model("ink-pair", "cuda")
        features = torch.from_numpy(
            observations({"tiles": pages["tiles"]}, "ink")
        ).cuda()
        basebank = base.encode(features)
        for arm in ("base", "independent", "persistent"):
            model = None if arm == "base" else load_cursor(arm, "cuda")
            results = {}

            def run(xs, cap):
                outputs = []
                for j in range(0, len(xs), 24):
                    seq = xs[j : j + 24]
                    if model is None:
                        ys, stop = original(base, basebank, seq, data.n, cap)
                    else:
                        ys, stop = generate(
                            base, model, features, seq, data.n, cap, arm == "persistent"
                        )
                    outputs.extend(zip(ys, stop, strict=True))
                return outputs

            details = []
            for k, (xs, cap) in enumerate(
                (([r["x"] for r in natural], 96), (erased, 96))
            ):
                rows = []
                n = 0
                for r, (out, end) in zip(natural, run(xs, cap), strict=True):
                    text = transcript(pages["chars"], out)
                    exact = end and normalize(text) in list(
                        map(normalize, r["references"])
                    )
                    verb = any(
                        normalize(s) in normalize(r["prompt"]) for s in r["references"]
                    )
                    p = r["kind"]
                    idx = n % 128
                    if k == 0 and idx < 2:
                        folder = MODEL / "examples" / arm / p / str(idx)
                        folder.mkdir(exist_ok=True, parents=True)
                        draw(folder, "input", pages["tiles"][r["x"]])
                        draw(folder, "answer", pages["tiles"][out])
                        draw(folder, "reference", pages["tiles"][r["y"][:-1]])
                        dump(
                            folder / "readout.json",
                            dict(
                                question=r["question"],
                                answer=text,
                                references=r["references"],
                                prompt_cells=len(r["x"]),
                                emitted=len(out),
                                stopped=end,
                                sha256=hashlib.sha256(
                                    pages["tiles"][out].tobytes()
                                ).hexdigest(),
                            ),
                        )
                    rows.append(
                        dict(
                            doc=r["doc"],
                            kind=p,
                            question=r["question"],
                            references=r["references"],
                            verbatim=verb,
                            output=text,
                            stopped=end,
                            exact=bool(exact),
                            f1=max(overlap(text, s) for s in r["references"]),
                        )
                    )
                    n += 1
                for p in sorted({r["kind"] for r in rows}):
                    g = [r for r in rows if r["kind"] == p]
                    sg = defaultdict(list)
                    for r in g:
                        sg[r["doc"]].append(r)
                    assert all(len(v) == 2 for v in sg.values())
                    results[p + ("-erased" if k else "")] = dict(
                        n=len(g),
                        exact=sum(r["exact"] for r in g),
                        mean_glyph_f1=np.mean([r["f1"] for r in g]),
                        pairs=len(sg),
                        both=sum(all(r["exact"] for r in v) for v in sg.values()),
                        different=sum(
                            v[0]["output"] != v[1]["output"] for v in sg.values()
                        ),
                        absent=sum(not r["verbatim"] for r in g),
                        absent_exact=sum(r["exact"] and not r["verbatim"] for r in g),
                        stopped=sum(r["stopped"] for r in g),
                    )
                details.extend(dict(r, erased=bool(k)) for r in rows)
            packed(PUBLIC / (arm + "-human.json.gz"), details)
            groups = defaultdict(list)
            for r, (out, end) in zip(ref, run([r["x"] for r in ref], 40), strict=True):
                groups[r["pair"]].append(end and out == r["y"][:-1])
            results["winograd"] = dict(
                exact=sum(sum(v) for v in groups.values()),
                both=sum(all(v) for v in groups.values()),
                n=len(ref),
                pairs=len(groups),
            )
            openrows = []
            for r, (out, end) in zip(
                tests, run([r["x"] for r in tests], 96), strict=True
            ):
                kind = r["task"]
                openrows.append(
                    dict(
                        task=kind,
                        id=r["id"],
                        output=transcript(pages["chars"], out),
                        stopped=end,
                        first_eight=out[:8] == r["y"][:8],
                        prompt=transcript(pages["chars"], r["x"]),
                        reference=transcript(pages["chars"], r["y"]),
                    )
                )
            packed(PUBLIC / (arm + "-open.json.gz"), openrows)
            for kind in blocks:
                subset = [r for r in openrows if r["task"] == kind]
                results[kind] = dict(
                    n=len(subset),
                    first_eight=sum(r["first_eight"] for r in subset),
                    stopped=sum(r["stopped"] for r in subset),
                    first=subset[0]["output"],
                )
            dump(PUBLIC / (arm + "-evaluation.json"), results)
            print(arm, results, flush=True)


if __name__ == "__main__":
    main()
