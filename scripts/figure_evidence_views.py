"""Export the actual generated glyphs and their source questions for print and replay."""

from pathlib import Path
import argparse
from hashlib import sha256
import numpy as np
import torch
from scripts.evaluate_evidence_views import groups, official_index
from scripts.evaluate_visual_discourse import draw, unpack
from scripts.prepare_evidence_views import Alphabet, MODEL, PUB, WORK, TRAIN
from scripts.prepare_open_exam import read, dump
from scripts.figure_recurrent_evidence import Plate, MUTED, TEAL, RUST
from scripts.write_open_exam import Writer, infer, load
from scripts.train_open_exam import corpus


def sample(root, work, report):
    judge, _, _, dev, _ = official_index()
    gs = groups(judge, dev)
    rows = gs["composition"]
    seen = {
        judge.normalize_answer(a["input_text"])
        for d in read(TRAIN)["data"]
        for a in d["answers"]
    }
    chinese, _, _ = corpus("train")
    seen.update(judge.normalize_answer(r["answer"]) for r in chinese)
    a = Alphabet()
    # Reconstruct the decoding alphabet in the original evaluation traversal order.
    for values in gs.values():
        for r in values:
            a.encode(r["question"])
            a.encode(r["page"])
    tiles = a.array()
    answers = {}
    for arm in ("parent", "whole", "highlight"):
        answers[arm] = {
            (z["doc"], z["turn"]): z
            for z in unpack(work / (arm + "-responses.json.gz"))
            if z["group"] in ("coqa", "composition")
        }

    def key(r):
        return (r["doc"], r["turn"])

    chosen = next(
        (
            r
            for r in rows
            if answers["highlight"][key(r)]["exact"]
            and not answers["whole"][key(r)]["exact"]
            and judge.normalize_answer(answers["highlight"][key(r)]["text"]) not in seen
        ),
        None,
    )
    if chosen is None:
        chosen = next(
            (
                r
                for r in rows
                if answers["highlight"][key(r)]["exact"]
                and not answers["whole"][key(r)]["exact"]
            ),
            None,
        )
    if chosen is None:
        chosen = next((r for r in rows if answers["highlight"][key(r)]["exact"]), None)
    first_group = "composition"
    if chosen is None:
        first_group = "coqa"
        chosen = next(
            (r for r in gs["coqa"] if answers["highlight"][key(r)]["exact"]),
            gs["coqa"][0],
        )
    other = next(
        r
        for r in rows
        if r["doc"] != chosen["doc"] and not answers["highlight"][key(r)]["exact"]
    )
    highlights = {
        (d["id"], a["turn_id"]): (a["span_start"], a["span_end"])
        for d in read(dev)["data"]
        for a in d["answers"]
    }
    data = []
    writer = Writer()
    models = {arm: load(arm, "cpu", root) for arm in answers}
    for i, r in enumerate((chosen, other)):
        d = root / "examples" / str(i + 1)
        d.mkdir(exist_ok=True, parents=True)
        q = a.encode(r["question"])
        p = a.encode(r["page"])
        raw = a.encode(r["raw_q"])
        reference = a.encode(r["refs"][0])
        # The published figure shows the current question; the complete model input
        # also contains the previous exchange on follow-up turns.
        lo, hi = highlights[key(r)]
        if not 0 <= lo < hi <= len(p):
            lo, hi = 0, len(p)
        cut = p[lo : min(lo + 72, hi)]
        draw(d, "source-excerpt", a.array()[cut])
        draw(d, "question", a.array()[raw])
        draw(d, "reference", a.array()[reference])
        seen_outputs = {}
        for arm, model in models.items():
            rec = answers[arm][key(r)]
            original = tiles[np.asarray(rec["ids"], int)]
            assert sha256(original.tobytes()).hexdigest() == rec["sha256"]
            draw(d, arm, original)
            pix, stop = infer(writer, model, tiles[q], tiles[p], geometry=(320, 4096))
            seen_outputs[arm] = dict(
                cells=len(original),
                exact=rec["exact"],
                stopped=rec["stop"],
                text=rec["text"],
                sha256=rec["sha256"],
                cpu_matches=stop == rec["stop"] and np.array_equal(pix, original),
            )
        item = dict(
            group=first_group if i == 0 else "composition",
            doc=r["doc"],
            turn=r["turn"],
            question=r["raw_q"],
            prior_context=r["question"] != r["raw_q"],
            question_cells=len(raw),
            input_question_cells=len(q),
            page_cells=len(p),
            reference=r["refs"][0],
            ref_cells=len(reference),
            excerpt_cells=len(cut),
            excerpt_start=lo,
            output=seen_outputs,
        )
        dump(d / "receipt.json", item)
        data.append(item)
    dump(report / "figure-cases.json", data)
    return data


def figure(root, report, cases):
    assets = report / "plates"
    assets.mkdir(parents=True, exist_ok=True)
    p = Plate("Figure_162_writing_from_evidence", assets)
    for i, r in enumerate(cases):
        top = i * 138
        d = root / "examples" / str(i + 1)
        p.label(5, top + 1, "ab"[i], 12, True)
        p.label(
            23,
            top + 2,
            f'Conversation turn {r["turn"]}  ·  {r["page_cells"]:,} input cells in the story',
            10,
            True,
        )
        p.label(8, top + 24, "Story excerpt", 8, True, MUTED)
        p.strip(
            str(i) + "-p",
            d / "source-excerpt.png",
            r["excerpt_cells"],
            98,
            top + 20,
            433,
            17,
        )
        p.label(8, top + 48, "Current question", 8, True, MUTED)
        p.strip(
            str(i) + "-q",
            d / "question.png",
            r["question_cells"],
            98,
            top + 44,
            433,
            17,
        )
        xs = (8, 181, 353)
        for x, (name, tag, ct) in zip(
            xs,
            [
                ("Human reference", "reference", r["ref_cells"]),
                ("Full-story control", "whole", r["output"]["whole"]["cells"]),
                (
                    "Evidence-trained writer",
                    "highlight",
                    r["output"]["highlight"]["cells"],
                ),
            ],
        ):
            p.label(x, top + 69, name, 8, True, TEAL if tag == "highlight" else MUTED)
            if ct:
                if ct <= 28:
                    p.strip(str(i) + tag, d / (tag + ".png"), ct, x, top + 86, 148, 21)
                else:
                    mid = (ct + 1) // 2
                    p.strip(
                        str(i) + tag + "1",
                        d / (tag + ".png"),
                        ct,
                        x,
                        top + 84,
                        165,
                        16,
                        0,
                        mid,
                    )
                    p.strip(
                        str(i) + tag + "2",
                        d / (tag + ".png"),
                        ct,
                        x,
                        top + 103,
                        165,
                        16,
                        mid,
                    )
            else:
                p.label(x, top + 89, "[empty]", 9, False, RUST)
    p.rule(129)
    dump(report / "plates.json", [p.render()])


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", type=Path, default=MODEL)
    p.add_argument("--work", type=Path, default=WORK)
    p.add_argument("--report", type=Path, default=PUB)
    z = p.parse_args()
    torch.set_num_threads(3)
    with torch.inference_mode():
        figure(z.model_dir, z.report, sample(z.model_dir, z.work, z.report))
