"""Run warm single-request image and text inference on the four reserved works."""

import json
import re
from time import monotonic
import torch
from ilm.visual_lm.prepared_glyph_responder import PreparedGlyphResponder
from ilm.visual_lm.causal_glyph_writer import packed_page
from scripts.benchmark_interval_teacher import MODEL, SYSTEM, DIGEST
from scripts.benchmark_bilingual_novel import tally
from scripts.evaluate_page_evidence_writer import tensor
from scripts.evaluate_pointing_causal_writer import load_streams
from scripts.local_translation_teacher import local_teacher
from scripts.reserve_first_glyph_books import PAGES, reserved
from scripts.train_bilingual_novel_writer import PUBLIC as OLD
from scripts.train_causal_glyph_writer import record, sha
from scripts.train_first_glyph_book import WORK, PUBLIC
from scripts.train_span_choice import gpu_owner
from scripts.train_typeface_writer import Face, FACES


def shelf(model):
    directory = OLD / "bookshelf"
    face = Face(FACES[0])
    rows = json.loads((directory / "result.json").read_text())["rows"]
    flags = []
    for r in rows:
        p = packed_page(directory / f"page_{r['page']}.png")
        q = packed_page(directory / f"question_{r['person']}_{r['lang']}.png")
        y, stop = model.answer_many([(p, q)])[0]
        flags.append(
            bool(stop and y.numpy().tobytes() in [face(t) for t in r["accepted"]])
        )
    return dict(exact=sum(flags), flags=flags)


@torch.inference_mode()
def run():
    folder = WORK / "serving"
    folder.mkdir(exist_ok=True)
    pages = json.loads(PAGES.read_text())
    groups = reserved()
    report, correct = {}, {}
    for arm in ("first", "later"):
        model = PreparedGlyphResponder(WORK / arm / "page", "cuda")
        report[arm + "_shelf"] = shelf(model)
        if arm == "first":
            for lang in ("zh", "en"):
                cases = groups["fresh_" + lang]
                batch = [(tensor(c.p), tensor(c.q)) for c in cases]
                model.answer_many(batch[:1])
                torch.cuda.synchronize()
                start = monotonic()
                generated = [model.answer_many([item])[0] for item in batch]
                torch.cuda.synchronize()
                elapsed = monotonic() - start
                ys, ends = load_streams(
                    WORK / "evaluation/first" / ("fresh_" + lang + ".npz")
                )
                flags, different = [], 0
                for c, (im, stop), y, end in zip(
                    cases, generated, ys, ends, strict=True
                ):
                    raw = im.numpy().tobytes()
                    different += (raw, bool(stop)) != (y, bool(end))
                    flags.append(c.grade(raw, stop)[2])
                correct["image_" + lang] = flags
                report["image_" + lang] = dict(
                    **tally(flags), seconds=elapsed, different=different
                )
        del model
        torch.cuda.empty_cache()
    ledger = folder / "qwen.jsonl"
    rows = (
        [json.loads(line) for line in ledger.read_text().splitlines()]
        if ledger.exists()
        else []
    )
    jobs = [
        (c.question_id, lang) for lang in ("zh", "en") for c in groups["fresh_" + lang]
    ]
    assert [(r["id"], r["lang"]) for r in rows] == jobs[: len(rows)]
    with local_teacher(MODEL, folder) as (ask, digest, size), ledger.open("a") as f:
        assert digest == DIGEST

        def query(p, q):
            return ask(
                SYSTEM,
                dict(role="user", content=f"材料：{p}\n问题：{q}"),
                think=False,
                options=dict(
                    seed=241179,
                    temperature=0,
                    num_predict=80,
                    num_ctx=2048,
                    num_thread=2,
                ),
            )

        query("小河流入大海。", "小河流入哪裡？")
        for id_, lang in jobs[len(rows) :]:
            key, i = id_.split("/")
            row = pages[key]
            q = row["questions"][int(i)]
            start = monotonic()
            response = query(row["page"], q[lang])
            response.update(id=id_, lang=lang, seconds=monotonic() - start)
            f.write(json.dumps(response, ensure_ascii=False) + "\n")
            f.flush()
            rows.append(response)
    for lang in ("zh", "en"):
        chosen = [r for r in rows if r["lang"] == lang]
        flags = []
        for r in chosen:
            key, i = r["id"].split("/")
            flags.append(
                r["stop"] == "stop"
                and r["text"] in pages[key]["questions"][int(i)]["refs"]
            )
        correct["qwen_" + lang] = flags
        report["qwen_" + lang] = dict(
            **tally(flags), seconds=sum(r["seconds"] for r in chosen)
        )
    for system in ("image", "qwen"):
        both = [
            a and b
            for a, b in zip(
                correct[system + "_zh"], correct[system + "_en"], strict=True
            )
        ]
        report[system + "_four"] = dict(
            complete=tally(both)["both"], boards=len(both) // 2
        )
    report.update(
        manifest=digest,
        text_bytes=size,
        ledger=sha(ledger),
        gpu_layers=re.findall(
            r"offloaded (\d+/\d+) layers to GPU", (folder / "server.log").read_text()
        ),
    )
    record(PUBLIC / "serving.json", report)


if __name__ == "__main__":
    with gpu_owner():
        run()
