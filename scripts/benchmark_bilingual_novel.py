"""Time both question languages on the same 64 Chinese BiPaR validation passages."""

import json
import re
from time import monotonic
import torch
from ilm.visual_lm.prepared_glyph_responder import PreparedGlyphResponder
from scripts.benchmark_interval_teacher import MODEL, SYSTEM, DIGEST
from scripts.evaluate_page_evidence_writer import tensor
from scripts.evaluate_pointing_causal_writer import load_streams, store_streams
from scripts.train_bilingual_novel_writer import opposite
from scripts.local_translation_teacher import local_teacher
from scripts.train_causal_glyph_writer import record, sha
from scripts.train_span_choice import gpu_owner
from scripts.train_bilingual_novel_writer import PUBLIC, WORK
from scripts.train_typeface_writer import FACES, Face
from scripts.typeface_writer_cases import Styled


def tally(right):
    return dict(
        n=len(right),
        exact=sum(right),
        both=sum(a and b for a, b in zip(right[::2], right[1::2], strict=True)),
    )


def run():
    folder = WORK / "serving"
    folder.mkdir(exist_ok=True)
    path = WORK / "valid.jsonl"
    assert (
        sha(path)
        == json.loads((PUBLIC / "corpus.json").read_text())["splits"]["valid"]["sha256"]
    )
    source = {}
    for line in path.read_text().splitlines()[:64]:
        row = json.loads(line)
        qs = row["questions"]
        pair = next(
            (a, b) for j, a in enumerate(qs) for b in qs[j + 1 :] if opposite(a, b)
        )
        for q in pair:
            source[q["id"]] = (row["page"], q)
    ids = list(source)
    assert len(ids) == 128
    face = canon = Face(FACES[0])
    model = PreparedGlyphResponder(WORK / "bilingual/page", "cuda")
    report = {
        "model_bytes": sum(
            p.stat().st_size for p in (WORK / "bilingual/page").iterdir()
        )
    }
    rows = []
    flags = {}
    for lang in ("zh", "en"):
        items = []
        cases = []
        started = monotonic()
        for i, id_ in enumerate(ids):
            p, z = source[id_]
            question, refs = z[lang], z["refs"]
            c = Styled(
                str(i // 2),
                face(p),
                face(question),
                tuple(map(canon, refs)),
                tuple(map(face, refs)),
                id_,
            )
            cases.append(c)
            items.append((tensor(c.p), tensor(c.q)))
            rows.append((id_, lang, p, question, refs))
        prep = monotonic() - started
        model.answer_many(items[:1])
        torch.cuda.synchronize()
        started = monotonic()
        outputs = [model.answer_many([pair])[0] for pair in items]
        torch.cuda.synchronize()
        elapsed = monotonic() - started
        answers = [(x.numpy().tobytes(), s) for x, s in outputs]
        ys, ends = zip(*answers, strict=True)
        dest = folder / ("ilm_" + lang + ".npz")
        store_streams(dest, ys, ends)
        baseline = WORK / "evaluation/bilingual" / ("book_" + lang + ".npz")
        assert (
            sha(baseline)
            == json.loads((PUBLIC / "result.json").read_text())["scores"]["bilingual"][
                "book_" + lang
            ]["sha256"]
        )
        old, stops = load_streams(baseline)
        flags["image_" + lang] = [
            c.grade(y, s)[2] for c, (y, s) in zip(cases, answers, strict=True)
        ]
        report["image_" + lang] = dict(
            **tally(flags["image_" + lang]),
            seconds=elapsed,
            prepare_seconds=prep,
            sha256=sha(dest),
            different=sum(
                v != (a, b)
                for v, a, b in zip(answers, old[:128], stops[:128], strict=True)
            ),
        )
    del model
    torch.cuda.empty_cache()
    ledger = folder / "qwen.jsonl"
    answers = (
        [json.loads(s) for s in ledger.read_text().splitlines()]
        if ledger.exists()
        else []
    )
    assert all((a["id"], a["lang"]) == r[:2] for a, r in zip(answers, rows))
    with local_teacher(MODEL, folder) as (ask, digest, size), ledger.open("a") as f:
        assert digest == DIGEST

        def question(p, q):
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

        question("小河流入大海。", "小河流入哪裡？")
        for id_, lang, p, q, refs in rows[len(answers) :]:
            started = monotonic()
            a = question(p, q)
            a.update(id=id_, lang=lang, seconds=monotonic() - started)
            f.write(json.dumps(a, ensure_ascii=False) + "\n")
            f.flush()
            answers.append(a)
    for lang in ("zh", "en"):
        items = [(a, r) for a, r in zip(answers, rows, strict=True) if r[1] == lang]
        flags["qwen_" + lang] = [
            a["stop"] == "stop" and a["text"] in r[-1] for a, r in items
        ]
        report["qwen_" + lang] = dict(
            **tally(flags["qwen_" + lang]),
            seconds=sum(a["seconds"] for a, _ in items),
        )
    for model_name in ("image", "qwen"):
        same = [
            a and b
            for a, b in zip(
                flags[model_name + "_zh"], flags[model_name + "_en"], strict=True
            )
        ]
        report[model_name + "_four"] = {
            "complete": tally(same)["both"],
            "boards": len(same) // 2,
        }
    report.update(
        text_model_bytes=size,
        manifest=digest,
        ledger=sha(ledger),
        gpu_layers=re.findall(
            r"offloaded (\d+/\d+) layers to GPU", (folder / "server.log").read_text()
        ),
    )
    record(PUBLIC / "serving.json", report)


if __name__ == "__main__":
    with gpu_owner():
        run()
