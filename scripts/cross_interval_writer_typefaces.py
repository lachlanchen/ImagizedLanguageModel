"""Intervene separately on the page and request typefaces of the frozen writer."""

import json
from functools import lru_cache

from ilm.visual_lm.prepared_glyph_responder import PreparedGlyphResponder
from scripts.evaluate_page_evidence_writer import tensor
from scripts.evaluate_pointing_causal_writer import load_streams, store_streams
from scripts.fit_font_invariant_reader import Face, FONTS, KAI
from scripts.interval_teacher_corpus import WORK, PUBLIC
from scripts.train_causal_glyph_writer import record, sha, in_cells
from scripts.train_span_choice import gpu_owner


def run():
    faces = {"regular": Face((FONTS[0], 2)), "kai": Face(KAI)}
    draw = {k: lru_cache(maxsize=2048)(f) for k, f in faces.items()}
    cohort = json.loads((PUBLIC / "typefaces.json").read_text())["kai"]
    ids = WORK / "other_faces/kai_ids.json"
    rows = WORK / "pairs_dev.jsonl"
    assert sha(ids) == cohort["ids"]
    assert sha(rows) == json.loads((PUBLIC / "corpus.json").read_text())["fresh"]["sha256"]
    source = list(map(json.loads, rows.read_text().splitlines()))
    by_id = {v["id"]: i for i, v in enumerate(source)}
    keep = [by_id[k] for k in json.loads(ids.read_text())]
    cases = [source[i] for i in keep]
    assert len(cases) == 192 and len(set(keep)) == len(keep)
    assert all(a["board"] == b["board"] for a,b in zip(cases[::2],cases[1::2],strict=True))
    root = WORK / "distilled"
    for n,h in json.loads((PUBLIC / "distilled_fit.json").read_text())["weights"].items():
        assert sha(root / "page" / n) == h
    net = PreparedGlyphResponder(root / "page", "cuda")
    dest = WORK / "crossed_faces"
    dest.mkdir(exist_ok=False)

    def score(outputs):
        correct = []
        for v, (y, stop) in zip(cases, outputs, strict=True):
            cells = in_cells(y)
            correct.append(bool(stop) and any(
                len(ref) == len(cells) and all(
                    ink in (faces["regular"].cell(c), faces["kai"].cell(c))
                    for c, ink in zip(ref,cells)
                ) for ref in v["refs"]
            ))
        return correct

    result = {}
    regular = []
    bench = json.loads((PUBLIC / "result.json").read_text())["arms"]["distilled"]
    for page, question in [("regular","regular"),("kai","kai"),("regular","kai"),("kai","regular")]:
        key = page + "_page_" + question + "_question"
        if page == question:
            if page == "kai":
                path = WORK / "other_faces/kai_distilled.npz"
                expected = cohort["scores"]["distilled"]["streams"]
            else:
                path = WORK / "evaluation/distilled/fresh.npz"
                expected = bench["scores"]["fresh"]["sha256"]
            assert sha(path) == expected
            images, ends = load_streams(path)
            outputs = list(zip(images,ends,strict=True))
            if page == "regular":
                outputs = [outputs[i] for i in keep]
        else:
            inputs = [(tensor(draw[page](v["p"])),tensor(draw[question](v["q"]))) for v in cases]
            outputs = []
            for start in range(0,len(inputs),16):
                outputs.extend((im.numpy().tobytes(), bool(stop)) for im,stop in net.answer_many(inputs[start:start+16]))
            path = dest / (key + ".npz")
            store_streams(path, [y for y,_ in outputs], [s for _,s in outputs])
        correct = score(outputs)
        if not regular:
            regular = correct
        result[key] = dict(correct=sum(correct), both=sum(a and b for a,b in zip(correct[::2],correct[1::2],strict=True)),
                           recovered_regular=sum(a and b for a,b in zip(correct,regular,strict=True)),
                           correct_indices=[i for i,g in zip(keep,correct,strict=True) if g], streams=sha(path))
    record(PUBLIC / "crossed_faces.json", dict(questions=len(cases), passages=len(cases)//2, ids=sha(ids), scores=result))
    print({k: (v["correct"],v["both"],v["recovered_regular"]) for k,v in result.items()})


if __name__ == "__main__":
    with gpu_owner():
        run()
