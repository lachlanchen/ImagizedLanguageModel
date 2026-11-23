"""Apply identical held Chinese requests in two other physical typefaces."""

from collections import defaultdict
import json
from functools import lru_cache
import torch
from ilm.visual_lm.prepared_glyph_responder import PreparedGlyphResponder
from scripts.evaluate_page_evidence_writer import tensor
from scripts.evaluate_pointing_causal_writer import store_streams
from scripts.fit_font_invariant_reader import Face, FONTS, KAI
from scripts.interval_teacher_corpus import WORK, PUBLIC, PARENT
from scripts.train_causal_glyph_writer import record, sha, in_cells
from scripts.train_span_choice import gpu_owner


def test_faces():
    raw = WORK / "pairs_dev.jsonl"
    assert sha(raw) == json.loads((WORK / "corpus.json").read_text())["fresh"]["sha256"]
    rows = list(map(json.loads, raw.read_text().splitlines()))
    dest = WORK / "other_faces"
    dest.mkdir(exist_ok=False)
    canonical = Face((FONTS[0], 2))
    result = {}
    for key, desc in (("bold", (FONTS[2], 2)), ("kai", KAI)):
        font = Face(desc)
        draw = lru_cache(maxsize=2048)(font)
        inputs = []
        cases = []
        for a, b in zip(rows[::2], rows[1::2], strict=True):
            pair = (a, b)
            if any(
                not c.isspace() and ord(c) not in font.coverage
                for v in pair
                for s in (v["p"], v["q"], *v["refs"])
                for c in s
            ):
                continue
            inputs.extend((tensor(draw(v["p"])), tensor(draw(v["q"]))) for v in pair)
            cases.extend(pair)
        if not inputs:
            raise ValueError(f"No complete pairs in {key}")
        record(dest / (key + "_ids.json"), [v["id"] for v in cases])
        stats = {}
        for arm in ("parent", "replay", "aligned", "distilled"):
            root = PARENT if arm == "parent" else WORK / arm
            assert all(
                sha(root / "page" / n) == h
                for n, h in json.loads((root / "page_fit.json").read_text())[
                    "weights"
                ].items()
            )
            model = PreparedGlyphResponder(root / "page", "cuda")
            outputs = []
            for i in range(0, len(inputs), 16):
                outputs.extend(
                    (im.numpy().tobytes(), bool(s))
                    for im, s in model.answer_many(inputs[i : i + 16])
                )
            same = []
            registered = []
            flexible = []
            for v, (y, end) in zip(cases, outputs, strict=True):
                same.append(end and any(y == draw(r) for r in v["refs"]))
                registered.append(end and any(y == canonical(r) for r in v["refs"]))
                bits = in_cells(y)
                flexible.append(
                    end
                    and any(
                        len(r) == len(bits)
                        and all(
                            tile in (font.cell(c), canonical.cell(c))
                            for c, tile in zip(r, bits)
                        )
                        for r in v["refs"]
                    )
                )
            path = dest / (key + "_" + arm + ".npz")
            store_streams(path, [y for y, _ in outputs], [end for _, end in outputs])
            stats[arm] = dict(
                n=len(inputs),
                same_font=sum(same),
                canonical=sum(registered),
                either_per_cell=sum(flexible),
                both=sum(
                    a and b for a, b in zip(flexible[::2], flexible[1::2], strict=True)
                ),
                boards=len(inputs) // 2,
                streams=sha(path),
            )
            del model
        result[key] = dict(
            font=desc[0].name,
            index=desc[1],
            font_sha256=sha(desc[0]),
            ids=sha(dest / (key + "_ids.json")),
            scores=stats,
        )
        print(key, stats, flush=True)
    record(PUBLIC / "typefaces.json", result)


if __name__ == "__main__":
    with gpu_owner():
        test_faces()
