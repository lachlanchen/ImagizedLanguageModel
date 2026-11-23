"""Measure the effect of independently rewriting the same pictured facts and questions."""

from copy import deepcopy
import json
import torch
from ilm.visual_lm.prepared_glyph_responder import PreparedGlyphResponder
from scripts.evaluate_page_evidence_writer import tensor
from scripts.evaluate_pointing_causal_writer import store_streams
from scripts.prepare_role_change_actions import PUBLIC, WORK
from scripts.role_change_corpus import fictional, load
from scripts.train_causal_glyph_writer import record, sha
from scripts.train_role_change_writer import make_rows, report_group
from scripts.train_span_choice import gpu_owner
from scripts.train_typeface_writer import FACES, Face

GLOSSES = [
    ("sealed the letter", "wrote the letter from dictation"),
    ("numbered the photos", "put the photos in the album"),
    ("reported the depth", "steered the boat"),
    ("projected the captions", "voiced the puppet"),
    ("watched the lamp", "listened for the bell"),
    ("played the boatman", "played the passenger"),
    ("guarded the east gate", "guarded the west gate"),
    ("bought groceries", "cooked the meal"),
    ("drew the map", "checked the scale"),
    ("repaired the book corners", "traced the inscription"),
    ("measured the bridge width", "measured the river depth"),
    ("organized the scores", "copied the lyrics"),
]


@torch.inference_mode()
def run():
    originals = load()["fresh"]
    face = Face(FACES[0])
    canonical = [
        fictional(
            c["key"], list(zip(c["actions"], eng, strict=True)), c["names"], c["kind"]
        )
        for c, eng in zip(originals, GLOSSES, strict=True)
    ]
    free = json.loads((PUBLIC / "scores.json").read_text())
    results = {}
    for arm in ("aligned", "independent"):
        writer = PreparedGlyphResponder(WORK / arm / "page", "cuda")
        report = {"original": free[arm]["fresh"]}
        for form in ("page", "questions", "both"):
            scenes = deepcopy(originals)
            for c, template in zip(scenes, canonical, strict=True):
                if form in ("page", "both"):
                    c["pages"] = template["pages"]
                if form in ("questions", "both"):
                    c["qs"] = template["qs"]
            rows = make_rows({"fresh": scenes}, face)["fresh"]
            ys = []
            ends = []
            for i in range(0, len(rows), 24):
                jobs = [(tensor(p), tensor(q)) for *_, p, q in rows[i : i + 24]]
                for ink, stopped in writer.answer_many(jobs):
                    ys.append(ink.numpy().tobytes())
                    ends.append(bool(stopped))
            scores, _ = report_group(rows, ys, ends, face)
            path = WORK / f"rewrite_{arm}_{form}.npz"
            store_streams(path, ys, ends)
            report[form] = scores | dict(sha256=sha(path))
        del writer
        results[arm] = report
    record(PUBLIC / "wording_probe.json", results)


if __name__ == "__main__":
    with gpu_owner():
        torch.set_num_threads(2)
        run()
