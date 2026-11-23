"""Replay a deployable CPU reader on novel pages and an illustrated public-domain-style example."""

import json
from safetensors.torch import load_file
import torch
from ilm.visual_lm.book_person_proposals import PersonDetector
from scripts.figure_role_change_writer import png
from scripts.read_book_participant import DEFAULT, answer, restore
from scripts.prepare_book_participant_questions import PUBLIC, WORK
from scripts.train_causal_glyph_writer import record, sha
from scripts.train_typeface_writer import Face, FACES
from scripts.train_individual_clause_writer import new_stories


@torch.inference_mode()
def main():
    face = Face(FACES[0])
    device = "cpu"
    net = restore(DEFAULT, device)
    path = DEFAULT.with_name("person_detector.safetensors")
    assert sha(path) == json.loads((PUBLIC / "person_fit.json").read_text())["weights"]
    assert (
        sha(DEFAULT) == json.loads((PUBLIC / "aligned_fit.json").read_text())["weights"]
    )
    detector = PersonDetector().eval()
    detector.load_state_dict(load_file(str(path)))
    pages = json.loads((WORK / "reserved.json").read_text())
    outputs = json.loads((WORK / "person_proposal_replies.json").read_text())
    if (
        sha(WORK / "person_proposal_replies.json")
        != json.loads((PUBLIC / "person_reserved.json").read_text())["sha256"]
    ):
        raise ValueError("Changed proposal outputs")
    exact = 0
    for i in range(0, len(pages), 7):
        r, case = pages[i], outputs[i]
        p = face(r["page"])
        for j, lang in enumerate(("zh", "en")):
            for k, q in enumerate(r["questions"]):
                result = answer(net, p, face(q[lang]), detector)
                exact += result.hex() == case["replies"][2 * j + k]["images"]["aligned"]
    folder = PUBLIC / "example"
    folder.mkdir(exist_ok=True)
    rows = []
    for i, c in enumerate(new_stories()):
        for w, p in enumerate(c["pages"]):
            names = c["names"][::-1] if w else c["names"]
            for lang, qs in c["qs"].items():
                for h, q in enumerate(qs):
                    raw = answer(net, face(p), face(q), detector)
                    if i == w == 0 and lang == "zh":
                        png(face(p), folder / "page.png")
                        png(face(q), folder / f"question{h}.png")
                        png(raw, folder / f"reply{h}.png")
                    rows.append(
                        dict(
                            scene=c["key"],
                            world=w,
                            lang=lang,
                            action=h,
                            correct=raw == face(names[h]),
                        )
                    )
    record(
        PUBLIC / "release.json",
        dict(
            reader_sha=sha(DEFAULT),
            detector_sha=sha(path),
            cpu_replayed=4 * len(pages[::7]),
            cpu_exact_to_gpu=exact,
            authored=rows,
            example={p.name: sha(p) for p in folder.glob("*.png")},
        ),
    )


if __name__ == "__main__":
    torch.set_num_threads(2)
    main()
