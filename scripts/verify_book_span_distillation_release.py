"""Test the released student on the earlier 96 authored minimal-pair questions."""

import json

import torch

from scripts.figure_role_change_writer import png
from scripts.prepare_book_span_distillation import PUBLIC
from scripts.read_book_participant import answer, restore
from scripts.train_causal_glyph_writer import ROOT, record, sha
from scripts.train_individual_clause_writer import new_stories
from scripts.train_typeface_writer import Face, FACES


def verify():
    face = Face(FACES[0])
    weights = ROOT / "models/book_span_distillation/reader.safetensors"
    assert (
        sha(weights) == json.loads((PUBLIC / "matched_fit.json").read_text())["weights"]
    )
    model = restore(weights, "cpu")
    folder = PUBLIC / "example"
    folder.mkdir(exist_ok=True)
    results = []
    example = None
    for story in new_stories():
        for world, page in enumerate(story["pages"]):
            names = story["names"][::-1] if world else story["names"]
            for lang, questions in story["qs"].items():
                for j, question in enumerate(questions):
                    image = answer(model, face(page), face(question))
                    correct = image == face(names[j])
                    results.append(
                        dict(
                            story=story["key"],
                            world=world,
                            lang=lang,
                            question=j,
                            correct=correct,
                        )
                    )
                    if example is None and correct and lang == "zh":
                        png(face(page), folder / "page.png")
                        png(face(question), folder / "question.png")
                        png(image, folder / "reply.png")
                        example = dict(
                            page=page,
                            question=question,
                            answer=names[j],
                            selection="first correct Chinese answer in authored source order",
                        )
    record(
        PUBLIC / "authored_release.json",
        dict(weights=sha(weights), cases=results, example=example),
    )


if __name__ == "__main__":
    torch.set_num_threads(2)
    verify()
