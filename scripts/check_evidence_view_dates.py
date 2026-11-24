"""Cross two questions and dates on bilingual library notices through the CPU PNG interface."""

from hashlib import sha256
import argparse
from pathlib import Path
import re

import numpy as np
import torch
from ilm.visual_lm.portable_writer import DEFAULT_FONT, read_image, read_text
from scripts.evaluate_visual_discourse import draw
from scripts.prepare_evidence_views import MODEL, PUB
from scripts.prepare_open_exam import dump
from scripts.write_open_exam import Writer, infer, load

NOTICES = {
    "en": (
        "The library was founded in {}. The reading room opened in {}.",
        "When was the library founded?",
        "When did the reading room open?",
    ),
    "zh": (
        "图书馆建于{}年，阅览室于{}年开放。",
        "图书馆建于哪一年？",
        "阅览室于哪一年开放？",
    ),
}


def main(root, report):
    writer = Writer()
    models = {arm: load(arm, "cpu", root) for arm in ("parent", "whole", "highlight")}
    output = []
    for lang, (template, *questions) in NOTICES.items():
        for version, dates in enumerate((("1891", "1903"), ("1903", "1891"))):
            passage = template.format(*dates)
            for j, question in enumerate(questions):
                d = root / "examples" / "library" / lang / str(version) / str(j)
                d.mkdir(exist_ok=True, parents=True)
                glyphs = {
                    bytes(t): c for t, c in zip(writer.palette, writer.transcript)
                }
                inputs = []
                for name, text in (("question", question), ("passage", passage)):
                    cells = read_text(text, DEFAULT_FONT)
                    draw(d, name, cells)
                    inputs.append(read_image(d / (name + ".png"), len(cells)))
                    assert np.array_equal(cells, inputs[-1])
                    for t, c in zip(cells, text, strict=True):
                        glyphs.setdefault(bytes(t), c)
                for arm, model in models.items():
                    pix, stop = infer(writer, model, *inputs)
                    answer = "".join(glyphs.get(bytes(t), "�") for t in pix)
                    draw(d, arm, pix)
                    output.append(
                        dict(
                            language=lang,
                            version=version,
                            question=question,
                            passage=passage,
                            arm=arm,
                            expected=dates[j],
                            text=answer,
                            stopped=stop,
                            exact_year_only=stop and answer == dates[j],
                            sole_year=stop
                            and re.findall(r"(?<![0-9])[0-9]{4}(?![0-9])", answer)
                            == [dates[j]],
                            sha256=sha256(pix.tobytes()).hexdigest(),
                        )
                    )
    dump(report / "library.json", output)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", type=Path, default=MODEL)
    p.add_argument("--report", type=Path, default=PUB)
    a = p.parse_args()
    torch.set_num_threads(3)
    with torch.inference_mode():
        main(a.model_dir, a.report)
