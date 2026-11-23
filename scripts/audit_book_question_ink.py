"""Count canonical non-whitespace question cells present in the fitted glyph bank."""

from collections import Counter
import json

from safetensors.numpy import load_file
from scripts.train_bilingual_novel_writer import WORK, PUBLIC, opposite
from scripts.train_typeface_writer import Face, FACES
from scripts.train_causal_glyph_writer import record, sha


def audit():
    face = Face(FACES[0])
    interface = WORK / "bilingual/page/interface.safetensors"
    assert (
        sha(interface)
        == json.loads((PUBLIC / "bilingual_fit.json").read_text())["weights"][
            interface.name
        ]
    )
    bank = set(map(bytes, load_file(str(interface))["packed"]))
    counts = {"en": Counter(), "zh": Counter()}
    for line in (WORK / "valid.jsonl").read_text().splitlines():
        qs = json.loads(line)["questions"]
        pair = next(
            (a, b) for j, a in enumerate(qs) for b in qs[j + 1 :] if opposite(a, b)
        )
        for item in pair:
            for lang, group in counts.items():
                hits = [face.cell(c) in bank for c in item[lang] if not c.isspace()]
                group.update(
                    n=len(hits), exact=sum(hits), questions=1, complete=int(all(hits))
                )
    record(PUBLIC / "question_inventory.json", counts)


if __name__ == "__main__":
    audit()
