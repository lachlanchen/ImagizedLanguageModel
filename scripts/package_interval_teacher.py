"""Release the predesignated distilled writer and cold-test its real image replies."""

from pathlib import Path
import json
from ilm.visual_lm.causal_glyph_writer import write_strip
from scripts.evaluate_interval_teacher import trials
from scripts.evaluate_page_evidence_writer import tensor
from scripts.evaluate_pointing_causal_writer import load_streams
from scripts.interval_teacher_corpus import WORK, PUBLIC, ROOT
from scripts.package_mixed_visual_answers import package
from scripts.train_causal_glyph_writer import sha


def export():
    r = json.loads((PUBLIC / "result.json").read_text())
    case = r["examples"].get("gain") or r["examples"]["error"]
    g = trials()
    ex = PUBLIC / "examples"
    ex.mkdir(exist_ok=True)
    for name, indices, labels in [
        ("fresh", [case["index"]], ["human"]),
        ("yan", [2, 3], ["yan_oracle", "yan_bronze"]),
    ]:
        f = WORK / "evaluation/distilled" / (name + ".npz")
        assert sha(f) == r["arms"]["distilled"]["scores"][name]["sha256"]
        ys, _ = load_streams(f)
        for idx, label in zip(indices, labels, strict=True):
            v = g[name][idx]
            for kind, ink in (
                ("page", v.p),
                ("question", v.q),
                ("reference", v.refs[0]),
                ("answer" if name == "fresh" else "output", ys[idx]),
            ):
                write_strip(tensor(ink).numpy(), ex / (label + "_" + kind + ".png"))
    package(
        WORK / "distilled",
        PUBLIC,
        ROOT / "models/interval_teacher/README.md",
        "interval-distilled-writer.zip",
    )


if __name__ == "__main__":
    export()
