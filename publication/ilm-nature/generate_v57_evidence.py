#!/usr/bin/env python3
"""Make the grounded-reading plate only from the completed locked audit."""

from __future__ import annotations

import hashlib
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from generate_evidence import HERE, ROOT, save

RUN = ROOT / "artifacts/visual_copy_language_v57_20261105"
ROUTES = ["copy", "copy_blank_question", "no_copy"]


def pad_answer(images, cells=17):
    result = np.zeros((32, cells * 32), dtype=np.float32)
    for i, image in enumerate(images[:cells]):
        result[:, i * 32 : (i + 1) * 32] = image[0].numpy()
    return result


def main():
    report = json.loads((RUN / "development/report.json").read_text())
    training = {
        arm: json.loads((RUN / arm / "training_report.json").read_text())
        for arm in ("copy", "no_copy")
    }
    if report["smoke_only"] or len(report["primary_ids"]) != 256:
        raise ValueError("a completed production audit is required")
    if any(r["update"] != 4000 for r in training.values()):
        raise ValueError("incomplete training")
    m = report["measurements"]
    c = report["comparisons"]["copy_minus_copy_blank_question_raster_f1"]
    values = {
        "QaExact": 100 * m["copy"]["exact_answer"],
        "QaFone": 100 * m["copy"]["raster_f1"],
        "QaBlankExact": 100 * m["copy_blank_question"]["exact_answer"],
        "QaBlankFone": 100 * m["copy_blank_question"]["raster_f1"],
        "QaControlExact": 100 * m["no_copy"]["exact_answer"],
        "QaControlFone": 100 * m["no_copy"]["raster_f1"],
        "QaQuestionGain": 100 * c["mean"],
        "QaQuestionLow": 100 * c["ci95"][0],
        "QaQuestionHigh": 100 * c["ci95"][1],
        "QaPairExact": 100 * report["paired_questions"]["both_exact_fraction"],
        "QaMotor": 100 * report["motor"]["exact_fraction"],
        "QaSeconds": training["copy"]["elapsed_seconds"],
        "QaMemory": training["copy"]["peak_allocated_vram_gib"],
        "QaControlSeconds": training["no_copy"]["elapsed_seconds"],
        "QaControlMemory": training["no_copy"]["peak_allocated_vram_gib"],
    }
    tex = {k: f"{v:.3f}" for k, v in values.items()}
    tex["QaDecision"] = report["decision"].replace("_", r"\_")
    (HERE / "qa_values.tex").write_text(
        "% Generated from the completed V57 audit.\n"
        + "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in tex.items())
    )
    sources = [RUN / "development/report.json"] + [
        RUN / arm / "training_report.json" for arm in training
    ]
    for source, name in [(sources[0], "v57_development_report.json")] + [
        (RUN / arm / "training_report.json", f"v57_{arm}_training_report.json")
        for arm in training
    ]:
        (HERE / "evidence" / name).write_text(
            source.read_text().replace(str(ROOT) + "/", "")
        )
    records = {
        r["id"]: r
        for r in [
            json.loads(s)
            for s in (ROOT / "data/visual_qa/cmrc2018_short/development.jsonl")
            .read_text()
            .splitlines()
        ]
    }
    # Imports stay inside main so report inspection does not need host rendering.
    import sys

    sys.path.insert(0, str(ROOT))
    from ilm.visual_lm.visual_qa_data import VisualQARenderer

    renderer = VisualQARenderer()
    ids = report["primary_ids"][:4]
    raw = {
        route: [
            torch.load(
                RUN / "development/predictions" / route / (id + ".pt"),
                map_location="cpu",
                weights_only=False,
            )["images"]
            for id in ids
        ]
        for route in ROUTES
    }
    evidence = {
        "ids": ids,
        "questions": [renderer(records[id]["question"]) for id in ids],
        "references": [renderer(records[id]["references"][0]["text"]) for id in ids],
        **raw,
    }
    torch.save(evidence, RUN / "development/publication_first_four.pt")
    sources.append(RUN / "development/publication_first_four.pt")
    (HERE / "evidence/v57_source_hashes.json").write_text(
        json.dumps(
            {
                str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sources
            },
            indent=2,
        )
        + "\n"
    )
    plt.rcParams.update(
        {
            "font.size": 8,
            "font.family": "DejaVu Sans",
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig = plt.figure(figsize=(7.25, 6.6), layout="constrained")
    grid = fig.add_gridspec(3, 1, height_ratios=(1.2, 1, 1))
    ax = fig.add_subplot(grid[0])
    x = np.arange(3)
    ax.bar(
        x - 0.17,
        [100 * m[r]["exact_answer"] for r in ROUTES],
        0.32,
        label="Exact stopped answer",
        color="#28638b",
    )
    ax.bar(
        x + 0.17,
        [100 * m[r]["raster_f1"] for r in ROUTES],
        0.32,
        label="Multiset raster F1",
        color="#ae593f",
    )
    ax.set(
        xticks=x,
        xticklabels=[
            "Copy / full question",
            "Copy / blank question",
            "Matched no-copy",
        ],
        ylabel="Score (%)",
        ylim=(0, 100),
    )
    ax.legend(frameon=False, fontsize=7)
    ax.set_title(
        "a  Natural questions on unseen Chinese paragraphs",
        loc="left",
        weight="bold",
        fontsize=9,
    )
    for index in range(2):
        sub = grid[index + 1].subgridspec(2, 4, height_ratios=(0.7, 1))
        q = fig.add_subplot(sub[0, :])
        q.axis("off")
        question = evidence["questions"][index]
        q.imshow(
            np.concatenate([im[0].numpy() for im in question], axis=1),
            cmap="gray_r",
            vmin=0,
            vmax=1,
            interpolation="nearest",
        )
        q.set_title(
            f"{chr(98+index)}  Input question: audit row {index+1}",
            loc="left",
            weight="bold",
            fontsize=8,
        )
        for col, (label, images) in enumerate(
            [("Reference 1", evidence["references"][index])]
            + [(r.replace("_", " "), evidence[r][index]) for r in ROUTES]
        ):
            panel = fig.add_subplot(sub[1, col])
            panel.axis("off")
            panel.imshow(
                pad_answer(images),
                cmap="gray_r",
                vmin=0,
                vmax=1,
                interpolation="nearest",
            )
            panel.set_title(label, fontsize=7)
    save(fig, "Figure_4_visual_reading")
    print(json.dumps(tex, indent=2))


if __name__ == "__main__":
    main()
