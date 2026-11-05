#!/usr/bin/env python3
"""Export verified paired-learning scores and unchanged raw raster comparisons."""

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from scripts.cache_semantic_images_v72 import ROOT, sha, atomic_json


def main():
    run = ROOT/"artifacts/clean_learning_v74_20261106"
    verification = json.loads((run/"verification.json").read_text())
    report = json.loads((run/"audit/report.json").read_text())
    training = json.loads((run/"training/report.json").read_text())
    plan = json.loads((run/"plan.json").read_text())
    if (verification["report_sha256"] != sha(run/"audit/report.json")
            or not verification["paired_learning_control_pass"]
            or sha(run/"training/checkpoint_latest.pt") != training["checkpoint_sha256"]):
        raise ValueError("completed independently verified learning control required")
    for name, digest in report["files_sha256"].items():
        if sha(run/"audit"/name) != digest:
            raise ValueError("raw evidence changed")
    out = ROOT/"references/evidence"
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    page = Image.new("RGB", (896, 64+32*44), "white")
    draw = ImageDraw.Draw(page)
    draw.text((16, 10), "Case       Original answer image", font=font, fill="black")
    draw.text((490, 10), "Generated image (seen training questions)", font=font, fill="black")
    for i in range(32):
        folder = run/"audit/predictions"/f"{i:02}"
        y = 48+i*44
        draw.text((16, y+6), f"{i:02}", font=font, fill="black")
        for name, x in (("reference.png", 80), ("answer.png", 490)):
            image = Image.open(folder/name).convert("RGB")
            if image.height != 32 or image.width > 384:
                raise ValueError("retain full physical answer width without resizing")
            page.paste(image, (x, y))
    image_path = out/"clean_learning_v74_raw.png"
    page.save(image_path)
    receipt = dict(source_run=str(run.relative_to(ROOT)), source_commit=plan["source_commit"],
        sources=plan["sources"], plan_sha256=sha(run/"plan.json"),
        training_report_sha256=sha(run/"training/report.json"),
        checkpoint_sha256=training["checkpoint_sha256"], report_sha256=sha(run/"audit/report.json"),
        verification_sha256=sha(run/"verification.json"),
        raw_image_plate_sha256=sha(image_path), training={k:training[k] for k in (
            "updates", "draws", "complete_selected_set_passes", "trainable_parameters",
            "elapsed_seconds", "peak_allocated_vram_gib")},
        inference={k:report[k] for k in ("actual_image_input_calls", "elapsed_seconds",
            "setup_seconds", "peak_allocated_vram_gib")},
        verification=verification, cases=plan["cases"],
        original_exact_count=sum(r["original"]["exact"] for r in report["cases"]),
        supplied_exact_count=sum(r["supplied"]["exact"] for r in report["cases"]),
        scope="32 seen training questions in 16 visible-context pairs; not unseen generalization",
        source_pixels_resized_or_repaired=False, external_pretraining=True,
        language_quality_claim=False, general_ilm_goal_complete=False)
    atomic_json(out/"clean_learning_v74_result.json", receipt)
    print(json.dumps(dict(original_exact=receipt["original_exact_count"],
        supplied_exact=receipt["supplied_exact_count"], receipt_sha256=sha(out/"clean_learning_v74_result.json"))))


if __name__ == "__main__":
    main()
