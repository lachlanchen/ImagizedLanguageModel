#!/usr/bin/env python3
"""Correct one native slide's completed-audit status; preserve all other parts."""

import io
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET
import zipfile

from generate_semantic_transfer_figures import HERE, ROOT, sha

BASELINE = "29ace23"


def main():
    prefix = "publication/ilm-nature/"
    raw = subprocess.check_output(["git", "show", BASELINE+":"+prefix+"ILM_Figures_editable.pptx"], cwd=ROOT)
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        parts = {name:archive.read(name) for name in archive.namelist()}
    old_footer = ("38.65M trainable writer parameters. External reader pretraining is retained. "
        "Full language training and audit are pending.")
    new_footer = ("38.65M trainable writer. External reader pretraining retained. "
        "Completed Chinese audit: 0/128 exact; 29.386% foreground F1.")
    changed = []
    for name in ("ppt/slides/slide26.xml", "ppt/notesSlides/semantic_transfer_notes26.xml"):
        text = parts[name].decode()
        if name.endswith("slide26.xml"):
            if text.count(old_footer) != 1:
                raise ValueError("unexpected native footer")
            text = text.replace(old_footer, new_footer)
        else:
            text = text.replace("Implemented V72 candidate, not a completed language result.",
                "Completed V72 velocity candidate; the autonomous language audit failed.")
            text = text.replace("Execution checks pass but full autonomous language quality remains unproven.",
                "The completed 40,000-update audit gives 0/128 exact Chinese answers and 29.386% foreground F1. "
                "Four of six Chinese requirements fail. Source: references/evidence/semantic_ilm_v72_result.json.")
        if text == parts[name].decode():
            raise ValueError("expected one changed status part")
        ET.fromstring(text)
        parts[name] = text.encode()
        changed.append(name)
    target = HERE/"ILM_Figures_editable.pptx"
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, content in parts.items():
            archive.writestr(name, content)
    receipt_path = HERE/"evidence/semantic_transfer_deck_build.json"
    receipt = json.loads(subprocess.check_output(["git", "show", BASELINE+":"+prefix+str(receipt_path.relative_to(HERE))], cwd=ROOT))
    receipt["pptx_sha256"] = sha(target)
    receipt["method_status"] = "completed language audit failed"
    receipt["sources_sha256"]["publication/ilm-nature/generate_semantic_transfer_figures.py"] = sha(HERE/"generate_semantic_transfer_figures.py")
    receipt["sources_sha256"]["publication/ilm-nature/update_completed_flow_figure.py"] = sha(Path(__file__))
    receipt["sources_sha256"]["references/evidence/semantic_ilm_v72_result.json"] = sha(ROOT/"references/evidence/semantic_ilm_v72_result.json")
    receipt["status_update"] = dict(baseline_commit=BASELINE, changed_parts=changed,
        other_parts_preserved=len(parts)-len(changed), prior_native_slides_preserved=25)
    receipt_path.write_text(json.dumps(receipt, indent=2)+"\n")
    print("Updated only slide 26 footer and notes; preserved all other native parts.")


if __name__ == "__main__":
    main()
