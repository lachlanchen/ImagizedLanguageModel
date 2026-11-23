"""Measure correspondence decisions on unseen print and source-plate inscriptions."""

from collections import Counter
import json
import numpy as np
import torch
from safetensors.torch import load_file
from ilm.visual_lm.typeface_map import TypefaceMap
from scripts.evaluate_page_evidence_writer import tensor
from scripts.fit_font_invariant_reader import Face
from scripts.pointing_glyph_corpus import plates
from scripts.ordered_image_corpus import PLATES
from scripts.train_causal_glyph_writer import record, sha, in_cells
from scripts.train_typeface_writer import WORK, BASE, PREVIOUS, PUBLIC, HELD, FACES
from scripts.train_span_choice import gpu_owner
from scripts.typeface_writer_cases import extra_sources, pictures


@torch.inference_mode()
def audit():
    packed = load_file(str(BASE / "interface.safetensors"))["packed"]
    bank = {bytes(p): i for i, p in enumerate(packed.numpy(), 1)}
    path = WORK / "typeface.safetensors"
    assert (
        sha(path) == json.loads((PUBLIC / "correspondence.json").read_text())["sha256"]
    )
    mapper = TypefaceMap.restore(path, packed).cuda()
    canonical = Face(FACES[0])
    trial, _ = extra_sources()
    fresh = list(
        map(json.loads, (PREVIOUS / "pairs_dev.jsonl").read_text().splitlines())
    )

    def resolve(inks):
        extra = [p for p in inks if p not in bank]
        result = {p: bank[p] for p in inks if p in bank}
        for start in range(0, len(extra), 256):
            batch = extra[start : start + 256]
            indices, accepted = mapper.identify(tensor(b"".join(batch)).cuda())
            result.update(
                (p, int(i) if ok else None)
                for p, i, ok in zip(
                    batch, indices.tolist(), accepted.tolist(), strict=True
                )
            )
        return result

    report = {}
    for key, desc in [("kai", FACES[2]), ("hana", HELD)]:
        f = Face(desc)
        for group, rows in [("development", fresh), ("trial", trial)]:
            keep = {c.question_id for c in pictures(rows, f, canonical)}
            chars = Counter(
                c for v in rows if v["id"] in keep for s in (v["p"], v["q"]) for c in s
            )
            mapping = resolve({f.cell(c) for c in chars})
            counts = Counter()
            for c, n in chars.items():
                predicted = mapping[f.cell(c)]
                target = bank.get(canonical.cell(c))
                kind = (
                    "abstained"
                    if predicted is None
                    else ("correct" if predicted == target else "wrong")
                )
                counts[kind + "_occurrences"] += n
                counts[kind + "_identities"] += 1
            report[group + "_" + key] = dict(counts)
    all_ink = {bytes(p) for p in np.load(PLATES / "pixels.npy")}
    historical = {
        x
        for r in plates("development", canonical)
        for x in in_cells(r.y)
        if x in all_ink
    }
    resolved = resolve(historical)
    report["inscriptions"] = dict(
        unique=len(historical),
        fixed=sum(p in bank for p in historical),
        abstained=sum(v is None for v in resolved.values()),
        mapped=sum(v is not None and p not in bank for p, v in resolved.items()),
    )
    record(PUBLIC / "identification.json", report)


if __name__ == "__main__":
    with gpu_owner():
        audit()
