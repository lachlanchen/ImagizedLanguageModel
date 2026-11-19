"""Score both named participants in a frozen pictured answer, using oracle name boundaries."""

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np
import torch
from safetensors.torch import load_file

from scripts.explicit_participant_cases import cases

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "artifacts/shared_sentence_writer_v1"
EVIDENCE = ROOT / "references/evidence/shared_sentence_writer_20261119"


def bound(path, digest):
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError(f"Unexpected bytes in {path.name}")


@torch.inference_mode()
def inspect(bundle):
    from ilm.visual_lm.interval_writer import IntervalWriter
    from ilm.visual_lm.portable_writer import (
        DEFAULT_FONT, GroundedGlyphDecoder, ScopedQuestion, descriptors, read_text,
    )

    state = json.loads((bundle / "writer.json").read_text())
    for name, digest in state["sha256"].items():
        bound(bundle / name, digest)
    model = GroundedGlyphDecoder().eval()
    model.question = ScopedQuestion(ordered=True).eval()
    model.load_state_dict(load_file(str(bundle / "final.safetensors")))
    locator = IntervalWriter().eval()
    locator.load_state_dict(load_file(str(bundle / "interval.safetensors")))
    rows = list(cases())
    alphabet = sorted({char for row in rows for text in row for char in text})
    font_cells = dict(zip(alphabet, map(bytes, read_text("".join(alphabet), DEFAULT_FONT)), strict=True))
    memory = list(map(bytes, np.load(bundle / "memory.npy")))
    tiles = [bytes(128), *memory]
    ids = {cell: i for i, cell in enumerate(tiles[1:], 1)}
    for cell in font_cells.values():
        if cell not in ids:
            ids[cell] = len(tiles)
            tiles.append(cell)
    bits = np.unpackbits(np.frombuffer(b"".join(tiles), np.uint8).reshape(-1, 128), axis=1)
    ink = model.ink(descriptors(torch.from_numpy(bits)))
    mem = torch.arange(1, len(memory) + 1)
    keys = model.write_key(ink[mem])

    score = json.loads((EVIDENCE / "calibrated.json").read_text())["scores"]["explicit"]
    archive = WORK / "calibrated/explicit.npz"
    bound(archive, score["archive"])
    with np.load(archive) as data:
        offsets, pixels, stops = data["offsets"], data["pixels"], data["stopped"]
    assert len(offsets) == len(rows) + 1 and stops.all()
    saved = [pixels[i:j].tobytes() for i, j in zip(offsets[:-1], offsets[1:], strict=True)]
    categories = Counter()
    preferences = []
    actual_reads = 0
    details = []
    for i, (scene, query, target) in enumerate(rows):
        other_scene, _, other = rows[i ^ 1]
        assert scene == other_scene and target != other
        p, q = (torch.tensor([[ids[font_cells[c]] for c in text]]) for text in (scene, query))
        action, scores = locator(model.encode(ink, p, q), p)
        actual_reads += int(action.item() >= 0)
        generated, ended = locator.generate(model, ink, p, q, keys, mem, limit=state["limit"])
        actual = b"".join(tiles[k] for k in generated[0].tolist() if k)
        assert actual == saved[i] and ended.item() == stops[i], i
        name_scores = []
        name_images = []
        for name in (target, other):
            positions = [j for j in range(len(scene)) if scene.startswith(name, j)]
            assert positions
            name_scores.append(scores[0, positions, len(name) - 1].max().item())
            name_images.append(b"".join(font_cells[c] for c in name))
        category = "requested_person" if actual == name_images[0] else (
            "other_person" if actual == name_images[1] else "neither_name"
        )
        categories[category] += 1
        margin = name_scores[0] - name_scores[1]
        preferences.append(margin)
        details.append(dict(row=i, free_answer=category, copied=action.item() >= 0,
                            requested_minus_other_logit=margin))
    assert actual_reads == score["read"] and categories["requested_person"] == score["exact"]
    margins = np.array(preferences).reshape(-1, 2, 2)
    return dict(
        saved_answers_sha256=score["archive"], reproduced=len(saved), free_answers=dict(categories),
        copy_actions=actual_reads,
        identical_image_for_opposite_questions=sum(saved[i] == saved[i + 1] for i in range(0, len(saved), 2)),
        same_sentence_question_pairs=len(saved) // 2,
        correct_given_both_name_boundaries=int((margins > 0).sum()),
        exact_logit_ties=int((margins == 0).sum()),
        all_four_given_both_name_boundaries=int((margins > 0).all(axis=(1, 2)).sum()),
        details=details,
    )


def main():
    torch.set_num_threads(2)
    archive = WORK / "shared-sentence-writer.zip"
    digest = json.loads((EVIDENCE / "portable.json").read_text())["bundle_sha256"]
    bound(archive, digest)
    with TemporaryDirectory() as tmp:
        with ZipFile(archive) as package:
            package.extractall(tmp)
        sys.path.insert(0, tmp)
        result = inspect(Path(tmp))
    result["bundle_sha256"] = digest
    path = EVIDENCE / "explicit_error_diagnosis.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print({k: v for k, v in result.items() if k != "details"})


if __name__ == "__main__":
    main()
