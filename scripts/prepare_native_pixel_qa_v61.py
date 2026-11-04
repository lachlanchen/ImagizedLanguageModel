#!/usr/bin/env python3
"""Losslessly cache question/answer image bits for native pixel training."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from ilm.visual_lm.native_pixel_instruction import training_strip
from ilm.visual_lm.visual_qa_data import VisualQARenderer, read_questions
from scripts.train_canonical_glyph_binding_v43 import _atomic_json, _atomic_save
from scripts.train_visual_answer_blocks_v58 import DATA_HASHES


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="data/visual_qa/cmrc2018_short")
    p.add_argument("--out", default="artifacts/native_pixel_instruction_v61_20261105/data")
    a = p.parse_args()
    data, out = Path(a.data), Path(a.out)
    if (out / "report.json").exists():
        raise FileExistsError("completed image cache exists")
    out.mkdir(parents=True, exist_ok=True)
    for name, digest in DATA_HASHES.items():
        if sha256(data/name) != digest:
            raise ValueError("prepared CMRC data changed")
    sources = {name: sha256(name) for name in [
        "scripts/prepare_native_pixel_qa_v61.py", "ilm/visual_lm/native_pixel_instruction.py",
        "ilm/visual_lm/visual_qa_data.py", "ilm/visual_lm/visual_cell_data.py",
        "ilm/visual_lm/visual_cell_stream.py", "ilm/visual_lm/canonical_glyph_language_data.py",
        "references/native_pixel_instruction_v61_protocol.md",
    ]}
    identity = dict(data_sha256=DATA_HASHES, sources=sources)
    identity_path = out/"identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError("partial image cache differs")
    _atomic_json(identity, identity_path)
    renderer = VisualQARenderer()
    begun = time.perf_counter()
    files, counts = {}, {}
    for split in ("train", "development"):
        rows = read_questions(data/(split+".jsonl"))
        folder = out/split; folder.mkdir(exist_ok=True)
        for i, record in enumerate(rows):
            path = folder/(record["id"]+".pt")
            if not path.exists():
                pixels, boundary = training_strip(renderer(record["prompt"]),
                    renderer(record["references"][0]["text"]))
                # Only bit-packed measurements and spatial bounds reach training.
                packed = np.packbits((pixels.numpy().reshape(-1) > 0).astype(np.uint8))
                saved = dict(bits=torch.from_numpy(packed), width=pixels.shape[-1],
                             prompt_blocks=boundary, prompt_cells=len(record["prompt"]),
                             answer_cells=len(record["references"][0]["text"]), id=record["id"])
                _atomic_save(saved, path)
            files[str(path.relative_to(out))] = sha256(path)
            if i % 500 == 0:
                print(json.dumps(dict(split=split, completed=i+1)), flush=True)
        counts[split] = len(rows)
    _atomic_json(dict(**identity, counts=counts, files_sha256=files,
                      elapsed_seconds=time.perf_counter()-begun, codec="numpy.packbits big-endian; native white=1",
                      model_input="binary pixels and prompt spatial boundary only"), out/"report.json")
    print(json.dumps(counts), flush=True)


if __name__ == "__main__":
    main()
