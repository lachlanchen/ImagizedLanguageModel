"""Build genuine out-of-chapter drafts for learning to revise a pictured gloss."""

import argparse
import fcntl
import hashlib
from pathlib import Path
import tempfile
import time

import numpy as np
from rapidfuzz.distance import Levenshtein

from ilm.visual_lm.bitmap_alignment import blocks
from ilm.visual_lm.visual_phrase_composer import VisualPhraseComposer
from scripts.align_visual_translation import ROOT, WORK, load, sha, tiles, write_new
from scripts.train_aligned_visual_translation import read_rows


def prepare(output):
    if output.exists():
        raise FileExistsError(output)
    started = time.monotonic()
    rows = read_rows("train")
    source_bank = tiles()
    bank_hash = hashlib.sha256(source_bank.tobytes()).hexdigest()
    if bank_hash != load(WORK / "preparation.json")["bank_sha256"]:
        raise ValueError("Glyph bank differs from aligned training roster")
    bank, canonical = np.unique(source_bank, axis=0, return_inverse=True)
    folds = [
        hashlib.sha256(("revise1/" + r["group"]).encode()).digest()[0] % 4 for r in rows
    ]
    pairs = [
        (tuple(canonical[r["x"]].tolist()), tuple(canonical[r["z"]].tolist()))
        for r in rows
    ]
    drafts = [None] * len(rows)
    stats = []
    for fold in range(4):
        held = [i for i, k in enumerate(folds) if k == fold]
        sources = {pairs[i][0] for i in held}
        fit = [p for k, p in zip(folds, pairs) if k != fold and p[0] not in sources]
        composer = VisualPhraseComposer.fit(bank, fit, blocks, weight=0.25)
        edits = 0
        for i in held:
            source, target = pairs[i]
            draft = composer.decode(source)
            edits += Levenshtein.distance(draft, target)
            drafts[i] = draft
        stats.append(
            dict(
                fold=fold,
                draft_rows=len(held),
                fit_rows=len(fit),
                duplicate_source_exclusions=len(rows) - len(held) - len(fit),
                draft_edits=edits,
            )
        )
    offsets = np.cumsum([0] + [len(v) for v in drafts], dtype=np.int64)
    output.parent.mkdir(exist_ok=True, parents=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as scratch:
        staged = Path(scratch) / "revision"
        staged.mkdir()
        archive = staged / "drafts.npz"
        np.savez_compressed(
            archive,
            offsets=offsets,
            images=np.concatenate(drafts).astype(np.int32),
            folds=np.asarray(folds, np.int8),
            bank=bank,
        )
        write_new(
            staged / "preparation.json",
            dict(
                rows=len(rows),
                chapters=len({r["group"] for r in rows}),
                source_sha256=sha(WORK / "train.jsonl"),
                glyph_bank_sha256=bank_hash,
                output_sha256=sha(archive),
                folds=stats,
                output_cells=int(offsets[-1]),
                target_cells=sum(len(y) for _, y in pairs),
                seconds=time.monotonic() - started,
            ),
        )
        staged.rename(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "artifacts/pictured_revision_drafts_v1"
    )
    with (ROOT / "artifacts/pictured_revision_preparation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prepare(parser.parse_args().output)
