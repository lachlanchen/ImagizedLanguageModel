"""Freeze free word streams from novel books and reproduce all earlier one-word tasks."""

from zipfile import ZIP_DEFLATED, ZipFile
import numpy as np
import torch

from ilm.visual_lm.causal_book_words import CausalBookWriter
from ilm.visual_lm.passage_selector import PassageWriter, VisualBank
from scripts.evaluate_passage_selector import (
    predict,
    record_ink,
    write_array,
    observations,
    donors,
)
from scripts.score_passage_selector import backoff
from scripts.train_causal_book_words import extended_bank, bindings
from scripts.prepare_causal_book_words import WORK, PUB, MODEL, ROOT, old
from scripts.train_span_choice import gpu_owner
from scripts.verse_gpu_runtime import configure_cuda_cli

ARCHIVE = ROOT / "publication/ilm-nature/evidence/causal_book_streams.zip"
FRESH = ("fiction", "gutenberg")
OLD = ("old_fiction", "old_gutenberg", "old_lambada")


def load_table(split):
    path = (
        old.WORK / (split[4:] + ".npz")
        if split.startswith("old_")
        else WORK / (split + ".npz")
    )
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


class Feedback:
    """Assign a newly emitted bitmap stream a cache address without consulting references."""

    def __init__(self, bank):
        self.bank = bank
        self.lookup = {w: i for i, w in enumerate(bank.words)}

    def addresses(self, words):
        new = list(dict.fromkeys(w for w in words if w not in self.lookup))
        if new:
            b = self.bank
            start = len(b.words)
            extra = VisualBank(b.face, new, b.device)
            b.words.extend(new)
            self.lookup.update((w, i) for i, w in enumerate(new, start))
            for k in ("keys", "lengths", "codes", "idf"):
                setattr(b, k, torch.cat((getattr(b, k), getattr(extra, k))))
        return np.array([self.lookup[w] for w in words], np.int32)


def generate(
    net, feedback, table, archive, prefix, *, long=False, count=None, budget=3
):
    src = table["source"].copy()
    marks = table["marks"].copy()
    tasks = table["task"]
    if long:
        indices = np.array(
            [
                np.flatnonzero((table["group"] == g) & (tasks == 0))[0]
                for g in np.unique(table["group"])
            ]
        )
    else:
        indices = np.arange(len(src))
    write_array(archive, prefix + "/indices", indices)
    src, marks, tasks = src[indices], marks[indices], tasks[indices]
    stop = np.zeros(len(src), np.int16)
    bank = feedback.bank
    for step in range(1, budget + 1):
        active = np.flatnonzero((stop == 0) & ((step == 1) | (tasks == 0)))
        raw = [np.empty((0, 128), np.uint8)] * len(src)
        if len(active):
            if count:
                ids = np.array(
                    [
                        count(row[-3:].tolist(), int(t))
                        for row, t in zip(src[active], tasks[active], strict=True)
                    ]
                )
                seq = [bank.words[k] for k in ids]
                finish = np.zeros(len(ids), bool)
            else:
                answers, finish = predict(
                    net, bank, src[active], marks[active], tasks[active]
                )
                seq = [x[0] for x in answers]
                ids = feedback.addresses(seq)
            for i, w in zip(active, seq, strict=True):
                raw[i] = bank.face.pixels(w)
            src[active] = np.column_stack((src[active, 1:], ids))
            marks[active] = np.column_stack((marks[active, 1:], np.zeros(len(active))))
            stop[active[finish]] = step
        record_ink(archive, prefix + f"/word{step}", raw)
    write_array(archive, prefix + "/stops", stop)


@torch.inference_mode()
def main():
    configure_cuda_cli()
    if ARCHIVE.exists():
        raise SystemExit("Frozen stream archive exists")
    d = old.base.read(PUB / "data.json")
    assert all(old.base.sha(WORK / k) == v for k, v in d["files"].items())
    original = old.base.read(old.PUB / "data.json")
    assert all(old.base.sha(old.WORK / k) == v for k, v in original["files"].items())
    for arm in ("dense", "endpoint"):
        assert old.base.read(MODEL / arm / "fit.json")["bindings"] == bindings()
    local = backoff()
    with gpu_owner():
        bank = extended_bank("cuda")
        feedback = Feedback(bank)
        models = {
            "dense": CausalBookWriter("dense", "cuda").net,
            "endpoint": CausalBookWriter("endpoint", "cuda").net,
            "released": PassageWriter("intact", "cuda").net,
        }
        forms = old.base.read(WORK / "forms.json")
        ink = old.base.Ink()
        tmp = ARCHIVE.with_suffix(".partial")
        with ZipFile(tmp, "w", ZIP_DEFLATED, compresslevel=4) as z:
            for split in (*FRESH, *OLD):
                table = load_table(split)
                for j in range(3 if split in FRESH else 1):
                    gold = []
                    for k in table["target"][:, j]:
                        ids = ink.encode(forms[k])
                        gold.append(ink.array()[ids])
                    record_ink(z, split + f"/gold{j+1}", gold)
                for arm, net in {**models, "counts": None}.items():
                    generate(
                        net,
                        feedback,
                        table,
                        z,
                        split + "/" + arm,
                        count=local if arm == "counts" else None,
                        budget=3 if split in FRESH else 1,
                    )
                    print(split, arm, flush=True)
                if split in FRESH:
                    indices = donors(table["source"], table["group"], table["task"])
                    write_array(z, split + "/donors", indices)
                    src, m = observations(table["source"], table["marks"], indices)
                    generate(
                        models["dense"],
                        feedback,
                        {**table, "source": src, "marks": m},
                        z,
                        split + "/donor",
                        budget=1,
                    )
                    generate(
                        models["dense"],
                        feedback,
                        table,
                        z,
                        split + "/long",
                        long=True,
                        budget=12,
                    )
        tmp.replace(ARCHIVE)
    print("Frozen", old.base.sha(ARCHIVE), flush=True)


if __name__ == "__main__":
    main()
