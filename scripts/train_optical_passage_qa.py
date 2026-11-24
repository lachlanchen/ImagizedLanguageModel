"""Fit a passage-span reader on visual word states from SQuAD 1.1."""

import argparse
from pathlib import Path
from time import monotonic
import json
import numpy as np
import torch
from torch.nn import functional as F
from safetensors.torch import save_file

from ilm.visual_lm.optical_passage_qa import OpticalPassageQA, word_runs
from scripts.answer_optical_passage_qa import (
    optical_words,
    visual_words,
    RELEASE,
    PARENT,
)
from scripts.prepare_optical_passage_qa import WORK, PUB, sha, dump
from scripts.train_typeface_writer import Face, FACES
from scripts.train_span_choice import gpu_owner
from scripts.measure_writer_energy import snapshot, consumed

SEED = 710043


def corpus():
    file = WORK / "corpus.json"
    if sha(file) != json.loads((PUB / "corpus.json").read_text())["corpus_sha256"]:
        raise ValueError("Changed corpus")
    return json.loads(file.read_text())


class VisualCorpus:
    """Integer addresses deduplicate immutable image encodings in the batch cache."""

    def __init__(self, net, encoder, pages, rows, device):
        face = Face(FACES[0])
        page_ids = sorted({r[2] for r in rows})
        questions = sorted({r[4] for r in rows})
        pidx = {k: i for i, k in enumerate(page_ids)}
        qidx = {k: i for i, k in enumerate(questions)}
        pwords = [pages[k].split() for k in page_ids]
        qwords = [q.split() for q in questions]
        words = sorted({w for line in pwords + qwords for w in line})
        wordid = {w: i + 1 for i, w in enumerate(words)}
        chunks = [torch.zeros(1, 115, device=device)]
        for i in range(0, len(words), 256):
            lines = [
                np.frombuffer(face(w), np.uint8).reshape(-1, 128)
                for w in words[i : i + 256]
            ]
            if any(len(word_runs(l)) != 1 for l in lines):
                raise ValueError("Unrepresentable internal glyph")
            chunks.append(visual_words(net, encoder, lines))
        self.features = torch.cat(chunks)
        self.widths = torch.tensor([0] + list(map(len, words)), device=device)

        def pack(lines):
            a = np.zeros((len(lines), max(map(len, lines))), np.int32)
            for j, line in enumerate(lines):
                a[j, : len(line)] = [wordid[w] for w in line]
            return torch.from_numpy(a).to(device), torch.tensor(
                list(map(len, lines)), device=device
            )

        self.page, self.plen = pack(pwords)
        self.question, self.qlen = pack(qwords)
        self.pidx = torch.tensor([pidx[r[2]] for r in rows], device=device)
        self.qidx = torch.tensor([qidx[r[4]] for r in rows], device=device)
        self.target = torch.tensor([r[5] for r in rows], device=device)
        self.rows = rows

    def batch(self, indices):
        p, q = self.pidx[indices], self.qidx[indices]
        pl, ql = self.plen[p], self.qlen[q]
        # Use common padded shapes to avoid resynchronizing every optimization step.
        pi, qi = self.page[p], self.question[q]
        return (
            (self.features[pi], pl, self.features[qi], ql),
            self.widths[pi],
            self.target[indices],
        )


def fit(output):
    if json.loads((PUB / "teacher.json").read_text())["qualified"]:
        raise ValueError("Protocol requires the separate paired teacher comparison")
    if (output / "reader.safetensors").exists():
        raise FileExistsError(RELEASE)
    data = corpus()
    rows = [r for r in data["rows"] if r[0] == "train"]
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    net = OpticalPassageQA().cuda()
    encoder = optical_words("cuda")
    meter = snapshot()
    begin = monotonic()
    bank = VisualCorpus(net, encoder, data["pages"], rows, "cuda")
    torch.cuda.synchronize()
    prep = dict(seconds=monotonic() - begin, energy=consumed(meter, snapshot()))
    opt = torch.optim.AdamW(net.parameters(), lr=0.001, weight_decay=0.015)
    trace = []
    begin = monotonic()
    meter = snapshot()
    torch.cuda.reset_peak_memory_stats()
    b = torch.arange(48, device="cuda")
    selected = np.zeros(len(rows), bool)
    for step in range(4000):
        indices = rng.integers(len(rows), size=48)
        selected[indices] = True
        x, _, t = bank.batch(indices)
        s, cuts = net(*x)
        l, r, dl, dr = t.T
        loss = (
            F.cross_entropy(s[:, :, 0], l)
            + F.cross_entropy(s[:, :, 1], r)
            + 0.35 * F.cross_entropy(cuts[b, l, 0], dl)
            + 0.35 * F.cross_entropy(cuts[b, r, 1], dr)
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 3.0)
        opt.step()
        if (step + 1) % 250 == 0:
            trace.append(float(loss.detach()))
            print(step + 1, trace[-1], round(monotonic() - begin, 1), flush=True)
            if (
                monotonic() - begin > 1800
                or torch.cuda.max_memory_allocated() > 12 * 2**30
            ):
                raise RuntimeError("Exceeded registered resources")
    torch.cuda.synchronize()
    cost = dict(
        seconds=monotonic() - begin,
        energy=consumed(meter, snapshot()),
        peak_bytes=torch.cuda.max_memory_allocated(),
    )
    output.mkdir(parents=True, exist_ok=True)
    file = output / "reader.safetensors"
    save_file(
        {k: v.detach().cpu().contiguous() for k, v in net.state_dict().items()},
        str(file),
    )
    meta = dict(
        architecture="optical_passage_qa_v1",
        sha256=sha(file),
        optical_parent=sha(PARENT),
        training_corpus=sha(WORK / "corpus.json"),
        seed=SEED,
        updates=4000,
        presentations=192000,
        distinct_sampled=int(selected.sum()),
        trainable=sum(v.numel() for v in net.parameters()),
        frozen_optical=sum(v.numel() for v in encoder.parameters()),
        loss_trace=trace,
        preprocessing=prep,
        fit=cost,
    )
    dump(output / "weights.json", meta)
    print(meta, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with gpu_owner():
        torch.set_num_threads(2)
        torch.set_float32_matmul_precision("high")
        fit(args.output)
