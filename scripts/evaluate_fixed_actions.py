"""Evaluate unrestricted ink replies, retaining the paired source-position logits."""

import argparse
from time import monotonic

import torch

from ilm.visual_lm.book_participant_reader import (
    InkBank,
    WIDTH,
    best_strip,
    group_spans,
)
from ilm.visual_lm.optical_role_reader import embeddings, logits, restore
from scripts.evaluate_role_question_binding import decode
from scripts.measure_writer_energy import snapshot, consumed
from scripts.prepare_fixed_actions import PUBLIC, WORK, read, record, sha
from scripts.train_optical_role_adapter import BASE, PUBLIC as PREVIOUS, WORK as FIT
from scripts.train_span_choice import gpu_owner
from scripts.train_typeface_writer import Face, FACES


@torch.inference_mode()
def evaluate(arm):
    spec = read(PUBLIC / "inputs.json")
    path = WORK / "inputs.json"
    assert spec["inputs"] == sha(path)
    items = read(path)
    if arm == "text":
        rows, cost = decode("text", items, WORK)
    else:
        hashes = read(PREVIOUS / "corpus.json")["base"]
        assert all(sha(BASE / k) == v for k, v in hashes.items())
        fit = "aligned" if arm == "generic" else arm
        adapter = None if arm == "parent" else FIT / fit / "adapter.safetensors"
        if adapter:
            assert sha(adapter) == read(PREVIOUS / (fit + "_fit.json"))["weights"]
        bank = InkBank()
        face = Face(FACES[0])
        pairs = []
        for r in items:
            q = (
                ("誰？" if r["lang"] == "zh" else "Who?")
                if arm == "generic"
                else r["question"]
            )
            pairs.append((bank.encode(face(r["page"])), bank.encode(face(q))))
        torch.cuda.reset_peak_memory_stats()
        start = monotonic()
        first = snapshot()
        net = restore(BASE, "cuda", adapter)
        emb = embeddings(net, bank)
        rows = []
        for i in range(0, len(items), 24):
            batch = pairs[i : i + 24]
            z = logits(net, emb, batch).cpu().numpy()
            for j, (p, _) in enumerate(batch):
                r = items[i + j]
                result = z[j, : len(p)]
                groups = group_spans(p)
                emitted = best_strip(result, groups)
                pixels = b"".join(bank.cells[k] for k in emitted)
                scores = []
                for name, offset in (
                    ("answer", "source_start"),
                    ("other", "rival_start"),
                ):
                    cells = tuple(bank.encode(face(r[name])))
                    start_at = r[offset]
                    assert groups[cells] == [start_at * WIDTH + len(cells) - 1]
                    scores.append(float(result[start_at, len(cells) - 1]))
                rows.append(
                    dict(
                        reply=pixels.hex(),
                        ended=True,
                        correct=pixels == face(r["answer"]),
                        rival=pixels == face(r["other"]),
                        source_minus_rival=scores[0] - scores[1],
                    )
                )
            if (
                monotonic() - start > 1200
                or torch.cuda.max_memory_allocated() > 12 * 1024**3
            ):
                raise RuntimeError("Resource ceiling exceeded")
        cost = dict(
            seconds=monotonic() - start,
            energy=consumed(first, snapshot()),
            peak_bytes=torch.cuda.max_memory_allocated(),
            base=hashes,
            adapter=sha(adapter) if adapter else None,
        )
    for row, item in zip(rows, items, strict=True):
        row.update(
            {
                k: item[k]
                for k in (
                    "group",
                    "scene",
                    "key",
                    "chapter",
                    "order",
                    "world",
                    "lang",
                    "h",
                )
            }
        )
    outfile = WORK / (arm + ".json")
    record(outfile, rows)
    record(
        PUBLIC / (arm + ".json"),
        dict(inputs=spec["inputs"], replies=sha(outfile), count=len(rows), **cost),
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "arm", choices=("parent", "aligned", "independent", "generic", "text")
    )
    with gpu_owner():
        torch.set_num_threads(2)
        evaluate(ap.parse_args().arm)
