"""Generate word-image answer proposals and measure their coverage on difficult questions."""

import argparse
import gzip
import json
from pathlib import Path
from time import monotonic
import numpy as np
import torch
from ilm.visual_lm.image_copy_search import copy_proposals
from scripts.answer_dialogue_surface import load
from scripts.evaluate_compose_dialogue import assignment
from scripts.dialogue_teacher_stories import absent
from scripts.evaluate_optical_passage_qa import normalized, metrics
from scripts.prepare_compose_dialogue import MODEL, prepared, kind
from scripts.prepare_dialogue_ink import dump, sha
from scripts.train_dialogue_surface import corpus, representation
from scripts.train_span_choice import gpu_owner
from scripts.train_dialogue_subword import timed
from scripts.measure_writer_energy import snapshot


def study(root, control, out):
    out.mkdir(parents=True, exist_ok=False)
    materials = prepared()
    face, unit, sep = representation("ink")
    decode = {face(chr(i)): chr(i) for i in range(32, 127)}
    net = load(root / "ink/lists", "cuda")
    stats = {}
    for group in ("fresh", "lists"):
        d = materials[group]
        rows = d["rows"]
        bank = corpus(d, "fresh", net, face, unit)
        indices = []
        for i, r in enumerate(rows):
            fields = (d["pages"][r[1]], r[3], r[4])
            if (
                kind(r[5], fields) == "assembly"
                if group == "fresh"
                else absent([r[5]], fields)
            ):
                indices.append(i)
        q, p = assignment(rows)
        cache = {}

        def spell(word):
            if word not in cache:
                raw = bank.bank.raw[word]
                cache[word] = "".join(
                    decode[raw[j : j + unit]] for j in range(0, len(raw), unit)
                )
            return cache[word]

        f = control / f"{group}-ink-lists.json.gz"
        bound = json.loads((control / "results.json").read_text())[group]["models"][
            "ink/lists"
        ]
        if (
            sha(f) != bound["outputs"]
            or sha(root / "ink/lists/writer.safetensors") != bound["weights"]
        ):
            raise ValueError("Changed baseline")
        greedy = json.loads(gzip.decompress(f.read_bytes()))
        answers = [{} for i in indices]
        start = monotonic()
        meter = snapshot()
        torch.cuda.reset_peak_memory_stats()
        for key in ("original", "question", "passage"):
            for off in range(0, len(indices), 8):
                ix = np.array(indices[off : off + 8])
                x, v = bank.inputs(
                    ix,
                    q[ix] if key == "question" else None,
                    p[ix] if key == "passage" else None,
                )
                candidates = copy_proposals(net, bank.bank, x, v)
                for j, entries in enumerate(candidates):
                    if len({ids for ids, _ in entries}) != len(entries):
                        raise ValueError("Repeated physical hypothesis")
                    answers[off + j][key] = [
                        (" ".join(spell(w) for w in ids), score)
                        for ids, score in entries
                    ]
        cost = timed(start, meter)
        result = {}
        for arm in ("original", "question", "passage"):
            ranks = []
            top = []
            greed = []
            for i, r in zip(indices, answers):
                target = normalized(rows[i][5])
                ranks.append(
                    next(
                        (
                            j + 1
                            for j, (s, _) in enumerate(r[arm])
                            if normalized(s) == target
                        ),
                        0,
                    )
                )
                top.append(metrics(r[arm][0][0], [rows[i][5]])[1])
                greed.append(
                    bool(
                        greedy[i][arm]["stop"]
                        and normalized(greedy[i][arm]["answer"]) == target
                    )
                )
            result[arm] = dict(
                ranks=ranks,
                top_exact=ranks.count(1),
                top8=sum(0 < n <= 8 for n in ranks),
                top32=sum(n > 0 for n in ranks),
                f1=float(np.mean(top)),
                greedy_exact=sum(greed),
            )
        source = lambda x: bool(x[0] == 1 and x[1] != 1 and x[2] != 1)
        result["grounded_top"] = sum(
            source(r)
            for r in zip(
                *(result[k]["ranks"] for k in ("original", "question", "passage"))
            )
        )
        for v in result.values():
            if isinstance(v, dict):
                v.pop("ranks")
        file = out / f"{group}-proposals.json.gz"
        file.write_bytes(
            gzip.compress(
                json.dumps(list(zip(indices, answers)), separators=(",", ":")).encode(),
                mtime=0,
            )
        )
        stats[group] = dict(
            n=len(indices), results=result, outputs=sha(file), cost=cost
        )
        print(group, stats[group], flush=True)
    dump(out / "results.json", stats)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", type=Path, default=MODEL)
    p.add_argument("--control", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    with gpu_owner():
        study(a.models, a.control, a.output)
