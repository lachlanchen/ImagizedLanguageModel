"""Exchange the passage while holding both pictured questions and candidate replies fixed."""

import argparse
import json
from pathlib import Path
import numpy as np
import torch
from scripts.answer_dialogue_surface import load
from scripts.dialogue_ink_images import WordImages, trims
from scripts.evaluate_compose_dialogue import confidence, paths_for
from scripts.evaluate_optical_passage_qa import normalized
from scripts.prepare_compose_dialogue import prepared, MODEL
from scripts.prepare_dialogue_ink import sha, dump
from scripts.score_compose_answer_pairs import log_likelihood
from scripts.train_dialogue_surface import corpus, representation
from scripts.train_span_choice import gpu_owner


@torch.inference_mode()
def run(root, original, out):
    out.mkdir(parents=True, exist_ok=False)
    data = prepared()["fresh"]
    rows = data["rows"]
    old = json.loads((original / "results.json").read_text())
    records = json.loads((original / "empty-ink-plain.json").read_text())
    pages = list(dict.fromkeys(v["story"] for v in records))
    exchange = {k: pages[(i + 1) % len(pages)] for i, k in enumerate(pages)}
    source = {p: data["pages"][q] for p, q in exchange.items()}
    filtered = []
    without = []
    overlap = []
    refids = []
    face, _, _ = representation("ink")
    net = load(paths_for("ink", root)["plain"])
    cells = set(p.numpy().tobytes() for p in net.palette[1:])
    alphabet = {chr(i) for i in range(33, 127) if face(chr(i)) in cells}
    for k, v in enumerate(records):
        ij = v["turns"]
        qq = [set(normalized(rows[i][3]).split()) for i in ij]
        aa = [set(normalized(rows[i][5]).split()) for i in ij]
        page = source[v["story"]]

        def covers(q, a):
            allowed = {
                form
                for w in (page + " " + rows[q][3]).split()
                for form in trims(w, 1)
                if form
            }
            return all(
                w in allowed or (len(w) <= 24 and set(w) <= alphabet)
                for w in rows[a][5].split()
            )

        if not all(covers(q, a) for q in ij for a in ij):
            continue
        refids.append(k)
        filtered.append(ij)
        s = np.array([[len(q & a) for a in aa] for q in qq])
        without.append(s.max() == 0)
        overlap.append(int(s[0, 0] + s[1, 1] - s[0, 1] - s[1, 0]))
    pseudo = [rows[i] for i, _ in filtered]
    idx = np.arange(len(filtered))
    result = dict(
        matched=len(idx),
        original=len(records),
        zero_overlap=int(sum(without)),
        overlap_wins=sum(x > 0 for x in overlap),
        overlap_ties=overlap.count(0),
        models={},
    )
    modified = {"pages": data["pages"] | source, "rows": [list(r) for r in rows]}
    for r in modified["rows"]:
        r[4] = ""
    for mode in ("ink", "byte"):
        face, unit, _ = representation(mode)
        for arm, folder in paths_for(mode, root).items():
            tag = mode + "/" + arm
            file = original / f"empty-{mode}-{arm}.json"
            if sha(file) != old["history"]["empty"][tag]["likelihoods"]:
                raise ValueError(file)
            prior = json.loads(file.read_text())
            if any(
                prior[k]["turns"] != ij for k, ij in zip(refids, filtered, strict=True)
            ):
                raise ValueError("Pair IDs changed")
            before = np.array([prior[k]["ll"] for k in refids])
            net = load(folder, "cuda")
            ds = corpus(modified, "fresh", net, face, unit)
            words = {face(w) for ij in filtered for i in ij for w in rows[i][5].split()}
            bank = WordImages(words, net.reader, None, unit, net.encode_words)
            scores = []
            for start in range(0, len(idx), 8):
                qa = [
                    (q, a) for ij in filtered[start : start + 8] for q in ij for a in ij
                ]
                scores.extend(
                    log_likelihood(
                        net,
                        ds,
                        bank,
                        face,
                        unit,
                        [q for q, _ in qa],
                        [rows[a][5] for _, a in qa],
                    )
                )
            after = np.array(scores).reshape(-1, 2, 2)

            def win(s):
                return (s[:, 0, 0] - s[:, 0, 1] + s[:, 1, 1] - s[:, 1, 0]) > 0

            b, t = win(before), win(after)
            entry = {}
            for label, selected in [("all", idx), ("zero_overlap", idx[without])]:
                entry[label] = dict(
                    n=len(selected),
                    original=int(b[selected].sum()),
                    exchanged=int(t[selected].sum()),
                    difference=confidence(b.astype(float) - t, pseudo, selected),
                )
            f = out / f"{mode}-{arm}.json"
            dump(
                f,
                [
                    dict(turns=ij, source=exchange[rows[ij[0]][1]], ll=s.tolist())
                    for ij, s in zip(filtered, after, strict=True)
                ],
            )
            result["models"][tag] = entry | {"likelihoods": sha(f)}
    dump(out / "results.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", type=Path, default=MODEL)
    p.add_argument("--original", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    with gpu_owner():
        run(a.models, a.original, a.output)
