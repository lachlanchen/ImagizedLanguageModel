"""Score both natural written answers under each of two questions about one passage."""

import argparse
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from scripts.answer_dialogue_surface import load
from scripts.dialogue_ink_data import pack
from scripts.dialogue_ink_images import WordImages, trims
from scripts.evaluate_compose_dialogue import paths_for, confidence
from scripts.evaluate_optical_passage_qa import normalized
from scripts.dialogue_teacher_stories import absent
from scripts.prepare_dialogue_ink import printable, dump, sha
from scripts.prepare_compose_dialogue import prepared, MODEL
from scripts.train_dialogue_surface import representation, corpus
from scripts.train_span_choice import gpu_owner


def pairs(data, alphabet):
    groups = defaultdict(list)
    rows = data["rows"]
    pair = []
    for i, r in enumerate(rows):
        if (
            printable(r[5])
            and 1 <= len(r[5].split()) <= 16
            and absent([r[5]], (data["pages"][r[1]], r[3], r[4]))
        ):
            groups[r[1]].append(i)
    for pid, indices in groups.items():
        used = set()

        def covers(i, j):
            source = (data["pages"][pid] + " " + rows[i][3]).split()
            possible = {t for w in source for t in trims(w, 1) if t}
            return all(
                w in possible or (len(w) <= 24 and set(w) <= alphabet)
                for w in rows[j][5].split()
            )

        for k, i in enumerate(indices):
            if i in used:
                continue
            for j in indices[k + 1 :]:
                if (
                    j in used
                    or normalized(rows[i][3]) == normalized(rows[j][3])
                    or normalized(rows[i][5]) == normalized(rows[j][5])
                ):
                    continue
                if all(covers(q, a) for q in (i, j) for a in (i, j)):
                    pair.append((i, j))
                    used.update((i, j))
                    break
    if not pair:
        raise ValueError("No eligible pairs")
    return pair


def log_likelihood(net, ds, bank, face, unit, query, answers):
    device = net.palette.device
    words = [[face(w) for w in a.split()] for a in answers]
    ids, lens = pack([[bank.ids[w] for w in row] for row in words], device)
    x, variants = ds.inputs(np.array(query))
    memory, _ = net.memory(*x)
    previous = bank.features[F.pad(ids.long(), (1, 0)), :50]
    route, pointer, z, _ = net.step(memory, previous, memory[-1][None])
    route = route.double().log_softmax(-1)
    width = ids.size(1)
    positions = variants.flatten(1)
    ptr = (
        pointer[:, :width]
        .flatten(2)
        .double()
        .masked_fill(positions[:, None] == 0, -torch.inf)
        .log_softmax(-1)
    )
    targets, _ = pack([[ds.bank.ids.get(w, -1) for w in row] for row in words], device)
    matches = (positions[:, None] == targets[:, :, None]) & (targets[:, :, None] > 0)
    copy = ptr.masked_fill(~matches, -torch.inf).logsumexp(-1)
    spell = torch.full_like(copy, -torch.inf)
    alphabet = {v.numpy().tobytes(): i for i, v in enumerate(net.palette.cpu()) if i}
    coords = []
    encoded = []
    for b, row in enumerate(words):
        for t, w in enumerate(row):
            cells = [alphabet.get(w[i : i + unit]) for i in range(0, len(w), unit)]
            if len(cells) <= net.max_letters and None not in cells:
                coords.append((b, t))
                encoded.append(cells + [0])
    if coords:
        letters, lengths = pack(encoded, device)
        letters = letters.long()
        b, t = zip(*coords)
        prefix = F.pad(letters[:, :-1], (1, 0))
        h, _ = net.spell(net.letter_inputs()[prefix], z[b, t][None].contiguous())
        logits = net.letter_out(h).double()
        logits[:, 0, 0] = -torch.inf
        ll = logits.log_softmax(-1).gather(-1, letters[:, :, None]).squeeze(-1)
        spell[b, t] = ll.masked_fill(
            torch.arange(letters.size(1), device=device)[None] >= lengths[:, None], 0
        ).sum(1)
    ll = torch.logaddexp(route[:, :width, 1] + copy, route[:, :width, 2] + spell)
    stop = route[torch.arange(len(lens), device=device), lens, 0]
    total = (
        ll.masked_fill(
            torch.arange(width, device=device)[None] >= lens[:, None], 0
        ).sum(1)
        + stop
    )
    if not total.isfinite().all():
        raise ValueError("Unreachable candidate entered evaluation")
    return total.tolist()


@torch.inference_mode()
def run(root, out):
    out.mkdir(parents=True, exist_ok=False)
    data = prepared()["fresh"]
    rows = data["rows"]
    net = load(paths_for("ink", root)["plain"])
    f, _, _ = representation("ink")
    palette = {r.numpy().tobytes() for r in net.palette[1:]}
    alphabet = {chr(i) for i in range(33, 127) if f(chr(i)) in palette}
    matched = pairs(data, alphabet)
    result = {
        "pairs": len(matched),
        "stories": len({rows[i][1] for i, _ in matched}),
        "history": {},
    }
    for history in ("actual", "empty"):
        select = dict(pages=data["pages"], rows=[list(r) for r in rows])
        if history == "empty":
            for r in select["rows"]:
                r[4] = ""
        group = result["history"][history] = {}
        for mode in ("ink", "byte"):
            face, unit, _ = representation(mode)
            paths = paths_for(mode, root)
            for arm, folder in paths.items():
                net = load(folder, "cuda")
                ds = corpus(select, "fresh", net, face, unit)
                words = {
                    face(w)
                    for pair in matched
                    for i in pair
                    for w in rows[i][5].split()
                }
                bank = WordImages(words, net.reader, None, unit, net.encode_words)
                scores = []
                for off in range(0, len(matched), 8):
                    trial = [
                        (q, a) for ij in matched[off : off + 8] for q in ij for a in ij
                    ]
                    scores.extend(
                        log_likelihood(
                            net,
                            ds,
                            bank,
                            face,
                            unit,
                            [q for q, _ in trial],
                            [rows[a][5] for _, a in trial],
                        )
                    )
                values = np.array(scores).reshape(-1, 2, 2)
                one = values[:, 0, 0] - values[:, 0, 1]
                two = values[:, 1, 1] - values[:, 1, 0]
                delta = one + two
                inds = np.arange(len(matched))
                pseudo = [rows[i] for i, _ in matched]
                path = out / f"{history}-{mode}-{arm}.json"
                dump(
                    path,
                    [
                        dict(turns=[i, j], story=rows[i][1], ll=v.tolist())
                        for (i, j), v in zip(matched, values)
                    ],
                )
                group[mode + "/" + arm] = dict(
                    assigned=int((delta > 0).sum()),
                    ties=int((delta == 0).sum()),
                    both=int(((one > 0) & (two > 0)).sum()),
                    independent=int((one > 0).sum() + (two > 0).sum()),
                    mean_log_odds=float(delta.mean()),
                    interval95=confidence((delta > 0).astype(float), pseudo, inds)[
                        "interval95"
                    ],
                    likelihoods=sha(path),
                )
        print(history, group, flush=True)
    dump(out / "results.json", result)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", type=Path, default=MODEL)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    with gpu_owner():
        run(a.models, a.output)
