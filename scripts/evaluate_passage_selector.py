"""Archive actual free emissions and independent controls on three disjoint panels."""

from collections import Counter
from io import BytesIO
from zipfile import ZIP_DEFLATED, ZipFile
import json

import numpy as np
import torch

from ilm.visual_lm.passage_selector import PassageWriter, VisualBank
from scripts.train_passage_selector import load_bank, bindings
from scripts.prepare_passage_selector import WORK, PUB, MODEL, ROOT, SEED, base
from scripts.train_span_choice import gpu_owner
from scripts.verse_gpu_runtime import configure_cuda_cli

ARCHIVE = ROOT / "publication/ilm-nature/evidence/learned_passage_words.zip"
SPLITS = ("fiction", "gutenberg", "lambada")


def write_array(z, name, data):
    stream = BytesIO()
    np.save(stream, data, allow_pickle=False)
    z.writestr(name + ".npy", stream.getvalue())


def read_array(z, name):
    return np.load(BytesIO(z.read(name + ".npy")), allow_pickle=False)


def record_ink(z, name, rows):
    offsets = np.cumsum([0] + [len(x) for x in rows], dtype=np.int64)
    write_array(z, name + "-offsets", offsets)
    write_array(z, name + "-pixels", np.concatenate(rows))


def read_ink(z, name):
    stops = read_array(z, name + "-offsets")
    pixels = read_array(z, name + "-pixels")
    assert stops[0] == 0 and stops[-1] == len(pixels) and (np.diff(stops) >= 0).all()
    return [pixels[a:b] for a, b in zip(stops[:-1], stops[1:], strict=True)]


def donors(src, group, task):
    rng = np.random.default_rng(SEED + 73)
    lengths = (src != 0).sum(1)
    choices = []
    for i, (n, g, t) in enumerate(zip(lengths, group, task, strict=True)):
        eligible = np.flatnonzero((group != g) & (task == t) & (lengths >= n))
        if not len(eligible):
            eligible = np.flatnonzero((group != g) & (task == t))
        dist = np.abs(lengths[eligible] - n)
        choices.append(rng.choice(eligible[dist == dist.min()]))
    return np.array(choices)


def observations(rows, marks, donor_ids):
    swapped = np.zeros_like(rows)
    gaps = np.zeros_like(marks)
    for i, d in enumerate(donor_ids):
        n = np.count_nonzero(rows[i])
        k = min(6, n)
        other = rows[d][rows[d] != 0]
        gm = marks[d][rows[d] != 0]
        take = min(n - k, max(0, len(other) - k))
        swapped[i, -k:] = rows[i, -k:]
        gaps[i, -k:] = marks[i, -k:]
        if take:
            swapped[i, -k - take : -k] = other[-k - take : -k]
            gaps[i, -k - take : -k] = gm[-k - take : -k]
    return swapped, gaps


def salience(row, bank):
    viable = [
        i
        for i in reversed(row)
        if i and 0 < len(bank.words[i]) <= 23 and 0 not in bank.words[i]
    ]
    counts = Counter(viable)
    return (
        bank.words[max(counts, key=lambda k: counts[k] * bank.face.idf(bank.words[k]))]
        if counts
        else ()
    )


def predict(model, bank, src, marks, tasks):
    outputs = []
    ends = []
    for start in range(0, len(src), 64):
        ss, mm, tt = [
            torch.as_tensor(v[start : start + 64], device=bank.device).long()
            for v in (src, marks, tasks)
        ]
        words, stops = model.predict(bank, ss, mm, tt)
        outputs.extend(words)
        ends.extend(stops)
    return outputs, np.array(ends)


def continuations(model, bank, table, output, ends, archive, split):
    groups = table["group"]
    task = table["task"]
    indices = np.concatenate(
        [np.flatnonzero((groups == g) & (task == 0))[:4] for g in np.unique(groups)]
    )
    outputs = [[output[i][0]] for i in indices]
    stopped = ends[indices].copy()
    stop_step = np.where(stopped, 1, 0)
    contexts = [[bank.words[w] for w in table["source"][i] if w] for i in indices]
    gaps = [table["marks"][i][table["source"][i] != 0].tolist() for i in indices]
    for step in (2, 3):
        active = np.flatnonzero(~stopped)
        if not len(active):
            break
        for k in active:
            contexts[k] = (contexts[k] + [outputs[k][-1]])[-96:]
            gaps[k] = (gaps[k] + [0])[-96:]
        forms = [(), *dict.fromkeys(w for k in active for w in contexts[k])]
        small = VisualBank(bank.face, forms, bank.device)
        handles = {w: i for i, w in enumerate(forms)}
        s = np.zeros((len(active), 96), np.int32)
        m = np.zeros_like(s)
        for j, k in enumerate(active):
            n = len(contexts[k])
            s[j, -n:] = [handles[w] for w in contexts[k]]
            m[j, -n:] = gaps[k]
        words, flags = predict(model, small, s, m, np.zeros(len(active)))
        for k, word, flag in zip(active, words, flags, strict=True):
            outputs[k].append(word[0])
            if flag:
                stopped[k] = True
                stop_step[k] = step
    prefix = split + "/stream"
    write_array(archive, prefix + "-rows", indices)
    write_array(archive, prefix + "-stop_step", stop_step)
    for j in range(3):
        record_ink(
            archive,
            prefix + str(j),
            [bank.face.pixels(seq[j] if j < len(seq) else ()) for seq in outputs],
        )


@torch.inference_mode()
def main():
    configure_cuda_cli()
    data = base.read(PUB / "data.json")
    assert all(base.sha(WORK / k) == h for k, h in data["files"].items())
    fits = [base.read(MODEL / arm / "fit.json") for arm in ("intact", "shuffled")]
    assert all(r["bindings"] == bindings() for r in fits)
    if ARCHIVE.exists():
        raise SystemExit("Frozen output archive exists")
    with gpu_owner():
        bank = load_bank("cuda")
        models = [PassageWriter(arm, "cuda").net for arm in ("intact", "shuffled")]
        forms = json.loads((WORK / "forms.json").read_text())
        face = base.Ink()
        tmp = ARCHIVE.with_suffix(".partial")
        with ZipFile(tmp, "w", ZIP_DEFLATED, compresslevel=4) as archive:
            for split in SPLITS:
                with np.load(WORK / (split + ".npz")) as arrays:
                    table = {k: arrays[k] for k in arrays.files}
                    src, marks, task, groups, y = (
                        table[k] for k in ("source", "marks", "task", "group", "target")
                    )
                    donor = donors(src, groups, task)
                    ds, dm = observations(src, marks, donor)
                    for k, v in [
                        ("donor", donor),
                        ("task", task),
                        ("group", groups),
                        ("ends", table["ends"]),
                        ("present", (src == y[:, 0, None]).any(1)),
                    ]:
                        write_array(archive, split + "/" + k, v)
                    for j in range(3 if split != "lambada" else 1):
                        gold = []
                        for address in y[:, j]:
                            if address:
                                # Render the full gold, including any symbol outside the neural output alphabet.
                                ids = face.encode(forms[address])
                                gold.append(face.array()[ids])
                            else:
                                gold.append(np.zeros((0, 128), dtype=np.uint8))
                        record_ink(archive, split + "/gold" + str(j), gold)
                    for name, net, ss, mm in [
                        ("intact", models[0], src, marks),
                        ("shuffled", models[1], src, marks),
                        ("donor", models[0], ds, dm),
                    ]:
                        words, ends = predict(net, bank, ss, mm, task)
                        record_ink(
                            archive,
                            split + "/" + name,
                            [bank.face.pixels(r[0]) for r in words],
                        )
                        write_array(archive, split + "/" + name + "-stop", ends)
                        if name == "intact":
                            for j, route in enumerate(("pointer", "spell"), 1):
                                record_ink(
                                    archive,
                                    split + "/" + route,
                                    [bank.face.pixels(r[j]) for r in words],
                                )
                            if split != "lambada":
                                continuations(
                                    net, bank, table, words, ends, archive, split
                                )
                        print(split, name, len(words), flush=True)
                    record_ink(
                        archive,
                        split + "/salience",
                        [bank.face.pixels(salience(row, bank)) for row in src],
                    )
        tmp.replace(ARCHIVE)
    print("Frozen output", base.sha(ARCHIVE), flush=True)


if __name__ == "__main__":
    main()
