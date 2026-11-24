"""Compare the question readers on held books; archive physical replies and errors."""

import csv
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

from ilm.visual_lm.causal_glyph_writer import write_strip
from ilm.visual_lm.gloss_focus import cell_features, encoding_windows
from ilm.visual_lm.ordered_gloss_question import OrderedGlossFocus
from scripts.answer_visual_gloss import answer
from scripts.everyday_gloss_questions import LABELS, TEST
from scripts.prepare_asked_gloss import ROOT, WORK, PRIOR, ORIGINAL, dump, read, sha
from scripts.train_asked_gloss import packed_batch, rows_and_split
from scripts.train_typeface_writer import Face, FACES
from scripts.write_visual_gloss import restore, respond

DEST = ROOT / "references/evidence/everyday_gloss_v1"
ARMS = ["ordered_contrast", "everyday", "everyday_exchanged"]


def fresh_rows(face, existing):
    editions = Path(os.environ.get("ILM_CLASSICAL_BOOKS", ROOT.parent / "ZhJpBook/books"))
    prior = read(ROOT / "references/evidence/visual_word_senses_20261124/split.json")
    assert sha(PRIOR / "train.npz") == prior["data"]["train"]["inputs"]
    assert sha(PRIOR / "pixels.npy") == prior["pixels"]
    pixels = np.load(PRIOR / "pixels.npy")[0]
    unique, inverse = np.unique(pixels, axis=0, return_inverse=True)
    ids = {x.tobytes(): i for i, x in enumerate(unique)}
    with np.load(PRIOR / "train.npz") as data:
        windows = {w.tobytes() for w in inverse[data["x"]].astype("<u2")}

    def encode(c):
        b = face.cell(c)
        return ids.setdefault(b, len(ids))

    seen = {tuple(map(encode, r["page"])) for r in existing}
    sources, dropped, rows = {}, [], []
    with (DEST / "passages.tsv").open() as f:
        for r in csv.DictReader(f, delimiter="\t"):
            book, page = r["book"], r["passage"]
            path = editions / book / "markdown/wenyan.md"
            text = path.read_text()
            start = text.index(page)
            chapter = text[:start].rsplit("\n## ", 1)[-1].split("\n", 1)[0]
            sources.setdefault(book, sha(path))
            focus = [page.index(r[k]) for k in ("predicate", "object")]
            assert all(page.count(page[i]) == 1 for i in focus)
            v = tuple(map(encode, page))
            padded = (encode(" "),) * 8 + v + (encode(" "),) * 8
            if v in seen:
                dropped.append([book, page, "duplicate passage"])
                continue
            if any(np.asarray(padded[i:i + 17], dtype="<u2").tobytes() in windows for i in focus):
                dropped.append([book, page, "trained writer window"])
                continue
            assert all(ord(c) in face.coverage for c in page)
            seen.add(v)
            rows.append(dict(book=book, page=page, chapter=chapter, focus=focus,
                             gloss=[r[k + "_glosses"].split("|") for k in ("predicate", "object")]))
    return rows, sources, dropped


def score(rows):
    totals = {}
    for arm in ARMS:
        by_subset = defaultdict(list)
        for r in rows:
            pred = np.asarray(r["selected"][arm])
            right = (pred == r["focus"]).all(-1)
            writable = all(r["words"][j] in r["gloss"][role] for role, j in enumerate(r["focus"]))
            group = "fresh" if r["book"] in {"hanfeizi", "zhuangzi"} else "previous"
            order = "fronted" if r["focus"][1] < r["focus"][0] else "postposed"
            for subset in (group, group + "/" + order, r["book"]):
                by_subset[subset].append((right, writable))
        totals[arm] = {}
        for subset, observations in by_subset.items():
            focus = np.stack([f for f, _ in observations])
            writable = np.asarray([w for _, w in observations])
            both = focus & writable[:, None]
            totals[arm][subset] = dict(pages=len(focus), writable=int(writable.sum()),
                pairs=both.sum(0).tolist(), classroom_focus=int(focus[:, :2].all(1).sum()),
                classroom=int(both[:, :2].all(1).sum()), ordered=int(both[:, 6:].all(1).sum()))
    return totals


@torch.inference_mode()
def main():
    torch.set_num_threads(3)
    face = Face(FACES[0])
    writer = restore().eval()
    existing, _ = rows_and_split()
    fresh, sources, exclusions = fresh_rows(face, existing)
    prior_rows = read(ORIGINAL / "predictions.json")
    rows = [dict(book=r["book"], page=r["page"], focus=r["focus"], gloss=r["gloss"],
                 words=r["all_words"]) for r in prior_rows] + fresh
    alphabet = {face.cell(c): c for c in "abcdefghijklmnopqrstuvwxyz-"}
    def ink(s):
        return np.frombuffer(face(s), np.uint8).copy().reshape(-1, 128)

    for r in fresh:
        r["words"] = []
        for j in range(len(r["page"])):
            raw, ended = respond(writer, ink(r["page"]), j)
            r["words"].append("".join(alphabet.get(x.tobytes(), "�") for x in raw) if ended else "<unfinished>")
    q, ql = packed_batch([s for pair in TEST for s in pair], face, "cpu")
    readers, queries = {}, {}
    for name in ARMS:
        path = WORK / name / "focus.safetensors"
        spec = read(path.parent / "fit.json")
        assert sha(path) == spec["sha256"]
        model = OrderedGlossFocus().eval()
        model.load_state_dict(load_file(str(path)))
        readers[name] = model
        queries[name] = model.read_questions(cell_features(q), ql)
    for a in range(0, len(rows), 32):
        batch = rows[a:a + 32]
        raw, lengths = packed_batch([r["page"] for r in batch], face, "cpu")
        ctx = encoding_windows(writer, cell_features(raw))
        for arm, model in readers.items():
            pos = model(ctx, lengths, queries[arm].expand(len(batch), -1, -1)).argmax(-1)
            for r, ix in zip(batch, pos.reshape(len(batch), -1, 2).tolist(), strict=True):
                r.setdefault("selected", {})[arm] = ix
    totals = score(rows)
    old = read(ORIGINAL / "paraphrases.json")["counts"]["combined"]
    assert totals[ARMS[0]]["previous"]["pairs"][2:6] == old["both"][:4]
    replay = []
    for j, r in enumerate(fresh):
        for k, query in enumerate(TEST[0] + TEST[1]):
            page = ink(r["page"])
            out, end, i = answer(writer, readers["everyday"], page, ink(query))
            word = "".join(alphabet.get(x.tobytes(), "�") for x in out) if end else "<unfinished>"
            assert i == r["selected"]["everyday"][k//2][k%2] and word == r["words"][i]
            name = f"page{j:02}-q{k}"
            folder = DEST / "samples"
            folder.mkdir(exist_ok=True)
            if k == 0:
                write_strip(page, folder / f"page{j:02}.png")
            write_strip(out, folder / (name + ".png"))
            replay.append(dict(page=r["page"], query=query, word=word, position=i,
                               stopped=end, image=name, sha256=sha(folder / (name + ".png"))))
    dump(DEST / "replay.json", replay)
    dump(DEST / "positions.json", rows)
    dump(DEST / "result.json", dict(labels=LABELS, questions=TEST, totals=totals,
         sources=sources, exclusions=exclusions, rows_sha256=sha(DEST/"positions.json"),
         replay_sha256=sha(DEST/"replay.json")))
    for a in ARMS:
        print(a, {g:totals[a][g] for g in ("previous","fresh","fresh/fronted")})


if __name__ == "__main__":
    main()
