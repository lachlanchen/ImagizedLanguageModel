"""Compare two named readings on the same held passages for a local text model and the image writer."""

import argparse
import json
from time import monotonic

import numpy as np
import torch

from scripts.prepare_visual_word_senses import WORK, PUBLIC, BOOKS, SEED, key, save, sha
from scripts.evaluate_visual_word_senses import restore
from scripts.train_visual_word_senses import corpus, images
from ilm.visual_lm.visual_word_sense import retinal_features
from scripts.measure_writer_energy import snapshot, consumed
from scripts.train_span_choice import gpu_owner


def prepare():
    pairs = json.loads((PUBLIC / "pairs.json").read_text())
    roster = []
    for book in BOOKS:
        rows = json.loads((WORK / (book + ".json")).read_text())
        ranked = sorted(pairs[book], key=lambda p: key(repr((book, p))))
        seen = set()
        for i, j in ranked:
            a, b = rows[i], rows[j]
            if a[4] in seen:
                continue
            seen.add(a[4])
            for index in (i, j):
                r = rows[index]
                options = sorted(
                    [a[6], b[6]], key=lambda s: key(repr((book, index, s)))
                )
                roster.append(
                    dict(
                        book=book,
                        index=index,
                        head=r[4],
                        left=r[3],
                        right=r[5],
                        choices=options,
                        gold=options.index(r[6]),
                    )
                )
            if len(seen) == 32:
                break
        assert len(seen) == 32
    save(PUBLIC / "comparison_roster.json", roster)


def text(rows):
    from scripts.evaluate_role_question_binding import local_teacher

    dest = WORK / "qwen"
    dest.mkdir(exist_ok=True)
    journal = dest / "answers.jsonl"
    existing = (
        [json.loads(s) for s in journal.read_text().splitlines()]
        if journal.exists()
        else []
    )
    with local_teacher("qwen3:8b-q8_0", dest) as (request, weights, _):
        start = monotonic()
        energy = snapshot()
        for i, r in enumerate(rows):
            if i < len(existing):
                assert (
                    existing[i]["index"] == r["index"]
                    and existing[i]["book"] == r["book"]
                )
                continue
            a, b = r["choices"]
            prompt = f"古文片段：{r['left']}【{r['head']}】{r['right']}\n依照句中的用法，括號內這個字較接近哪個英文詞義？\nA. {a}\nB. {b}\n只回答 A 或 B。"
            try:
                result = request(
                    "請選出字在指定古文語境中的詞義。只回答 A 或 B。",
                    dict(role="user", content=prompt),
                    think=False,
                    options=dict(temperature=0, seed=SEED, num_ctx=2048, num_predict=8),
                )
            except ValueError as e:
                if str(e) != "Incomplete local response":
                    raise
                result = dict(text="", stop="incomplete")
            word = result["text"].strip().upper()
            choice = {"A": 0, "B": 1}.get(word) if result["stop"] == "stop" else None
            outcome = dict(book=r["book"], index=r["index"], raw=result, choice=choice)
            existing.append(outcome)
            with journal.open("a") as f:
                f.write(json.dumps(outcome, ensure_ascii=False) + "\n")
        duration = monotonic() - start
        cost = consumed(energy, snapshot())
    return dict(rows=existing, seconds=duration, energy=cost, weights=weights)


@torch.inference_mode()
def visual(rows):
    fit = json.loads((PUBLIC / "aligned_fit.json").read_text())
    net = restore(WORK / "aligned/model.safetensors", fit["weights"], "cuda")
    feat = retinal_features(images()[0])
    chars = json.loads((WORK / "cells.json").read_text())
    bank = np.load(WORK / "pixels.npy")[0]
    x = {book: corpus(book)[0] for book in BOOKS}
    ready = []
    for r in rows:
        choices = []
        for s in r["choices"]:
            bits = np.unpackbits(bank[[chars.index(c) for c in s]], axis=-1)[
                :, net.landmarks.cpu().numpy()
            ]
            choices.append([*list(bits @ (2 ** np.arange(bits.shape[1]))), net.stop])
        t = max(map(len, choices))
        target = torch.tensor(
            [a + [net.stop] * (t - len(a)) for a in choices], device="cuda"
        )
        mask = torch.tensor(
            [[j < len(a) for j in range(t)] for a in choices], device="cuda"
        )
        ready.append(
            (
                feat[torch.as_tensor(x[r["book"]][r["index"]], device="cuda").long()][
                    None
                ],
                target,
                mask,
            )
        )
    torch.cuda.synchronize()
    start = monotonic()
    energy = snapshot()
    out = []
    for r, (source, target, mask) in zip(rows, ready):
        scores = (
            net.teacher(net.encode(source).expand(2, -1), target)
            .log_softmax(-1)
            .gather(-1, target[:, :, None])[:, :, 0]
        )
        scores = scores.masked_fill(~mask, 0).sum(1).tolist()
        out.append(
            dict(
                book=r["book"],
                index=r["index"],
                log_likelihood=scores,
                choice=int(scores[1] > scores[0]),
            )
        )
    return dict(
        rows=out,
        seconds=monotonic() - start,
        energy=consumed(energy, snapshot()),
        weights=fit["weights"],
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", choices=("prepare", "visual", "text"))
    a = ap.parse_args().action
    if a == "prepare":
        prepare()
    else:
        rows = json.loads((PUBLIC / "comparison_roster.json").read_text())
        with gpu_owner():
            result = (visual if a == "visual" else text)(rows)
        result["roster"] = sha(PUBLIC / "comparison_roster.json")
        save(PUBLIC / ("choices_" + a + ".json"), result)
