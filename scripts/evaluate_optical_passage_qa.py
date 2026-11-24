"""Score held answer images, paired question exchange and uncached PNG replay."""

from collections import Counter, defaultdict
import argparse
from pathlib import Path
import gzip
import json
import re
import string
import numpy as np
from PIL import Image
import torch

from ilm.visual_lm.causal_glyph_writer import write_strip
from ilm.visual_lm.portable_writer import read_image
from scripts.answer_optical_passage_qa import RELEASE, answer, load
from scripts.prepare_optical_passage_qa import WORK, dump
from scripts.train_optical_passage_qa import VisualCorpus, corpus
from scripts.train_typeface_writer import Face, FACES
from scripts.train_span_choice import gpu_owner


def normalized(s):
    s = s.lower().translate(str.maketrans("", "", string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", s).split())


def metrics(s, answers):
    tokens = normalized(s).split()
    best = 0.0
    exact = False
    for a in answers:
        ref = normalized(a).split()
        exact |= tokens == ref
        common = sum((Counter(tokens) & Counter(ref)).values())
        f1 = 2 * common / (len(tokens) + len(ref)) if common else 0.0
        best = max(best, f1)
    return exact, best


def offsets(text, decision):
    words = list(re.finditer(r"\S+", text))
    l, r, dl, dr = decision
    return words[l].start() + dl, words[r].end() - dr


def save_page(text, path, face):
    packed = np.frombuffer(face(text), np.uint8).reshape(-1, 128)
    cols = min(48, len(packed))
    grid = np.zeros(((len(packed) + cols - 1) // cols * cols, 128), np.uint8)
    grid[: len(packed)] = packed
    pixels = (
        np.unpackbits(grid, axis=1)
        .reshape(-1, cols, 32, 32)
        .transpose(0, 2, 1, 3)
        .reshape(-1, cols * 32)
    )
    Image.fromarray(255 * (1 - pixels)).save(path)


def paired(rows):
    by_page = defaultdict(list)
    for row in rows:
        by_page[row["page"]].append(row)
    choices = []
    for group in by_page.values():
        first = group[0]
        refs = {normalized(s) for s in first["answers"]}
        second = next(
            (
                r
                for r in group[1:]
                if r["question"] != first["question"]
                and refs.isdisjoint(map(normalized, r["answers"]))
            ),
            None,
        )
        if second is None:
            continue
        actual = first["exact"] and second["exact"]
        exchanged = (
            metrics(first["prediction"], second["answers"])[0]
            and metrics(second["prediction"], first["answers"])[0]
        )
        choices.append((first, second, actual, exchanged))
    return choices


def grouped_summary(records):
    pairs = paired(records)
    return dict(
        queries=len(records),
        exact=sum(r["exact"] for r in records),
        mean_f1=sum(r["f1"] for r in records) / len(records),
        exact_pixels=sum(r["pixels"] for r in records),
        paired_total=len(pairs),
        both_exact=sum(x[2] for x in pairs),
        both_exchanged=sum(x[3] for x in pairs),
        changed_image=sum(x[0]["prediction"] != x[1]["prediction"] for x in pairs),
    )


@torch.inference_mode()
def evaluate(folder, output):
    data = corpus()
    net, encoder = load("cuda", folder)
    face = Face(FACES[0])
    groups = {}
    evidence = []
    for group in ("reserved", "dev"):
        rows = [r for r in data["rows"] if r[0] == group]
        bank = VisualCorpus(net, encoder, data["pages"], rows, "cuda")
        decisions = []
        for i in range(0, len(rows), 48):
            x, widths, _ = bank.batch(slice(i, i + 48))
            scores, trim = net(*x)
            decisions.extend(net.select(scores, trim, x[1], widths).tolist())
        records = []
        for (_, title, key, qid, q, _, refs), decision in zip(
            rows, decisions, strict=True
        ):
            text = data["pages"][key]
            a, b = offsets(text, decision)
            pred = text[a:b]
            exact, f1 = metrics(pred, refs)
            physical = any(face(pred) == face(" ".join(ref.split())) for ref in refs)
            records.append(
                dict(
                    id=qid,
                    group=group,
                    title=title,
                    page=key,
                    question=q,
                    decision=decision,
                    bounds=[a, b],
                    prediction=pred,
                    answers=refs,
                    exact=exact,
                    f1=f1,
                    pixels=physical,
                )
            )
        groups[group] = grouped_summary(records)
        evidence.extend(records)
    with (output / "decisions.json.gz").open("wb") as f:
        f.write(
            gzip.compress(
                json.dumps(
                    evidence, ensure_ascii=False, separators=(",", ":")
                ).encode(),
                mtime=0,
            )
        )
    return data, evidence, groups


def replay(data, records, groups, model, output):
    net, encoder = load("cpu", model)
    folder = output / "examples"
    folder.mkdir(exist_ok=True)
    face = Face(FACES[0])
    checks = []
    for group in groups:
        pairs = paired([r for r in records if r["group"] == group])
        for passed in (True, False):
            example = next((p for p in pairs if p[2] == passed), None)
            if example is None:
                continue
            name = group + ("-correct" if passed else "-miss")
            a, b = example[:2]
            page = folder / (name + "-page.png")
            save_page(data["pages"][a["page"]], page, face)
            for j, r in enumerate((a, b), 1):
                prefix = name + "-" + str(j)
                qpath = folder / (prefix + "-question.png")
                save_page(r["question"], qpath, face)
                image, bounds = answer(
                    net, encoder, read_image(page, None), read_image(qpath, None)
                )
                if list(bounds) != r["bounds"] or image.tobytes() != face(
                    r["prediction"]
                ):
                    raise ValueError(
                        (
                            "Independent PNG replay mismatch",
                            r["id"],
                            bounds,
                            r["bounds"],
                        )
                    )
                write_strip(image, folder / (prefix + "-answer.png"))
                checks.append(
                    dict(
                        id=r["id"],
                        prefix=prefix,
                        title=r["title"],
                        question=r["question"],
                        predicted=r["prediction"],
                        accepted=r["answers"],
                        exact=r["exact"],
                    )
                )
    return checks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=RELEASE)
    parser.add_argument("--output", type=Path, default=WORK / "evaluation")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    with gpu_owner():
        torch.set_num_threads(2)
        data, records, groups = evaluate(args.model, args.output)
    checks = replay(data, records, groups, args.model, args.output)
    dump(args.output / "result.json", dict(groups=groups, png_replays=checks))
    print(groups)
