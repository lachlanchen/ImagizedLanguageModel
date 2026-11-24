"""Compare the passage learner with count backoff on the identical fitting windows."""

from collections import Counter, defaultdict
from time import monotonic
import gzip
import hashlib
import json
import numpy as np
import torch
from ilm.visual_lm.portable_writer import read_image
from ilm.visual_lm.causal_glyph_writer import write_strip
from ilm.visual_lm.visual_gap import fill
from scripts.answer_dialogue_surface import load, answer
from scripts.dialogue_ink_data import read_data
from scripts.evaluate_compose_dialogue import confidence
from scripts.evaluate_optical_passage_qa import metrics
from scripts.prepare_context_cloze import PUB, MODEL, WORK, reserved
from scripts.prepare_compose_dialogue import prepared, kind
from scripts.prepare_dialogue_ink import dump, sha
from scripts.train_context_cloze import SEED, STEPS
from scripts.train_dialogue_surface import representation


def read_outputs(name, checksum):
    file = PUB / "responses" / (name + ".json.gz")
    if sha(file) != checksum:
        raise ValueError("Different predictions")
    return json.loads(gzip.decompress(file.read_bytes()))


def keys(words, t):
    a, b, c, d = (
        words[i] if 0 <= i < len(words) else None for i in (t - 2, t - 1, t + 1, t + 2)
    )
    return (a, b, c, d), (b, c), (a, b), b, c


def run():
    clock = monotonic()
    torch.set_num_threads(2)
    data = reserved()
    pages = [data["pages"][pid].split() for pid in sorted(data["pages"])]
    result = json.loads((PUB / "results.json").read_text())["models"]
    rec = result["ink-passage"]
    meta = json.loads((MODEL / "ink/passage/weights.json").read_text())
    if rec["weights"] != meta["sha256"]:
        raise ValueError("Changed checkpoint")
    published = read_outputs("ink-passage-page", rec["pages"]["outputs"])
    reg = read_data()
    extra = prepared()["extra"]
    train_pages = reg["pages"] | extra["pages"]
    used = sorted({r[1] for r in reg["rows"] + extra["rows"] if r[0] == "train"})
    if set(used) & set(data["pages"]):
        raise ValueError("Page overlap")
    train = [train_pages[k].split() for k in used]
    net = load(MODEL / "ink/passage")
    face, unit, sep = representation("ink")
    letters = {face(chr(i)): chr(i) for i in range(32, 127)}
    if len(letters) != 95:
        raise ValueError("Distinct printable cells required for count comparator")
    vocabulary = [
        "".join(letters[w[i : i + unit]] for i in range(0, len(w), unit))
        for w in net.prototype_words
    ]
    inv = {w: i for i, w in enumerate(vocabulary)}
    coord = [
        (i, t) for i, page in enumerate(train) for t, w in enumerate(page) if w in inv
    ]
    if len(coord) != meta["eligible_page_centres"]:
        raise ValueError("Different eligible centres")
    draws = np.random.default_rng(SEED + 1).integers(len(coord), size=STEPS * 64)
    if (
        hashlib.sha256(draws.astype("<i8").tobytes()).hexdigest()
        != meta["page_schedule"]
    ):
        raise ValueError("Different sampled centres")
    tables = [defaultdict(Counter) for _ in range(5)]
    freq = Counter()
    for j in draws:
        i, t = coord[j]
        w = inv[train[i][t]]
        freq[w] += 1
        for table, k in zip(tables, keys(train[i], t), strict=True):
            table[k][w] += 1

    def mode(c):
        return min(c, key=lambda w: (-c[w], w))

    compact = [
        {k: (sum(c.values()), mode(c)) for k, c in table.items()} for table in tables
    ]
    common = mode(freq)

    def predict(context, hierarchy):
        for j in hierarchy:
            count, leader = compact[j].get(context[j], (0, 0))
            if count >= (2 if j < 3 else 1):
                return leader
        return common

    qa_stats = result["ink-qa"]["pages"]
    qa_rows = read_outputs("ink-qa-page", qa_stats["outputs"])
    groups = {"all": Counter(), "unseen": Counter()}
    gained = []
    examples = {}
    raw_strips = {}
    ids = []
    for row, q in zip(published, qa_rows, strict=True):
        page, t, expected, pred, swap, left = row
        if q[:3] != row[:3] or expected != inv.get(pages[page][t], -1):
            raise ValueError("Wrong masked page target")
        ks = keys(pages[page], t)
        novel = ks[0] not in compact[0]
        back = predict(ks, (0, 1, 3, 4))
        prev = predict(ks, (2, 3))
        gained.append(int(pred == expected) - int(back == expected))
        ids.append([None, page])
        for g in ["all"] + (["unseen"] if novel else []):
            groups[g].update(
                n=1,
                neural=pred == expected,
                backoff=back == expected,
                left_neural=left == expected,
                left_backoff=prev == expected,
                only_neural=pred == expected and back != expected,
                only_backoff=back == expected and pred != expected,
                left_only_neural=left == expected and prev != expected,
                left_only_backoff=prev == expected and left != expected,
                exchanged=swap == expected,
                unigram=common == expected,
            )
        if (
            novel
            and pred == expected
            and swap != expected
            and q[3] != expected
            and back != expected
            and len(pages[page][t]) >= 4
            and "gap" not in examples
        ):
            examples["gap"] = dict(
                story=page,
                position=t,
                left=pages[page][t - 2 : t],
                right=pages[page][t + 1 : t + 3],
                prediction=vocabulary[pred],
                reference=pages[page][t],
                backoff=vocabulary[back],
                exchanged=vocabulary[swap],
            )
            raw_strips.update(
                left=examples["gap"]["left"], right=examples["gap"]["right"]
            )
    if groups["all"]["neural"] != rec["pages"]["local"]:
        raise ValueError("Wrong accuracy")

    def strip(words):
        return np.frombuffer(sep.join(face(w) for w in words), np.uint8).reshape(
            -1, unit
        )

    def save(name, raw):
        f = PUB / (name + ".png")
        write_strip(np.frombuffer(raw, np.uint8).reshape(-1, unit), f)
        return sha(f)

    if raw_strips:
        x = {}
        for name, ws in raw_strips.items():
            f = PUB / (name + ".png")
            write_strip(strip(ws), f)
            x[name] = read_image(f, None)
        out = fill(net, x["left"], x["right"])
        if out != face(examples["gap"]["reference"]):
            raise ValueError("Standalone gap replay differs")
        examples["gap"]["png_sha256"] = save("gap", out)
    trials = read_outputs("ink-passage", rec["outputs"])
    for i, r in enumerate(data["rows"]):
        y = trials[i]
        difficult = kind(r[5], (data["pages"][r[1]], r[3], r[4])) != "ordinary"

        def correct(k):
            return y[k]["stop"] and bool(metrics(y[k]["answer"], [r[5]])[0])

        for name, select in (
            (
                "answer",
                difficult
                and correct("original")
                and not correct("question")
                and not correct("passage"),
            ),
            ("miss", difficult and not correct("original")),
        ):
            if not select or name in examples:
                continue
            fields = []
            for j, text in enumerate((data["pages"][r[1]], r[3], r[4])):
                image = strip(text.split())
                if len(image):
                    f = WORK / f"{name}-{j}.png"
                    write_strip(image, f)
                    image = read_image(f, None)
                fields.append(image)
            raw, info = answer(net, fields)
            text = y["original"]["answer"]
            if (
                raw != sep.join(face(w) for w in text.split())
                or info["stopped"] != y["original"]["stop"]
            ):
                raise ValueError("Standalone QA replay differs")
            examples[name] = dict(
                row=i,
                question=r[3],
                reference=r[5],
                prediction=text,
                exchanged_question=y["question"]["answer"],
                exchanged_passage=y["passage"]["answer"],
                png_sha256=save(name, raw),
            )
    dump(
        PUB / "backoff.json",
        dict(
            groups=groups,
            training_centres=len(draws),
            prior_mode=common,
            paired=confidence(np.array(gained), ids, np.arange(len(ids))),
            combined_cpu_audit_seconds=monotonic() - clock,
        ),
    )
    dump(PUB / "examples.json", examples)
    print(groups, examples, flush=True)


if __name__ == "__main__":
    run()
