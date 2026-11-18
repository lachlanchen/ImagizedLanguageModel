"""Prepare both orders of human-written questions and their two-part pictured answers."""

from collections import defaultdict
from itertools import combinations
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic

import numpy as np

from scripts.prepare_pictured_entry_answers import (
    Rasterizer,
    save_sequences,
    sha,
    write_json,
    load,
)
from scripts.train_visual_extractive_reader import load_cases, ROOT, DATA, MANIFEST


DEST = ROOT / "data/paired_passage_questions_v1"
BASE = ROOT / "artifacts/mixed_ink_reader_v1"


def contains(whole, part):
    at = whole.find(part)
    while at != -1:
        if at % 128 == 0:
            return True
        at = whole.find(part, at + 1)
    return False


def encode_split(folder, cases, replay=None):
    folder.mkdir()
    bank = {}
    groups = defaultdict(list)

    def encode(image):
        return [bank.setdefault(bytes(tile), len(bank)) for tile in image]

    passages = []
    questions = []
    answers = []
    origin = []
    for i, c in enumerate(cases):
        if c.group not in groups:
            passages.append(encode(c.passage))
        groups[c.group].append(i)
        origin.append(c.label)
        questions.append(encode(c.question))
        answers.append(encode(c.answer))
    targets = []
    single_span = 0
    segmented = 0
    render = Rasterizer()
    prefix = "请依次回答这两个问题："
    sep = "；"
    instructions = [
        encode(
            np.frombuffer(b"".join(render.tile(c) for c in s), np.uint8).reshape(
                -1, 128
            )
        )
        for s in (prefix, sep)
    ]
    for g, indices in enumerate(groups.values()):
        passage = cases[indices[0]].passage.tobytes()
        for i, j in combinations(indices, 2):
            a, b = cases[i].answer.tobytes(), cases[j].answer.tobytes()
            if questions[i] == questions[j] or contains(a, b) or contains(b, a):
                continue
            # A pair is useful only if exchanging the order changes the required physical result.
            join = render.tile(sep)
            if a + join + b == b + join + a:
                continue
            for l, r in ((i, j), (j, i)):
                target = cases[l].answer.tobytes() + join + cases[r].answer.tobytes()
                single_span += int(contains(passage, target))
                if replay is not None:
                    predicted = replay[l][0] + join + replay[r][0]
                    segmented += int(
                        replay[l][1] and replay[r][1] and predicted == target
                    )
                targets.append((g, l, r))
    for name, seqs in (
        ("passages", passages),
        ("questions", questions),
        ("answers", answers),
        ("instruction_cells", instructions),
    ):
        save_sequences(folder / (name + ".npz"), seqs)
    np.save(
        folder / "pixels.npy", np.frombuffer(b"".join(bank), np.uint8).reshape(-1, 128)
    )
    np.save(folder / "pairs.npy", np.asarray(targets, np.int32))
    write_json(folder / "source_ids.json", origin)
    write_json(folder / "passage_groups.json", list(groups))
    return dict(
        passages=len(passages),
        original_questions=len(cases),
        paired_requests=len(targets),
        single_source_span_exact_ceiling=single_span,
        supplied_split_and_composer_exact=segmented if replay is not None else None,
        files={p.name: sha(p) for p in sorted(folder.iterdir())},
    )


def prepare():
    if DEST.exists():
        raise FileExistsError(DEST)
    train, dev = load_cases(monotonic())
    assert len(set(c.group for c in train) & set(c.group for c in dev)) == 0
    src = BASE / "normal.npz"
    assert sha(src) == load(BASE / "report.json")["output_sha256"]["normal.npz"]
    with np.load(src) as z:
        pixels, off, stops, exact = (
            z[k] for k in ("pixels", "offsets", "stopped", "exact")
        )
    replay = [
        (pixels[a:b].tobytes(), s)
        for a, b, s in zip(off[:-1], off[1:], stops, strict=True)
    ]
    assert [
        s and p == c.answer.tobytes() for (p, s), c in zip(replay, dev, strict=True)
    ] == exact.tolist()
    with TemporaryDirectory(dir=DEST.parent) as tmp:
        stage = Path(tmp) / "ready"
        stage.mkdir()
        report = dict(
            train=encode_split(stage / "train", train),
            development=encode_split(stage / "development", dev, replay),
            source_sha=sha(DATA / "report.json"),
            passage_manifest_sha=sha(MANIFEST),
            archive_sha=sha(src),
            single_reader_exact=int(exact.sum()),
            code_sha=sha(Path(__file__)),
        )
        write_json(stage / "report.json", report)
        stage.rename(DEST)
    print(
        {
            k: {a: b for a, b in report[k].items() if a != "files"}
            for k in ("train", "development")
        }
    )


if __name__ == "__main__":
    prepare()
