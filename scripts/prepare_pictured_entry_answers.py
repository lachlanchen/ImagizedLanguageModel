"""Build pictured requests for a source-listed fact and an attested historical form."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
from itertools import combinations
import json
from pathlib import Path
import random
import re
from tempfile import TemporaryDirectory

import numpy as np
import torch

from scripts.prepare_source_plate_questions import CACHE, SPLITS, STAGES, Rasterizer, load, sha

DEFAULT = SPLITS.parents[1] / "data/pictured_entry_answers_v1"
FIELDS = ("本义", "读音")


def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def save_sequences(path, seqs):
    lengths = np.array([len(s) for s in seqs], dtype=np.int64)
    np.savez_compressed(path, offsets=np.r_[0, lengths.cumsum()],
                        cells=np.asarray([v for s in seqs for v in s], dtype=np.int32))


def fact_values(entry):
    fields = entry["fields"]
    meaning = re.fullmatch(r"Meaning\s+([\w\s,'()/-]+)\.?", fields.get("original_meaning", "").strip())
    if not meaning:
        return
    gloss = " ".join(meaning[1].split())
    sound = fields.get("main_pronunciation", "").strip()
    if 2 <= len(gloss) <= 45 and gloss.isascii() and re.search("[A-Za-z]{2}", gloss) and re.fullmatch(r"[^\W\d_]{1,12}", sound):
        if gloss.lower() not in {"not applicable", "unknown", "not exist"}:
            return gloss, sound


def prepare_split(folder, split, renderer, packed):
    rng = random.Random(88613 if split == "train" else 88614)
    entries = load(SPLITS / (split + ".json"))
    rng.shuffle(entries)
    groups, selected = defaultdict(list), set()
    for entry in entries:
        family = entry["family_sha256"]
        facts = fact_values(entry)
        if family in selected or not facts:
            continue
        try:
            for ch in entry["char"] + "".join(facts):
                renderer.tile(ch)
        except ValueError:
            continue
        stages = defaultdict(list)
        for glyph in entry["glyphs"]:
            stages[glyph["stage"]].append(glyph)
        choices = list(combinations(sorted(stages), 2))
        rng.shuffle(choices)
        for pair in choices:
            glyphs = [rng.choice(stages[stage]) for stage in pair]
            if glyphs[0]["raster_sha256"] == glyphs[1]["raster_sha256"]:
                continue
            groups[pair].append((entry, facts, glyphs))
            selected.add(family)
            break
    bank = {}
    def cell(bitmap):
        return bank.setdefault(bitmap, len(bank))
    def render(s):
        return [cell(renderer.tile(c)) for c in s]

    cards, questions, paraphrases, answers, metadata = [], [], [], [], []
    retained = defaultdict(set)
    for stage_pair, items in sorted(groups.items()):
        rng.shuffle(items)
        while len(items) > 1:
            first = items.pop()
            partner = next((i for i, second in enumerate(items)
                            if all(x != y for x, y in zip(first[1], second[1]))
                            and len({g["raster_sha256"] for record in (first, second) for g in record[2]}) == 4), None)
            if partner is None:
                continue
            records = [first, items.pop(partner)]
            card = []
            payloads = []
            for entry, facts, glyphs in records:
                head = entry["char"]
                retained["family"].add(entry["family_sha256"])
                retained["entry"].add(entry["entry_id"])
                fields = [render(label + "：" + value) for label, value in zip(FIELDS, facts)]
                historical = []
                for stage, g in zip(stage_pair, glyphs):
                    raw = packed[g["raster_index"]]
                    if sha256(raw.tobytes()).hexdigest() != g["raster_sha256"]:
                        raise ValueError("Historical raster digest changed")
                    retained["raster"].add(g["raster_sha256"])
                    ink = cell(np.bitwise_not(raw).tobytes())  # cache uses white=1
                    historical.append(ink)
                    fields.append(render(STAGES[stage][0] + "：") + [ink])
                rng.shuffle(fields)
                card += render("【" + head + "】")
                for field in fields:
                    card += field + render("；")
                payloads.append(historical)
            idx = len(cards)
            cards.append(card)
            for j, (entry, facts, glyphs) in enumerate(records):
                for f, label in enumerate(FIELDS):
                    for k, stage in enumerate(stage_pair):
                        script = STAGES[stage][0]
                        head = entry["char"]
                        questions.append(render(f"请写出{head}的{label}和{script}字形。"))
                        if split == "development":
                            paraphrases.append(render(f"查{head}：它的{label}是什么？并列出{script}。"))
                        answers.append(render(head + "的" + label + "是" + facts[f] + "，" + script + "作")
                                       + [payloads[j][k]] + render("。"))
                        metadata.append(dict(card=idx, entry=entry["entry_id"], head=head,
                                             family=entry["family_sha256"], field=label, stage=stage,
                                             value=facts[f], glyph=glyphs[k]["label"], raster=glyphs[k]["raster_sha256"]))
    if not cards or len(set(map(tuple, answers))) != len(answers):
        raise ValueError("Empty split or repeated complete answer")
    folder.mkdir()
    np.save(folder / "pixels.npy", np.frombuffer(b"".join(bank), np.uint8).reshape(-1, 128))
    for name, seqs in (("cards", cards), ("questions", questions), ("answers", answers)):
        save_sequences(folder / (name + ".npz"), seqs)
    if paraphrases:
        save_sequences(folder / "question_paraphrases.npz", paraphrases)
    write_json(folder / "index.json", metadata)
    return dict(cards=len(cards), questions=len(questions), families=len(retained["family"]),
                field_counts=dict(Counter(r["field"] for r in metadata)),
                historical_forms=len(retained["raster"]), max_answer_cells=max(map(len,answers)),
                files={p.name:sha(p) for p in sorted(folder.iterdir())}), retained


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT)
    out = parser.parse_args().output.resolve()
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(exist_ok=True, parents=True)
    info = load(SPLITS / "report.json")
    if sha(CACHE) != info["raster_cache"]["cache_sha256"]:
        raise ValueError("Unknown historical raster source")
    torch.set_num_threads(2)
    cache = torch.load(CACHE, weights_only=True, map_location="cpu")["pixels"].numpy()
    packed = np.packbits(cache.reshape(len(cache), -1), axis=1)
    report, groups = {}, {}
    with TemporaryDirectory(dir=out.parent) as temp:
        root = Path(temp) / "prepared"
        root.mkdir()
        renderer = Rasterizer()
        for split in ("train", "development"):
            path = SPLITS / (split + ".json")
            if sha(path) != info["partitions"][split]["file_sha256"]:
                raise ValueError("Unknown source manifest")
            report[split], groups[split] = prepare_split(root / split, split, renderer, packed)
        overlap = {k:len(v & groups["development"][k]) for k,v in groups["train"].items()}
        if any(overlap.values()):
            raise ValueError("Source overlap across partitions")
        evidence = dict(splits=report, train_dev_overlap=overlap,
                        source_report_sha=sha(SPLITS / "report.json"), cache_sha=sha(CACHE),
                        source_code_sha=sha(Path(__file__)), uses_sealed_partition=False)
        write_json(root / "report.json", evidence)
        root.rename(out)
    print({k:(v["cards"],v["questions"]) for k,v in report.items()}, overlap)


if __name__ == "__main__":
    main()
