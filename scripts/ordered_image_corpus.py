"""Pictured compound questions and their complete answer images."""
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from scripts.audit_pictured_entry_answers import sequences
from scripts.fit_font_invariant_reader import Face, FONTS, ROOT, sha
from scripts.read_paired_passage_questions import PairedPassages

STAGES = {"oracle":"甲骨文", "bronze":"金文", "seal":"篆書", "liushutong":"六書通"}
PLATES = ROOT / "data/source_plates_v1"
PAIRS = ROOT / "data/paired_passage_questions_v1"
OPEN = ROOT / "data/composed_chinese_answers_v1"


@dataclass
class Example:
    task: str
    case: str
    passage: bytes
    question: bytes
    target: bytes


def plate_examples(rows, pixels, render, card, variant, held, task):
    names = sorted({r["char"] for r in rows})
    stages = sorted({r["stage"] for r in rows}, key=list(STAGES).index)
    glyph = {(r["char"], r["stage"]): bytes(pixels[r["tile"]]) for r in rows}
    entries = []
    for k, name in enumerate(names[::1 if variant % 2 == 0 else -1]):
        a, b = stages[::1 if (card + k + variant // 2) % 2 == 0 else -1]
        entries.append(render(f"「{name}」的{STAGES[a]}寫作") + glyph[name,a] +
                       render(f"，{STAGES[b]}寫作") + glyph[name,b] + render("。"))
    passage = b"".join(entries)
    for name in names:
        for order, wanted in enumerate(([stages[0]], [stages[1]], stages, stages[::-1])):
            labels = "、".join(STAGES[s] for s in wanted)
            if held:
                question = f"我在找{name}。請依{labels}的排列把字形抄給我。"
            elif len(wanted) == 1:
                question = f"請寫出「{name}」的{labels}字形。"
            else:
                question = f"請按{labels}的順序，只寫出「{name}」的兩個字形。"
            target = b"".join(glyph[name,s] for s in wanted)
            yield Example(task, f"{card}/{variant}/{name}/{order}", passage, render(question), target)


def corpus(split):
    train = split == "train"
    rng = np.random.default_rng(6749 if train else 6750)
    render = Face((FONTS[0],2))
    report = json.loads((PLATES/"report.json").read_text())
    assert not any(report["overlap"].values())
    cards_file = PLATES / (split+"_cards.json")
    assert sha(cards_file) == report["splits"][split]["cards_sha256"]
    assert sha(PLATES/"pixels.npy") == report["pixels_sha256"]
    cards = json.loads(cards_file.read_text())
    pixels = np.load(PLATES/"pixels.npy")
    selected = rng.permutation(len(cards))[:768 if train else 24]
    examples = []
    for number in selected:
        variants = [int(rng.integers(4))] if train else [0,3]
        for v in variants:
            for held in ([False] if train else [False,True]):
                examples.extend(plate_examples(cards[number], pixels, render, int(number), v, held,
                                                "plate_reworded" if held else "plate"))
    paired = PairedPassages(PAIRS, split)
    # Sample unordered pairs, always keeping both orders and identical evidence together.
    pairs = paired.pairs.reshape(-1,2,3)
    assert np.array_equal(pairs[:,0], pairs[:,1,[0,2,1]])
    seen = 0
    for index in rng.permutation(len(pairs)):
        both = []
        for row in (2*index,2*index+1):
            passage, query = paired.inputs(row)
            if (len(passage)+len(query))//128 > 509:
                break
            both.append(Example("paired", str(row), passage, query, paired.target(row)))
        if len(both)==2:
            examples.extend(both)
            seen += 1
        if seen == (1536 if train else 128):
            break
    assert seen == (1536 if train else 128)
    folder = OPEN/split
    recorded = json.loads((OPEN/"preparation.json").read_text())[split]
    for name, digest in recorded["files"].items():
        if sha(folder/name) != digest:
            raise ValueError("Changed DuReader file "+name)
    tiles = np.load(folder/"pixels.npy")
    p,q,y = [sequences(folder, name, tiles) for name in ("passages","questions","answers")]
    examples.extend(Example("open",str(i),a,b,c) for i,(a,b,c) in enumerate(zip(p,q,y,strict=True)))
    if not train:
        from scripts.pictured_participant_scenes import fresh_cases
        for i,(p,q,y) in enumerate(fresh_cases()):
            examples.append(Example("single",str(i),render(p),render(q),render(y)))
        # Match the previously published 言/征 plate exactly, including stage and headword order.
        rows = json.loads((PLATES/"yan/source.json").read_text())["rows"]
        glyph = {(r["char"],r["stage"]):bytes(pixels[r["tile"]]) for r in rows}
        for v, names in enumerate(("言征","征言")):
            source = b"".join(render(f"「{c}」的甲骨文寫作")+glyph[c,"oracle"]+
                              render("，金文寫作")+glyph[c,"bronze"]+render("。") for c in names)
            for c in names:
                for k,wanted in enumerate((["oracle"],["bronze"],["oracle","bronze"],["bronze","oracle"])):
                    labels="、".join(STAGES[s] for s in wanted)
                    query=(f"請寫出「{c}」的{labels}字形。" if len(wanted)==1 else
                           f"請按{labels}的順序，只寫出「{c}」的兩個字形。")
                    examples.append(Example("yan",f"{v}/{c}/{k}",source,render(query),
                                             b"".join(glyph[c,s] for s in wanted)))
    return examples
