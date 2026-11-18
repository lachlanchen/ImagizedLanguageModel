"""Build balanced questions where two grammatical choices select the fact and form."""

import argparse
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from ilm.visual_lm.entry_span_writer import alternatives, parse_card
from scripts.prepare_pictured_entry_answers import (
    STAGES,
    FIELDS,
    Rasterizer,
    save_sequences,
    write_json,
)
from scripts.train_pictured_entry_writer import DEFAULT, ROOT, assets, corpus, load, sha

DEST = ROOT / "data/pictured_entry_request_order_v1"
# The two status expressions in each pair have the same number of physical cells.
FRAMES = {
    "train_a": ("我已知", "我想问", "我已见", "我想看"),
    "train_b": ("已了解", "请说明", "已看过", "请展示"),
    "train_c": ("不必说", "请告诉", "不必写", "请写出"),
    "new": ("不用讲", "想弄清", "不用画", "要给我"),
}


# New wording includes previously unused clauses. The scorekeeper appends it only on development.
def make_question(head, field_pair, stage_pair, field, stage, frame, outer):
    known, wanted, seen, show = FRAMES[frame]
    relation = lambda x, y: x + "，" + y
    f = relation(*(s + (wanted if s == field else known) for s in field_pair))
    t = relation(*(s + (show if s == stage else seen) for s in stage_pair))
    if outer == 0:
        return "关于" + head + "：" + f + "；" + t + "。"
    return head + "这个字，" + t + "；" + f + "。"


def write_split(folder, split):
    rng = np.random.default_rng(41556 if split == "train" else 41557)
    data = corpus(split)
    meta = load(DEFAULT / split / "index.json")
    marks, literals = assets()
    raster = Rasterizer()
    catalog = {}
    orders = []
    rows = []
    aliases = []
    indices = []
    collision_groups = 0
    total_groups = 0

    def encode(s):
        return [catalog.setdefault(raster.tile(c), len(catalog)) for c in s]

    for card_id, card in enumerate(data["cards"]):
        records = parse_card(card, marks)
        for head_side in range(2):
            start = 8 * card_id + 4 * head_side
            four = meta[start : start + 4]
            assert len({r["head"] for r in four}) == 1
            form_pair = [STAGES[four[i]["stage"]][0] for i in (0, 1)]
            # Cross both orders for both lists. Each complete target occupies all four positions.
            for permutation in range(4):
                fi = FIELDS[::-1] if permutation & 1 else FIELDS
                st = form_pair[::-1] if permutation & 2 else form_pair
                frame = ("train_a", "train_b", "train_c")[int(rng.integers(3))]
                outer = int(rng.integers(2))
                orders.append((frame, outer, permutation))
                equivalents = []
                old_features = []
                for j, r in enumerate(four):
                    stage = STAGES[r["stage"]][0]
                    args = (r["head"], fi, st, r["field"], stage)
                    q = make_question(*args, frame, outer)
                    physical = [raster.tile(c) for c in q]
                    equivalents.append(Counter(physical))
                    indices.append(start + j)
                    rows.append(encode(q))
                    if split == "development":
                        novel = make_question(*args, "new", 1 - outer)
                        aliases.append(encode(novel))
                        old_features.append(
                            alternatives(records, b"".join(physical), literals).features
                        )
                assert all(hist == equivalents[0] for hist in equivalents)
                assert len({data["answers"][start + j] for j in range(4)}) == 4
                if old_features:
                    collision_groups += int(
                        all(np.array_equal(f, old_features[0]) for f in old_features)
                    )
                total_groups += 1
    folder.mkdir()
    save_sequences(folder / "questions.npz", rows)
    if aliases:
        save_sequences(folder / "unseen_wording.npz", aliases)
    np.save(folder / "target_row.npy", np.asarray(indices, np.int32))
    np.save(
        folder / "query_pixels.npy",
        np.frombuffer(b"".join(catalog), np.uint8).reshape(-1, 128),
    )
    write_json(folder / "layouts.json", orders)
    return dict(
        questions=len(rows),
        equal_histogram_quartets=total_groups,
        identical_old_features=collision_groups if split == "development" else None,
        further_wordings=len(aliases),
        files={p.name: sha(p) for p in sorted(folder.iterdir())},
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=DEST)
    out = p.parse_args().output.resolve()
    if out.exists():
        raise FileExistsError(out)
    with TemporaryDirectory(dir=out.parent) as tmp:
        root = Path(tmp) / "built"
        root.mkdir()
        report = {key: write_split(root / key, key) for key in ("train", "development")}
        write_json(
            root / "report.json",
            dict(
                splits=report,
                source_data_sha=sha(DEFAULT / "report.json"),
                source_sha=sha(Path(__file__)),
            ),
        )
        root.rename(out)
    print(
        {
            k: (
                v["questions"],
                v["equal_histogram_quartets"],
                v["identical_old_features"],
            )
            for k, v in report.items()
        }
    )
