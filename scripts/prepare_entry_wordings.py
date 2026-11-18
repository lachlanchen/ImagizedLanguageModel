"""Render training-only questions with varying clause scope, synonyms and cue position."""

from pathlib import Path
import numpy as np

from scripts.prepare_pictured_entry_answers import (
    DEFAULT,
    FIELDS,
    STAGES,
    Rasterizer,
    load,
    write_json,
    save_sequences,
    sha,
)
from scripts.prepare_entry_request_order import DEST as ORDER_DATA
from scripts.check_entry_question_order import reordered_question
from scripts.train_pictured_entry_writer import ROOT

DEST = ROOT / "data/pictured_entry_wordings_v1"
FIELD_WORDS = (
    ("本义", "本来的意思", "起初的含义", "最早的意思", "原来的意义"),
    ("读音", "普通话发音", "现在的念法", "今天的读法", "现代的读音"),
)
STAGE_WORDS = {
    "oracle": ("甲骨文", "甲骨文字", "甲骨上的写法"),
    "bronze": ("金文", "青铜器上的写法", "金文字形"),
    "seal": ("篆書", "篆书", "小篆"),
    "liushutong": ("六書通", "六书通", "《六书通》"),
}
# Every frame specifies a requested fact and form. Distractors have explicit known/rejected roles.
FORMS = (
    "我想查{h}的{f}，还要看{s}的字形。",
    "能给我看看{h}在{s}里怎么写，再说说它的{f}吗？",
    "{h}的{n}我知道了，{f}呢？{u}我也见过了，给我看看{s}。",
    "我已经知道{h}的{n}，也看过{u}。麻烦补上{f}和{s}的写法。",
    "{h}的资料里我只缺两项：{s}的字形和{f}。{n}和{u}的字形都有了。",
    "关于{h}，先告诉我{f}，再把{s}写给我。{n}和{u}就跳过吧。",
    "不用介绍{n}或者展示{u}，请给出{h}的{f}，并附上{s}字形。",
    "查{h}。字形要{s}，不用{u}；文字说明写{f}，不必写{n}。",
    "我手上的笔记只有{n}和{u}的写法，{h}的{f}和{s}字形应该怎么补？",
    "请写出{s}里的{h}，再把它的{f}写在旁边。",
    "就{h}来说，{f}是什么？还想看它的{s}。",
    "{u}那一种已经有了。请给我看{h}的{s}，另外还要知道{f}。",
)


def prepare(destination=DEST):
    destination.mkdir(exist_ok=False)
    rows = load(DEFAULT / "train/index.json")
    orders = load(ORDER_DATA / "train/layouts.json")
    order_target = np.load(ORDER_DATA / "train/target_row.npy")
    rng = np.random.default_rng(373921)
    renderer = Rasterizer()
    tile_ids = {}
    sequences = []
    target = []

    def add(index, question):
        glyphs = [
            tile_ids.setdefault(renderer.tile(c), len(tile_ids)) for c in question
        ]
        sequences.append(glyphs)
        target.append(index)

    for start in range(0, len(rows), 4):
        a, b = rows[start]["stage"], rows[start + 1]["stage"]
        subset = rng.choice(len(FORMS), 4, replace=False)
        for template in subset:
            words = [rng.choice(w) for w in FIELD_WORDS]
            images = {s: rng.choice(STAGE_WORDS[s]) for s in (a, b)}
            for j, r in enumerate(rows[start : start + 4]):
                k = FIELDS.index(r["field"])
                s = r["stage"]
                add(
                    start + j,
                    FORMS[template].format(
                        h=r["head"],
                        f=words[k],
                        n=words[1 - k],
                        s=images[s],
                        u=images[b if s == a else a],
                    ),
                )
    for group, (frame, outer, perm) in enumerate(orders):
        if perm != group // 4 % 4:
            continue
        i = 4 * group
        indices = order_target[i : i + 4]
        fields = FIELDS[::-1] if perm & 1 else FIELDS
        stages = [STAGES[rows[j]["stage"]][0] for j in indices[:2]]
        if perm & 2:
            stages.reverse()
        for index in indices:
            add(index, reordered_question(rows[index], fields, stages, frame, outer))
    assert len(target) == 5 * len(rows)
    save_sequences(destination / "queries.npz", sequences)
    np.save(
        destination / "pixels.npy",
        np.frombuffer(b"".join(tile_ids), np.uint8).reshape(-1, 128),
    )
    np.save(destination / "target.npy", np.asarray(target, np.int32))
    write_json(
        destination / "report.json",
        dict(
            questions=len(target),
            mixed=4 * len(rows),
            prefix=len(rows),
            source=sha(DEFAULT / "train/index.json"),
            layouts=sha(ORDER_DATA / "train/layouts.json"),
            generator=sha(Path(__file__)),
            files={p.name: sha(p) for p in sorted(destination.iterdir())},
        ),
    )


if __name__ == "__main__":
    prepare()
