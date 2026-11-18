"""Hold two bronze images fixed and vary the pictured predicate between them."""

import random
import torch

from ilm.visual_lm.contextual_ink_binding import ContextualInkBinding
from scripts.prepare_visual_event_roles import OUT, load, write_new
from scripts.evaluate_historical_event_roles import WORK as HIST, mix
from scripts.evaluate_paired_student import read_streams, save_streams
from scripts.fit_contextual_ink_binding import RESULT, write
from scripts.prepare_visual_sequence_corpus import Rasterizer
from scripts.train_causal_glyph_translator import read_rows
from scripts.train_paired_student import restore
from scripts.train_pictured_revision import composer

WORK = OUT / "predicates"


def run():
    torch.set_num_threads(2)
    WORK.mkdir()
    bind = ContextualInkBinding.restore(RESULT / "binding.msgpack.gz")
    model = restore(OUT / "swapped.safetensors").eval()
    hist = load(HIST / "rows.json")
    render = Rasterizer()
    names = {}
    for r, x in zip(hist, read_streams("source", HIST)):
        for c in ["秦", "楚"]:
            if c in (r["first"], r["second"]):
                names[c] = x[r["source"].index(c)]
    assert len(names) == 2
    engine = composer()
    train = read_rows("train")
    previous = {r["source"] for r in train}
    for r in load(OUT / "train.json"):
        a, b = r["a"], r["b"]
        previous.update(
            [r["source"], r["source"].translate(str.maketrans({a: b, b: a}))]
        )
    lex = {
        render.tile(c): c
        for c in sorted(set("".join(r["new"] for r in train) + "攻救围入灭秦楚。"))
    }
    lex.update({bytes(v): "⟦" + c + "⟧" for c, v in names.items()})
    outputs = {k: [] for k in ("source", "swapped", "context", "composer")}
    rows = []
    prepared = []
    for verb in "攻救围入灭":
        for a, b in [("秦", "楚"), ("楚", "秦")]:
            text = a + verb + b + "。"
            x = mix(text, names, render)
            p = bind.choose(x)
            parts = dict(
                source=x,
                swapped=write(model, x, [])[0],
                context=write(model, x, p)[0],
                composer=engine.generate(x),
            )
            i = len(rows)
            rows.append(
                dict(
                    source=text,
                    source_seen_in_printed_training=text in previous,
                    first=a,
                    second=b,
                    verb=verb,
                    bindings=len(p),
                )
            )
            opts = {}
            for k, v in parts.items():
                outputs[k].append(v)
                opts[k] = "".join(lex.get(bytes(c), "▧") for c in v)
            prepared.append((i, opts))
    rng = random.Random(93715)
    rng.shuffle(prepared)
    review = []
    mapping = []
    for i, opts in prepared:
        choices = list(dict.fromkeys(opts.values()))
        rng.shuffle(choices)
        review.append(dict(source=opts["source"], choices=choices))
        mapping.append(
            dict(row=i, options={k: choices.index(v) for k, v in opts.items()})
        )
    for k, v in outputs.items():
        save_streams(k, v, WORK)
    write_new(WORK / "rows.json", rows)
    write_new(WORK / "review.json", review)
    write_new(WORK / "mapping.json", mapping)


if __name__ == "__main__":
    run()
