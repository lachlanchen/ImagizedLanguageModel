"""Measure source reading and independently stopped writing on reserved page images."""

from collections import defaultdict
import json
from time import monotonic

import torch
from ilm.visual_lm.font_invariant_reader import FontInvariantReader
from ilm.visual_lm.pictured_answer_selection import read_options
from ilm.visual_lm.prepared_glyph_responder import PreparedGlyphResponder
from scripts.evaluate_page_evidence_writer import tensor
from scripts.evaluate_pointing_causal_writer import load_streams, store_streams
from scripts.mine_stopped_book_replies import PARENT
from scripts.prepare_bilingual_book_intervals import PUBLIC, WORK
from scripts.reserve_bilingual_interval_books import CHECK, packets
from scripts.reserve_first_glyph_books import reserved
from scripts.reserve_preference_books import literary_cases
from scripts.train_causal_glyph_writer import record, sha
from scripts.train_span_choice import gpu_owner
from scripts.train_visual_facts import READER

READERS = {
    "parent": READER,
    "aligned": WORK / "aligned",
    "exchanged": WORK / "exchanged",
}


def save(directory, label, cases, images, stops, elapsed):
    path = directory / (label + ".npz")
    store_streams(path, images, stops)
    assert load_streams(path) == (images, stops)
    flags = [
        bool(c.grade(y, end)[2]) for c, y, end in zip(cases, images, stops, strict=True)
    ]
    return dict(
        n=len(flags),
        exact=sum(flags),
        stopped=sum(stops),
        flags=flags,
        seconds=elapsed,
        sha256=sha(path),
    )


def four(groups):
    both = [
        z and e
        for z, e in zip(groups["zh"]["flags"], groups["en"]["flags"], strict=True)
    ]
    return dict(
        complete=sum(a and b for a, b in zip(both[::2], both[1::2], strict=True)),
        n=len(both) // 2,
    )


@torch.inference_mode()
def evaluate():
    groups = packets()
    groups.update(dev_four=reserved(), dev_six=literary_cases())
    scores = defaultdict(dict)
    directory = WORK / "evaluation"
    directory.mkdir(exist_ok=False)
    for arm, path in READERS.items():
        weights = (
            json.loads((PUBLIC / "aligned_fit.json").read_text())["parent"]
            if arm == "parent"
            else json.loads((PUBLIC / (arm + "_fit.json")).read_text())["weights"]
        )
        assert all(sha(path / k) == v for k, v in weights.items())
        reader = FontInvariantReader.restore(path, "cuda").eval()
        dest = directory / arm
        dest.mkdir()
        for register, sets in groups.items():
            stats = {}
            for key, cases in sets.items():
                lang = key.removeprefix("fresh_")
                start = monotonic()
                _, images = read_options(reader, [(c.p, c.q) for c in cases])
                stats[lang] = save(
                    dest,
                    register + "_" + lang,
                    cases,
                    images,
                    [True] * len(cases),
                    monotonic() - start,
                )
            stats["four"] = four(stats)
            scores[arm][register] = stats
        del reader
    net = PreparedGlyphResponder(PARENT, "cuda")
    for arm in ("full", *READERS):
        dest = directory / ("writer_" + arm)
        dest.mkdir()
        for register in ("classical", "modern"):
            stats = {}
            for key, cases in groups[register].items():
                lang = key.removeprefix("fresh_")
                label = register + "_" + lang
                spans = (
                    [c.p for c in cases]
                    if arm == "full"
                    else load_streams(directory / arm / (label + ".npz"))[0]
                )
                pairs = [
                    (tensor(p), tensor(c.q)) for p, c in zip(spans, cases, strict=True)
                ]
                start = monotonic()
                replies = [
                    reply
                    for i in range(0, len(pairs), 8)
                    for reply in net.answer_many(pairs[i : i + 8])
                ]
                images = [y.numpy().tobytes() for y, _ in replies]
                stops = [bool(stopped) for _, stopped in replies]
                stats[lang] = save(
                    dest, label, cases, images, stops, monotonic() - start
                )
                if arm != "full":
                    was = scores[arm][register][lang]["flags"]
                    now = stats[lang]["flags"]
                    stats[lang]["lost"] = sum(
                        a and not b for a, b in zip(was, now, strict=True)
                    )
                    stats[lang]["repaired"] = sum(
                        b and not a for a, b in zip(was, now, strict=True)
                    )
            stats["four"] = four(stats)
            scores["writer_" + arm][register] = stats
    gates = {}
    for register in ("classical", "modern"):
        group = scores["aligned"][register]
        gates[register] = dict(
            english=all(
                group["en"]["exact"] > scores[a][register]["en"]["exact"]
                for a in ("parent", "exchanged")
            ),
            chinese=group["zh"]["exact"] >= scores["parent"][register]["zh"]["exact"],
        )
        if register == "classical":
            gates[register]["four"] = all(
                group["four"]["complete"] > scores[a][register]["four"]["complete"]
                for a in ("parent", "exchanged")
            )
    record(
        PUBLIC / "result.json",
        dict(
            scores=scores,
            gates=gates,
            selected=all(all(g.values()) for g in gates.values()),
            reservation=sha(CHECK),
            writer_parameters=sum(p.numel() for p in net.net.parameters()),
            writer_source=str(PARENT),
        ),
    )


if __name__ == "__main__":
    with gpu_owner():
        evaluate()
