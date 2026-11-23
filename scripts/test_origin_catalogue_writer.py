"""Evaluate a model on new catalogue cards and opposed requests about the same evidence."""

import json
from collections import defaultdict

import numpy as np
import torch

from ilm.etymology.image_catalogue import ImageCatalogue, STAGES
from ilm.visual_lm.prepared_glyph_responder import PreparedGlyphResponder
from scripts.audit_interval_teacher import save_image
from scripts.evaluate_page_evidence_writer import tensor
from scripts.evaluate_pointing_causal_writer import store_streams, load_streams
from scripts.fit_font_invariant_reader import Face
from scripts.ordered_image_corpus import PLATES
from scripts.train_causal_glyph_writer import record, sha, in_cells
from scripts.train_span_choice import gpu_owner
from scripts.train_typeface_writer import WORK, PUBLIC, BASE, FACES, HELD

__test__ = False


def assemble():
    cat = ImageCatalogue(WORK / "catalogue")
    canonical, held = Face(FACES[0]), Face(HELD)
    cards = json.loads((PLATES / "development_cards.json").read_text())
    previous = np.random.default_rng(15324).permutation(len(cards))[:48]
    excluded = {r["char"] for i in previous for r in cards[i]}
    names = sorted(
        c
        for c, es in cat.entries.items()
        if c not in excluded
        and all(v["split"] == "development" for e in es for v in e["forms"].values())
    )
    rng = np.random.default_rng(241205)
    groups = defaultdict(list)
    for i in rng.permutation(len(names)):
        name = names[i]
        ordered = max(cat.available(name).values(), key=len)[:2]
        if len(ordered) != 2 or ord(name) not in held.coverage:
            continue
        try:
            selected = cat.card(name, ordered, pool=names)
        except ValueError:
            continue
        p = cat.page(selected, canonical)
        stages = {r["stage"]: r for r in selected if r["char"] == name}
        raw = {s: bytes(cat.pixels[r["tile"]]) for s, r in stages.items()}
        if raw[ordered[0]] == raw[ordered[1]]:
            continue
        source = dict(name=name, rows=selected)
        for first, second in (ordered, ordered[::-1]):
            a, b = STAGES[first], STAGES[second]
            y = (
                canonical(f"{name}的{a}為")
                + raw[first]
                + canonical(f"，{b}為")
                + raw[second]
                + canonical("。")
            )
            usual = f"請介紹「{name}」的字形，依{a}、{b}的順序作答。"
            requests = {
                "familiar": canonical(usual),
                "rephrased": canonical(
                    f"先列{a}，再列{b}。要查的是「{name}」，各寫出名稱和字形。"
                ),
                "hana": held(usual),
                "english": canonical(
                    f"Show {name} in {first} then {second} script. Label both forms in Chinese."
                ),
            }
            for k, q in requests.items():
                groups[k].append((p, q, y, raw[first], raw[second], source))
        if len(groups["familiar"]) == 200:
            break
    return groups


def score(cases, ys, ends):
    exact = []
    forms = []
    for (_, _, y, a, b, _), out, end in zip(cases, ys, ends, strict=True):
        exact.append(bool(end and out == y))
        cells = in_cells(out)
        forms.append(bool(end and [v for v in cells if v in (a, b)] == [a, b]))
    return dict(
        n=len(ys),
        exact=sum(exact),
        forms_in_order=sum(forms),
        stops=sum(ends),
        pairs=len(ys) // 2,
        both=sum(a and b for a, b in zip(exact[::2], exact[1::2], strict=True)),
    )


@torch.inference_mode()
def evaluate():
    groups = assemble()
    stats = {}
    dest = WORK / "catalogue_evaluation"
    dest.mkdir()
    record(dest / "cases.json", [v[-1] for v in groups["familiar"][::2]])
    for arm, path in [
        ("parent", BASE),
        ("replay", WORK / "replay/page"),
        ("mapped", WORK / "mapped/page"),
    ]:
        assert all(
            sha(path / k) == h
            for k, h in json.loads((path.parent / "page_fit.json").read_text())[
                "weights"
            ].items()
        )
        net = PreparedGlyphResponder(path, "cuda")
        stats[arm] = {}
        for g, cases in groups.items():
            pairs = [(tensor(v[0]), tensor(v[1])) for v in cases]
            outputs = []
            for start in range(0, len(pairs), 16):
                outputs.extend(
                    (im.numpy().tobytes(), stop)
                    for im, stop in net.answer_many(pairs[start : start + 16])
                )
            ys, ends = zip(*outputs, strict=True)
            f = dest / (g + "_" + arm + ".npz")
            store_streams(f, ys, ends)
            original, stopped = load_streams(f)
            stats[arm][g] = score(cases, original, stopped) | {"sha256": sha(f)}
        del net
    ex = PUBLIC / "catalogue_example"
    ex.mkdir()
    ys, ends = load_streams(dest / "familiar_mapped.npz")
    i = next(
        (
            j
            for j in range(0, len(ys), 2)
            if all(ends[k] and ys[k] == groups["familiar"][k][2] for k in (j, j + 1))
        ),
        0,
    )
    p, q, y, _, _, source = groups["familiar"][i]
    for n, raw in (
        ("page", p),
        ("question", q),
        ("reference", y),
        ("forward", ys[i]),
        ("reverse", ys[i + 1]),
    ):
        save_image(ex / (n + ".png"), raw)
    record(ex / "source.json", source)
    record(
        PUBLIC / "catalogue_evaluation.json",
        dict(
            scores=stats,
            cases_sha256=sha(dest / "cases.json"),
            example_index=i,
            example_name=source["name"],
            example_both_exact=all(
                ends[k] and ys[k] == groups["familiar"][k][2] for k in (i, i + 1)
            ),
        ),
    )
    print(stats, flush=True)


if __name__ == "__main__":
    with gpu_owner():
        evaluate()
