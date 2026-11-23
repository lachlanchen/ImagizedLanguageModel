"""Evaluate frozen answer choices, source coverage, and actual stopped image streams."""

from collections import Counter, defaultdict
import json

from safetensors.torch import load_file
import torch
from scipy.stats import binomtest

from ilm.visual_lm.bound_fact_reader import render_fact
from ilm.visual_lm.fact_realizer import FactRealizer
from ilm.visual_lm.span_choice import EXPERTS, ChoiceHead, pooled_choice
from scripts.evaluate_grounded_propositions import store_streams
from scripts.evaluate_participant_margin import save_png
from scripts.evaluate_pointing_causal_writer import load_streams
from scripts.prepare_span_choice import CACHE, PUBLIC
from scripts.prepare_visual_facts import Face, FONTS, WORK, write_json
from scripts.probe_participant_margin import broad_cases
from scripts.train_fact_realizer import target
from scripts.train_participant_margin import read
from scripts.train_pointing_causal_writer import sha
from scripts.train_span_choice import MODELS, gpu_owner, model, records
from scripts.train_visual_facts import rows


@torch.inference_mode()
def evaluate():
    if (PUBLIC / "results.json").exists():
        raise FileExistsError("Frozen evaluation already exists")
    reader = model()
    face = Face((FONTS[0], 2))
    fit = json.loads((PUBLIC / "fit.json").read_text())["arms"]
    permuted = ChoiceHead().to("cuda").eval()
    for arm, path, net in [
        ("trained", MODELS / "choice.safetensors", reader.choice),
        ("permuted", CACHE / "permuted.safetensors", permuted),
    ]:
        if sha(path) != fit[arm]["sha256"]:
            raise ValueError("Unbound selector")
        net.load_state_dict(load_file(str(path)))
    fresh = records("fresh")
    sets = {
        "fresh": [
            ((face(v["p"]), face(v["q"])), [face(t) for t in v["refs"]]) for v in fresh
        ],
        "prior_broad": broad_cases(face),
    }
    sets["historical"] = [
        ((p, q), [ref])
        for p, q, ref in zip(
            *(
                load_streams(WORK / f"alias_{k}.npz")[0]
                for k in ("pages", "questions", "reference")
            ),
            strict=True,
        )
    ]
    totals, examples = {}, {}
    direct_ledger = []
    for split, data in sets.items():
        counts = Counter()
        raw = []
        for k in range(0, len(data), 24):
            batch = data[k : k + 24]
            feat, agree, fields = reader.proposals([b[0] for b in batch])
            picks = [
                pooled_choice(net(feat), agree).tolist()
                for net in (permuted, reader.choice)
            ]
            for j, ((s, r, options), ((p, q), refs)) in enumerate(
                zip(fields, batch, strict=True)
            ):
                iy = picks[1][j]
                choices = (
                    options
                    if split != "historical"
                    else [render_fact(reader.base.frames[r], s, o) for o in options]
                )
                selected = choices[iy]
                raw.append(selected)
                hit = [v in refs for v in choices]
                covered, correct = any(hit), hit[iy]
                marks = dict(
                    zip(EXPERTS, hit, strict=True),
                    trained=correct,
                    permuted=hit[picks[0][j]],
                    oracle=covered,
                    uncovered=not covered,
                    wrong_selection=covered and not correct,
                    gained=correct and not hit[0],
                    lost=hit[0] and not correct,
                )
                counts.update(marks)
                item = dict(split=split, index=k + j, **marks)
                if split == "fresh":
                    v = fresh[k + j]
                    item.update(source=v["group"], key=v["key"])
                    tag = (
                        "recovery"
                        if marks["gained"]
                        else "miss"
                        if not covered
                        else None
                    )
                    if tag and tag not in examples:
                        examples[tag] = dict(
                            index=k + j,
                            key=v["key"],
                            source=v["group"],
                            question=v["q"],
                            refs=v["refs"],
                        )
                        for label, buf in [
                            ("question", q),
                            ("answer", selected),
                            ("generic", choices[0]),
                            ("reference", refs[0]),
                        ]:
                            save_png(PUBLIC / f"{tag}_{label}.png", buf)
                direct_ledger.append(item)
        counts["n"] = len(data)
        totals[split] = dict(counts)
        store_streams(CACHE / (split + "_answers.npz"), raw, [True] * len(raw))
    assert (
        totals["prior_broad"]["generic"] == 187
        and totals["prior_broad"]["likelihood"] == 119
    )
    write_json(CACHE / "direct_ledger.json", direct_ledger)
    write_json(PUBLIC / "examples.json", examples)
    writer = FactRealizer(WORK / "writer", "cuda")
    all_rows = rows("held") + read("new")
    stats = defaultdict(Counter)
    boards = defaultdict(lambda: defaultdict(list))
    emitted, stops = [[], []], [[], []]
    for start in range(0, len(all_rows), 24):
        batch = all_rows[start : start + 24]
        fields = reader.fields_many([(face(v["p"]), face(v["q"])) for v in batch])
        outputs = [
            writer.answer_fields(fields, order) for order in ("subject", "object")
        ]
        for j, v in enumerate(batch):
            good = []
            c = stats[v["group"]]
            for order in range(2):
                raw, stop = outputs[order][j]
                emitted[order].append(raw)
                stops[order].append(stop)
                good.append(bool(stop and raw == face(target(v, order))))
                c["stops"] += bool(stop)
            c.update(n=1, both=all(good), subject_order=good[0], object_order=good[1])
            if "board" in v:
                boards[v["group"]][v["board"]].append(all(good))
    for g, by_key in boards.items():
        assert all(len(x) == (8 if g == "clause_order" else 4) for x in by_key.values())
        stats[g].update(
            boards=len(by_key), all_both=sum(all(x) for x in by_key.values())
        )
    for i, suffix in enumerate(("subject", "object")):
        store_streams(CACHE / (suffix + ".npz"), emitted[i], stops[i])
    f = totals["fresh"]
    checks = dict(
        fresh=f["trained"] >= f["generic"],
        human=stats["cmrc"]["both"] + stats["drcd"]["both"] >= 34,
        exchanges=stats["new_source"]["all_both"] >= 79,
        clause_order=stats["clause_order"]["all_both"] >= 37,
    )
    p = (
        binomtest(f["gained"], f["gained"] + f["lost"]).pvalue
        if f["gained"] + f["lost"]
        else 1.0
    )
    report = dict(
        direct=totals,
        writer=dict(stats),
        checks=checks,
        passed=all(checks.values()),
        paired_fresh_p=p,
        by_source={
            g: dict(
                n=sum(v.get("source") == g for v in direct_ledger),
                trained=sum(
                    v["trained"] for v in direct_ledger if v.get("source") == g
                ),
                generic=sum(
                    v["generic"] for v in direct_ledger if v.get("source") == g
                ),
            )
            for g in ("cmrc", "drcd")
        },
        sha256={p.name: sha(p) for p in sorted(CACHE.glob("*.npz"))},
        ledger_sha256=sha(CACHE / "direct_ledger.json"),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
    )
    write_json(PUBLIC / "results.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    with gpu_owner():
        evaluate()
