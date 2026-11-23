"""Score the frozen rerankers on physical answers and the full autoregressive writer."""

from collections import Counter, defaultdict
import json
import torch
from scipy.stats import binomtest
from safetensors.torch import load_file

from ilm.visual_lm.bound_fact_reader import render_fact
from ilm.visual_lm.fact_realizer import FactRealizer
from ilm.visual_lm.span_choice import EXPERTS, pooled_choice
from ilm.visual_lm.span_list import SpanListReader, SpanRanker, image_choice
from scripts.evaluate_participant_margin import save_png
from scripts.evaluate_pointing_causal_writer import load_streams, store_streams
from scripts.prepare_span_list import CACHE, PUBLIC, split
from scripts.prepare_span_choice import PUBLIC as EARLIER
from scripts.prepare_visual_facts import ROOT, WORK, Face, FONTS, write_json
from scripts.train_fact_realizer import authored_clause, target
from scripts.train_participant_margin import read
from scripts.train_pointing_causal_writer import sha
from scripts.train_span_choice import model, gpu_owner, records
from scripts.train_span_list import WEIGHTS
from scripts.train_visual_facts import rows


def pictured(face, items):
    return [((face(v["p"]), face(v["q"])), [face(t) for t in v["refs"]]) for v in items]


def contains(raw, part):
    return any(
        raw[i : i + len(part)] == part for i in range(0, len(raw) - len(part) + 1, 128)
    )


@torch.inference_mode()
def evaluate():
    if (PUBLIC / "results.json").exists():
        raise FileExistsError("Frozen list evaluation")
    face = Face((FONTS[0], 2))
    experts = model()
    old_path = ROOT / "models/span_choice/choice.safetensors"
    assert (
        sha(old_path)
        == json.loads((EARLIER / "fit.json").read_text())["arms"]["trained"]["sha256"]
    )
    experts.choice.load_state_dict(load_file(str(old_path)))
    reader = SpanListReader(experts)
    arms = {}
    for arm in ("wide", "first", "shuffled"):
        hp = (WEIGHTS if arm == "wide" else CACHE) / (arm + ".safetensors")
        assert (
            sha(hp)
            == json.loads((PUBLIC / "fit.json").read_text())["arms"][arm]["weights"]
        )
        h = SpanRanker().cuda().eval()
        h.load_state_dict(load_file(str(hp)))
        arms[arm] = h
    reader.head = arms["wide"]
    current, prior = split("fresh"), records("fresh")
    sets = {
        "second": pictured(face, current),
        "first_sample": pictured(face, prior),
        "historical": [
            ((p, q), [ref])
            for p, q, ref in zip(
                *(
                    load_streams(WORK / f"alias_{n}.npz")[0]
                    for n in ("pages", "questions", "reference")
                ),
                strict=True,
            )
        ],
    }
    earlier = json.loads((EARLIER / "coverage.json").read_text())["cases"]
    summary = {}
    examples = {}
    ledger = []
    for name, data in sets.items():
        counts = Counter()
        outputs = []
        for k in range(0, len(data), 24):
            chunk = data[k : k + 24]
            ps = [x[0] for x in chunk]
            feat, valid, classes, fields = reader.proposals(ps)
            old, agree, oldfields = experts.proposals(ps)
            old_idx = pooled_choice(experts.choice(old), agree).tolist()
            picks = {
                arm: image_choice(
                    h(
                        feat,
                        valid
                        if arm != "first"
                        else valid & (torch.arange(32, device="cuda") < 4),
                    ),
                    classes,
                ).tolist()
                for arm, h in arms.items()
            }
            for j, ((s, r, opts), (_, _, orig), ((_, q), refs)) in enumerate(
                zip(fields, oldfields, chunk, strict=True)
            ):

                def output(o):
                    return (
                        render_fact(experts.base.frames[r], s, o)
                        if name == "historical"
                        else o
                    )

                guesses = {arm: output(opts[ix[j]]) for arm, ix in picks.items()}
                guesses.update(
                    {arm: output(o) for arm, o in zip(EXPERTS, orig, strict=True)},
                    pooled=output(orig[old_idx[j]]),
                )
                hits = {arm: v in refs for arm, v in guesses.items()}
                cover = any(output(o) in refs for o in opts if o)
                top = any(output(o) in refs for o in orig)
                counts.update(hits)
                counts.update(
                    n=1,
                    oracle8=cover,
                    oracle1=top,
                    additional=cover and not top,
                    recovered=hits["wide"] and not top,
                    gain=hits["wide"] and not hits["generic"],
                    loss=hits["generic"] and not hits["wide"],
                )
                i = k + j
                outputs.append(guesses["wide"])
                row = dict(sample=name, i=i, **hits, oracle8=cover, oracle1=top)
                if name == "first_sample":
                    assert (
                        earlier[i]["first_rank_if_at_most_eight"] is not None
                    ) == cover
                    assert (earlier[i]["first_rank_if_at_most_eight"] == 1) == top
                if name == "second":
                    v = current[i]
                    row.update(source=v["group"], key=v["key"])
                    tag = (
                        "gain"
                        if hits["wide"] and not top
                        else "loss"
                        if hits["generic"] and not hits["wide"]
                        else None
                    )
                    if tag and tag not in examples:
                        examples[tag] = dict(
                            key=v["key"], i=i, q=v["q"], refs=v["refs"]
                        )
                        for label, buf in [
                            ("question", q),
                            ("answer", guesses["wide"]),
                            ("generic", guesses["generic"]),
                            ("reference", refs[0]),
                        ]:
                            save_png(PUBLIC / f"{tag}_{label}.png", buf)
                ledger.append(row)
        summary[name] = dict(counts)
        store_streams(CACHE / (name + ".npz"), outputs, [True] * len(outputs))
    write_json(PUBLIC / "examples.json", examples)
    write_json(CACHE / "direct.json", ledger)
    writer = FactRealizer(WORK / "writer", "cuda")
    stats = defaultdict(Counter)
    boards = defaultdict(lambda: defaultdict(list))
    streams = [[], []]
    stops = [[], []]
    natural, free_stops = [], []
    free_cases = []
    allrows = rows("held") + read("new")
    for k in range(0, len(allrows), 24):
        chunk = allrows[k : k + 24]
        fields = reader.fields_many([(face(v["p"]), face(v["q"])) for v in chunk])
        outputs = [
            writer.answer_fields(fields, order) for order in ("subject", "object")
        ]
        take = [j for j, v in enumerate(chunk) if authored_clause(v)][
            : max(0, 64 - len(natural))
        ]
        if take:
            for j, (y, stop) in zip(
                take,
                writer.answer_fields([fields[j] for j in take], "natural"),
                strict=True,
            ):
                v = chunk[j]
                src = authored_clause(v)
                natural.append(y)
                free_stops.append(stop)
                free_cases.append(
                    dict(
                        key=v["key"],
                        stopped=stop,
                        contains_both=contains(y, face(v["s"]))
                        and contains(y, face(v["o"])),
                        exact_source_clause=bool(stop and y == face(src)),
                    )
                )
        for j, v in enumerate(chunk):
            c = stats[v["group"]]
            good = []
            for order in range(2):
                y, stop = outputs[order][j]
                streams[order].append(y)
                stops[order].append(stop)
                c["stops"] += stop
                good.append(bool(stop and y == face(target(v, order))))
            c.update(n=1, both=all(good), object=fields[j][2] == face(v["o"]))
            if "board" in v:
                c["rival"] += fields[j][2] == face(v["rival"])
                boards[v["group"]][v["board"]].append(all(good))
    for g, b in boards.items():
        assert all(len(v) == (8 if g == "clause_order" else 4) for v in b.values())
        stats[g].update(boards=len(b), all_both=sum(all(v) for v in b.values()))
    for i, n in enumerate(("subject", "object")):
        store_streams(CACHE / (n + ".npz"), streams[i], stops[i])
    store_streams(CACHE / "natural.npz", natural, free_stops)
    write_json(CACHE / "natural.json", free_cases)
    fresh = summary["second"]
    checks = dict(
        generic=fresh["wide"] > fresh["generic"],
        first=fresh["wide"] > fresh["first"],
        human=stats["cmrc"]["both"] + stats["drcd"]["both"] >= 34,
        exchanges=stats["new_source"]["all_both"] >= 79,
        reordered=stats["clause_order"]["all_both"] >= 37,
    )
    by_source = {
        g: {
            key: sum(v[key] for v in ledger if v.get("source") == g)
            for key in ("generic", "wide", "first", "pooled", "oracle8")
        }
        for g in ("cmrc", "drcd")
    }
    report = dict(
        direct=summary,
        writer=dict(stats),
        free=dict(
            n=len(natural),
            **{
                k: sum(v[k] for v in free_cases)
                for k in ("stopped", "contains_both", "exact_source_clause")
            },
        ),
        by_source=by_source,
        checks=checks,
        passed=all(checks.values()),
        paired_p=binomtest(fresh["gain"], fresh["gain"] + fresh["loss"]).pvalue
        if fresh["gain"] + fresh["loss"]
        else 1.0,
        output_sha256={p.name: sha(p) for p in sorted(CACHE.glob("*.npz"))},
        direct_ledger=sha(CACHE / "direct.json"),
    )
    write_json(PUBLIC / "results.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    with gpu_owner():
        evaluate()
