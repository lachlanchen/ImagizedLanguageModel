"""Measure independent answer images, crossed supplied evidence and physical edit collisions."""

from collections import defaultdict
from hashlib import sha256
import numpy as np
import torch
from scripts.prepare_fact_channel import WORK, PUB, MODEL, Ink, read, dump, sha
from scripts.write_fact_channel import VisualAnswer
from scripts.train_region_writer import gpu_owner
from scripts.verse_gpu_runtime import configure_cuda_cli
from scripts.evaluate_region_writer import changed, offpage, e2e_scorer
from scripts.evaluate_phrase_region import measurements, group_counts, intervals
from scripts.evaluate_visual_discourse import packed
from scripts.prepare_region_writer import WORK as BASE
from scripts.audit_region_writer import wording


def observed(net, rows, ink, gold=False):
    qs = [ink.encode(r["q"]) for r in rows]
    ps = [[ink.encode(b) for b in r["blocks"]] for r in rows]
    supplied = []
    if gold:
        for r in rows:
            fact = (
                r["blocks"][r["valid"][0]]
                if r["task"] == "restaurant"
                else next(iter(r["facts"].values()))
            )
            k = next((i for i in r["valid"] if fact in r["blocks"][i]), r["valid"][0])
            supplied.append((k, r["blocks"][k].find(fact), ink.encode(fact)))
    tiles = ink.array()
    chars = dict(enumerate(net.chars))
    for c, i in ink.chars.items():
        chars.setdefault(i, c)
    output = []
    for j in range(0, len(rows), 12):
        results, spans = net.answer(
            tiles,
            qs[j : j + 12],
            ps[j : j + 12],
            supplied[j : j + 12] if gold else None,
        )
        for i, ((ids, stop), span) in enumerate(zip(results, spans), j):
            raw = tiles[np.asarray(ids, int)].tobytes()
            glyphs = "".join(map(chr, ids))
            blocks = ps[i]

            def checksum(v):
                return sha256(tiles[np.asarray(v, int)].tobytes()).hexdigest()

            rec = dict(
                task=rows[i]["task"],
                family=rows[i]["family"],
                stop=stop,
                text="".join(chars[k] for k in ids),
                cells=len(ids),
                sha256=sha256(raw).hexdigest(),
                ink_hex=raw.hex(),
                selected=None,
                visible_reply=bool(glyphs)
                and any(glyphs in "".join(map(chr, x)) for x in [qs[i], *blocks]),
            )
            if span is not None:
                b, offset, f = span
                rec.update(
                    selected=b,
                    offset=offset,
                    fact_sha256=checksum(f),
                    fact_cells=len(f),
                    evidence_region_sha256=checksum(blocks[b]),
                    fact_text="".join(chars[k] for k in f),
                    fact_hex=tiles[np.asarray(f, int)].tobytes().hex(),
                )
            output.append(rec)
    return output


def edits_audit(rows, original, edits, altered, arm, ink, metrics, after):
    """Measure the deterministic obstruction separately for the writer's exact inputs."""
    groups = defaultdict(lambda: defaultdict(int))
    for j, (e, out) in enumerate(zip(edits, altered, strict=True)):
        i = e["original"]
        r = rows[i]
        old = original[i]

        def refset(row):
            return {
                b"".join(ink.tiles[k] for k in ink.encode(s)) for s in row["answers"]
            }

        assert r["q"] == e["q"]
        disjoint = refset(e).isdisjoint(refset(r))
        same = (
            (old["fact_sha256"] == out["fact_sha256"])
            if arm != "direct"
            else r["blocks"] == e["blocks"]
        )
        if arm == "predicted":
            same = (
                same and old["evidence_region_sha256"] == out["evidence_region_sha256"]
            )
        g = groups[e["task"]]
        g["edits"] += 1
        g["disjoint_accepted_images"] += disjoint
        g["identical_writer_inputs"] += same
        g["incompatible_identical_inputs"] += same and disjoint
        both = metrics[i]["bitmap_exact"] and after[j]["bitmap_exact"]
        g["both_exact"] += both
        g["identical_input_different_output"] += same and (
            old["sha256"] != out["sha256"] or old["stop"] != out["stop"]
        )
        if same and disjoint and both:
            raise ValueError("Violated paired collision bound")
    return groups


def evaluate():
    spec = read(PUB / "corpus.json")
    if sha(WORK / "confirmation.json") != spec["files"]["confirmation.json"]:
        raise ValueError("Changed confirmation")
    rows = read(WORK / "confirmation.json")
    edits = changed(rows)
    score = e2e_scorer()
    older = read(BASE / "dev-source.json")
    coqa, too_large, norm = offpage()
    metrics = {}
    summary = {}
    retention = {}
    audit = {}
    with gpu_owner(), torch.inference_mode():
        torch.set_float32_matmul_precision("highest")
        for arm in ("direct", "human", "predicted"):
            net = VisualAnswer(arm, "cuda")
            ink = Ink(net.palette)
            z = {}
            for name, rs, gold in (
                ("original", rows, False),
                ("edited", edits, False),
                ("supplied", rows, True),
                ("edited_supplied", edits, True),
                ("prior", older, False),
                ("coqa", coqa, False),
            ):
                z[name] = observed(net, rs, ink, gold)
            packed(WORK / ("fresh-" + arm + ".json.gz"), z)
            for original, edit in (
                ("original", "edited"),
                ("supplied", "edited_supplied"),
            ):
                m = measurements(rows, z[original], ink, score)
                e = measurements(edits, z[edit], ink, score)
                result = group_counts(rows, m, edits, e)
                if arm != "direct" and original == "original":
                    for t in result:
                        group = [
                            (r, v) for r, v in zip(rows, z[original]) if r["task"] == t
                        ]
                        result[t]["extracted_exact_fact"] = sum(
                            v["fact_hex"]
                            == b"".join(
                                ink.tiles[k] for k in ink.encode(r["facts"]["answer"])
                            ).hex()
                            for r, v in group
                        )
                summary.setdefault(arm, {})[original] = result
                if original == "original":
                    metrics[arm] = m
                    audit[arm] = edits_audit(
                        rows, z[original], edits, z[edit], arm, ink, m, e
                    )
            ret = group_counts(
                older, measurements(older, z["prior"], ink, score), [], []
            )
            ret["coqa"] = dict(
                n=len(coqa),
                outside_geometry=too_large,
                exact=sum(
                    v["stop"] and any(norm(v["text"]) == norm(s) for s in r["answers"])
                    for r, v in zip(coqa, z["coqa"])
                ),
            )
            retention[arm] = ret
            dump(PUB / "scores.json", summary)
            print(arm, summary[arm], flush=True)
    dump(PUB / "retention.json", retention)
    dump(PUB / "edit-collisions.json", audit)
    dump(
        PUB / "intervals.json",
        {
            k: intervals(rows, {"phrase": metrics[k], "single": metrics["direct"]})
            for k in ("human", "predicted")
        },
    )


if __name__ == "__main__":
    configure_cuda_cli()
    evaluate()
