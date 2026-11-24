"""Score complete answer images on intact conversations and independent passage pairs."""

from collections import defaultdict
from hashlib import sha256
from pathlib import Path
import argparse
import numpy as np
import pyarrow.parquet as pq
import torch

from ilm.visual_lm.book_prefix import observations
from scripts.prepare_evidence_views import (
    Alphabet,
    DROP,
    PUB,
    WORK,
    MODEL,
    SOURCES,
    query,
    sha,
)
from scripts.prepare_open_exam import read, dump
from scripts.probe_passage_rationales import pages as rationale_pages
from scripts.score_dialogue_ink_references import official_index
from scripts.write_recurrent_evidence import generate
from scripts.write_open_exam import load
from scripts.write_visual_discourse import Writer
from scripts.train_paired_evidence import gpu_owner
from scripts.evaluate_visual_discourse import packed


def pair_rows(rows, norm):
    pair = []
    for r in rows:
        refs = {norm(x) for x in r["refs"]}
        refs.discard("")
        if not refs or not 0 < len(r["question"]) <= 320:
            continue
        if pair and (
            r["raw_q"] == pair[0]["raw_q"]
            or refs.intersection(map(norm, pair[0]["refs"]))
        ):
            continue
        pair.append(r)
        if len(pair) == 2:
            return pair
    return []


def requires_new_phrase(row, norm):
    visible = " " + norm(row["page"] + " " + row["question"]) + " "
    refs = list(map(norm, row["refs"]))
    return all(len(z.split()) >= 2 and " " + z + " " not in visible for z in refs)


def groups(judge, dev):
    coqa, composition, composition_evidence = [], [], []
    for d in read(dev)["data"]:
        p = d["story"]
        if not 0 < len(p) <= 4096:
            continue
        previous = ""
        rows = []
        for q, a in zip(d["questions"], d["answers"], strict=True):
            raw = q["input_text"]
            turn = q["turn_id"]
            if 0 < len(raw) <= 160:
                rows.append(
                    dict(
                        doc=d["id"],
                        turn=turn,
                        question=query(raw, previous),
                        raw_q=raw,
                        page=p,
                        refs=judge.gold_data[d["id"], turn],
                    )
                )
            previous = raw + " " + a["input_text"]
        coqa.extend(pair_rows(rows, judge.normalize_answer))
        evidence = {a["turn_id"]: a for a in d["answers"]}
        for r in rows:
            if not requires_new_phrase(r, judge.normalize_answer):
                continue
            composition.append(r)
            a = evidence[r["turn"]]
            lo, hi = a["span_start"], a["span_end"]
            if 0 <= lo < hi <= len(p) and p[lo:hi] == a["span_text"]:
                composition_evidence.append(dict(r, page=p[lo:hi], highlight=p[lo:hi]))
    if sha(DROP) != SOURCES[DROP]:
        raise ValueError("Changed independent evaluation source")
    drop = defaultdict(list)
    for r in pq.read_table(DROP).to_pylist():
        refs = r["answers_spans"]["spans"]
        if (
            len(set(refs)) != 1
            or not 0 < len(r["passage"]) <= 4096
            or len(r["question"]) > 160
        ):
            continue
        drop[(r["section_id"], r["passage"])].append(
            dict(
                doc=r["section_id"],
                turn=r["query_id"],
                raw_q=r["question"],
                question=r["question"],
                page=r["passage"],
                refs=refs,
            )
        )
    cases = []
    for key in sorted(
        drop, key=lambda k: sha256((k[0] + "\0" + k[1]).encode()).digest()
    ):
        cases.extend(pair_rows(drop[key], judge.normalize_answer))
        if len(cases) == 256:
            break
    assert len(cases) == 256 and len({r["doc"] for r in cases}) == 128
    rationales, _ = rationale_pages(dev)
    exchanged = []
    for i, r in enumerate(coqa):
        changed = coqa[i ^ 1]
        assert r["doc"] == changed["doc"] and r["page"] == changed["page"]
        exchanged.append(dict(r, question=changed["question"]))
    wrong_pages = [
        dict(r, page=coqa[(i + 2) % len(coqa)]["page"]) for i, r in enumerate(coqa)
    ]
    assert all(a["page"] != b["page"] for a, b in zip(wrong_pages, coqa))
    result = {
        "coqa": coqa,
        "exchanged_page": wrong_pages,
        "drop": cases,
        "composition": composition,
        "composition_evidence": composition_evidence,
        "exchanged": exchanged,
    }
    for view in ("full", "evidence", "disjoint", "masked"):
        result[view] = [
            dict(
                doc=r["doc"],
                turn=1,
                question=r["question"],
                raw_q=r["question"],
                page=r[view],
                refs=judge.gold_data[r["doc"], 1],
                highlight=r["evidence"],
            )
            for r in rationales
        ]
    return result


def summaries(rows):
    result = {}
    for name, g in (
        ("all", rows),
        ("lexical", [r for r in rows if r["lexical"]]),
        ("absent_multiword", [r for r in rows if r["absent_multiword"]]),
        ("absent_evidence", [r for r in rows if r["absent_evidence"]]),
    ):
        bydoc = defaultdict(list)
        for r in g:
            bydoc[r["doc"]].append(r["exact"])
        pairs = [p for p in bydoc.values() if len(p) == 2]
        result[name] = dict(
            n=len(g),
            exact=sum(r["exact"] for r in g),
            stopped=sum(r["stop"] for r in g),
            both=sum(all(p) for p in pairs),
            pairs=len(pairs),
            mean_f1=100 * np.mean([r["f1"] for r in g])
            if g and g[0]["f1"] is not None
            else None,
        )
    return result


def run(arms, model_dir, report, work):
    judge, _, _, dev, _ = official_index()
    norm = judge.normalize_answer
    data = groups(judge, dev)
    alphabet = Alphabet()
    coded = {
        k: [(alphabet.encode(r["question"]), alphabet.encode(r["page"])) for r in rows]
        for k, rows in data.items()
    }
    tiles = alphabet.array()
    report.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    with gpu_owner(), torch.inference_mode():
        writer = Writer(device="cuda")
        x = torch.from_numpy(observations({"tiles": tiles}, "ink")).cuda()
        for arm in arms:
            net = load(arm, "cuda", model_dir)
            metrics = {}
            private = []
            public = []
            for k, rows in data.items():
                results = []
                for i in range(0, len(rows), 12):
                    qs, ps = map(list, zip(*coded[k][i : i + 12]))
                    out = generate(
                        writer.model,
                        net,
                        x,
                        alphabet.n,
                        qs,
                        ps,
                        [p + q for q, p in zip(qs, ps)],
                        64,
                        geometry=(320, 4096),
                    )
                    for r, (ids, stop) in zip(rows[i : i + 12], out, strict=True):
                        text = "".join(alphabet.chars[int(v)] for v in ids)
                        t = norm(text)
                        refs = list(map(norm, r["refs"]))
                        visible = " " + norm(r["page"] + " " + r["question"]) + " "
                        missing = all(
                            len(z.split()) >= 2 and " " + z + " " not in visible
                            for z in refs
                        )
                        e = " " + norm(r.get("highlight", "")) + " "
                        no_ref = "highlight" in r and all(
                            z and " " + z + " " not in e for z in refs
                        )
                        f1 = (
                            judge.compute_turn_score(r["doc"], r["turn"], text)["f1"]
                            if k != "drop"
                            else None
                        )
                        row = dict(
                            doc=r["doc"],
                            turn=r["turn"],
                            exact=stop and t in refs,
                            stop=stop,
                            cells=len(ids),
                            absent_multiword=missing,
                            lexical=not bool(set(refs) & {"yes", "no", "unknown"}),
                            absent_evidence=no_ref,
                            f1=(f1 if stop else 0) if f1 is not None else None,
                            sha256=sha256(
                                tiles[np.asarray(ids, int)].tobytes()
                            ).hexdigest(),
                        )
                        results.append(row)
                        private.append(dict(row, group=k, text=text, ids=ids))
                metrics[k] = summaries(results)
                public.extend(dict(r, group=k) for r in results)
                print(arm, k, metrics[k]["all"], flush=True)
            packed(work / (arm + "-responses.json.gz"), private)
            packed(report / (arm + "-outcomes.json.gz"), public)
            dump(report / (arm + "-scores.json"), metrics)


if __name__ == "__main__":
    from scripts.verse_gpu_runtime import configure_cuda_cli

    configure_cuda_cli()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm", choices=("parent", "whole", "highlight"), action="append"
    )
    parser.add_argument("--model-dir", type=Path, default=MODEL)
    parser.add_argument("--output", type=Path)
    a = parser.parse_args()
    run(
        a.arm or ("parent", "whole", "highlight"),
        a.model_dir,
        a.output or PUB,
        a.output or WORK,
    )
