"""Recount saved composition outcomes against the bound source questions and answers."""

import argparse
import gzip
import json
import math
from collections import Counter
from pathlib import Path

from scripts.dialogue_teacher_stories import absent
from scripts.evaluate_compose_dialogue import paths_for, sample_pixels
from scripts.evaluate_optical_passage_qa import normalized
from scripts.prepare_compose_dialogue import MODEL, PUB, prepared
from scripts.prepare_dialogue_ink import sha


def require(equal, message):
    if not equal:
        raise ValueError(message)


def bound_rows(folder, file, digest):
    path = PUB / "responses" / folder / file
    require(sha(path) == digest, str(path))
    content = path.read_bytes()
    if path.suffix == ".gz":
        content = gzip.decompress(content)
    return json.loads(content)


def audit_replies(data, folder, models):
    counts = {}
    for group in ("lists", "fresh"):
        rows, pages = data[group]["rows"], data[group]["pages"]
        difficult = [absent([r[5]], (pages[r[1]], r[3], r[4])) for r in rows]
        for tag, arm in models[group].items():
            rec = bound_rows(
                folder, group + "-" + tag.replace("/", "-") + ".json.gz", arm["outputs"]
            )
            totals = {label: Counter() for label in ("all", "difficult")}
            for row, record, hard in zip(rows, rec, difficult, strict=True):
                hits = []
                for name in ("original", "question", "passage"):
                    out = record[name]
                    hits.append(
                        out["stop"] and normalized(out["answer"]) == normalized(row[5])
                    )
                for label in ("all", "difficult") if hard else ("all",):
                    totals[label].update(
                        n=1,
                        original=hits[0],
                        question=hits[1],
                        passage=hits[2],
                        grounded=hits == [True, False, False],
                    )
            for label, t in totals.items():
                expected = arm["scores"][label]
                require(
                    t["n"] == expected["n"]
                    and t["grounded"] == expected["grounded"]
                    and [t[k] for k in ("original", "question", "passage")]
                    == expected["exact"],
                    f"{group}/{tag}/{label}",
                )
            counts[group + "/" + tag] = totals["difficult"]["grounded"]
    return counts


def audit_search(data, result):
    examples = {}
    for group in ("fresh", "lists"):
        info = result[group]
        records = bound_rows("search", group + "-proposals.json.gz", info["outputs"])
        counts = {k: Counter() for k in ("original", "question", "passage")}
        require(
            len(records) == info["n"] == len({r[0] for r in records}),
            "Search cohort mismatch",
        )
        grounded = 0
        for index, record in records:
            row = data[group]["rows"][index]
            target = normalized(row[5])
            ranks = []
            for arm, c in counts.items():
                candidates = record[arm]
                scores = [v for _, v in candidates]
                require(
                    0 < len(scores) <= 32
                    and all(map(math.isfinite, scores))
                    and scores == sorted(scores, reverse=True)
                    and len({word for word, _ in candidates}) == len(scores),
                    "Invalid proposal list",
                )
                rank = next(
                    (
                        i
                        for i, (word, _) in enumerate(candidates, 1)
                        if normalized(word) == target
                    ),
                    0,
                )
                c.update(top_exact=rank == 1, top8=0 < rank <= 8, top32=rank > 0)
                ranks.append(rank)
            grounded += ranks[0] == 1 and ranks[1] != 1 and ranks[2] != 1
            outcome = "found" if ranks[0] else "miss"
            examples.setdefault(
                group + "/" + outcome,
                dict(
                    row=index,
                    question=row[3],
                    reference=row[5],
                    top=record["original"][0][0],
                    rank=ranks[0] or None,
                ),
            )
        for arm, c in counts.items():
            require(
                all(v == info["results"][arm][k] for k, v in c.items()),
                "Proposal ranks differ",
            )
        require(
            grounded == info["results"]["grounded_top"],
            "Exchange-sensitive search count differs",
        )
    return examples


def count_assignments(rows):
    counts = Counter()
    for r in rows:
        (a, b), (c, d) = r["ll"]
        require(all(map(math.isfinite, (a, b, c, d))), "Nonfinite assignment")
        counts.update(assigned=a + d > b + c, ties=a + d == b + c, both=a > b and d > c)
    return counts


def audit_pairs(pairs, exchange):
    baseline = {}
    for history, tags in pairs["history"].items():
        for tag, summary in tags.items():
            file = history + "-" + tag.replace("/", "-") + ".json"
            rows = bound_rows("pairs", file, summary["likelihoods"])
            turns = [tuple(r["turns"]) for r in rows]
            require(
                len({i for ij in turns for i in ij}) == 2 * pairs["pairs"],
                "Reused question",
            )
            counts = count_assignments(rows)
            require(all(v == summary[k] for k, v in counts.items()), file)
            if history == "empty":
                baseline[tag] = {tuple(r["turns"]): r for r in rows}
    for tag, summary in exchange["models"].items():
        file = tag.replace("/", "-") + ".json"
        rows = bound_rows("exchange", file, summary["likelihoods"])
        original = [baseline[tag][tuple(r["turns"])] for r in rows]
        require(
            all(r["source"] != o["story"] for r, o in zip(rows, original, strict=True)),
            file,
        )
        require(
            len(rows) == summary["all"]["n"]
            and count_assignments(original)["assigned"] == summary["all"]["original"]
            and count_assignments(rows)["assigned"] == summary["all"]["exchanged"],
            file,
        )


def main(output, render):
    load = lambda name: json.loads((PUB / (name + ".json")).read_text())
    data = prepared()
    evaluation = {k: v["models"] for k, v in load("results").items()}
    report = dict(
        greedy=audit_replies(data, "evaluation", evaluation),
        marginal=audit_replies(data, "marginal", load("marginal")["groups"]),
        proposal_examples=audit_search(data, load("search")),
    )
    pairs = load("pairs")
    audit_pairs(pairs, load("exchange"))
    report["disjoint_pairs"] = pairs["pairs"]
    if render:
        render.mkdir(exist_ok=True, parents=True)
        expected = load("example")
        group = data["lists"]
        index, row = next(
            (i, r)
            for i, r in enumerate(group["rows"])
            if r[1] == expected["page"] and r[2] == expected["index"]
        )
        chosen = {}
        for arm in ("plain", "assembly", "lists"):
            tag = "ink/" + arm
            records = bound_rows(
                "evaluation",
                f"lists-ink-{arm}.json.gz",
                evaluation["lists"][tag]["outputs"],
            )
            chosen[arm] = records[index]["original"]
        image = sample_pixels(
            "list-composition", row, group, chosen, paths_for("ink", MODEL), render
        )
        require(image == expected, "Standalone inference changed")
        for arm in ("plain", "lists"):
            file = "list-composition-" + arm + ".png"
            require(
                sha(render / "examples" / file) == sha(PUB / file),
                "Different standalone pixels",
            )
    encoded = json.dumps(report, indent=2) + "\n"
    if output:
        output.write_text(encoded)
    else:
        print(encoded, end="")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--render-example",
        type=Path,
        help="Write the illustrated input and output PNGs in a local folder",
    )
    args = parser.parse_args()
    main(args.output, args.render_example)
