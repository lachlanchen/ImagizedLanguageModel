"""Match held turns to the additional independent CoQA answers and official scorer."""

from collections import defaultdict
import gzip
import hashlib
import json
import runpy
from scripts.prepare_dialogue_ink import ROOT, PUB, WORK, sha, dump, clean


def official_index():
    raw = ROOT / "data/raw/coqa/dev-official.json"
    code = ROOT / "data/raw/coqa/official_eval.py"
    if (
        sha(raw) != "dfa367a9733ce53222918d0231d9b3bedc2b8ee831a2845f62dfc70701f2540a"
        or sha(code)
        != "d42f70d686abc724a9169947a487f4509087ee69804277cc73950682cb700b4d"
    ):
        raise ValueError("Official reference or scorer checksum mismatch")
    evaluator = runpy.run_path(str(code))["CoQAEvaluator"](str(raw))
    rows = {}
    stories = {}
    for story in json.loads(raw.read_text())["data"]:
        text = clean(story["story"])
        key = hashlib.sha256(text.encode()).hexdigest()
        stories[key] = text
        history = ""
        for j, (q, a) in enumerate(
            zip(story["questions"], story["answers"], strict=True)
        ):
            question, answer = clean(q["input_text"]), clean(a["input_text"])
            case = (key, j, question, " ".join(history.split()[-48:]), answer)
            marker = (story["id"], q["turn_id"])
            if (
                case in rows
                and evaluator.gold_data[marker] != evaluator.gold_data[rows[case]]
            ):
                raise ValueError("Ambiguous official references")
            rows[case] = marker
            history = question + " " + answer
    return evaluator, rows, stories, raw, code


def score():
    evaluator, rows, stories, raw, code = official_index()
    norm = evaluator.normalize_answer
    out = {}
    for name, folder in (
        ("plain", "evaluation"),
        ("word_code", "word_code_evaluation"),
    ):
        recs = json.loads(
            gzip.decompress((WORK / folder / "decisions.json.gz").read_bytes())
        )
        scores = defaultdict(lambda: defaultdict(list))
        counts = defaultdict(int)
        for r in recs:
            if r["group"] != "dev":
                continue
            marker = rows[
                r["story"], r["turn"], r["question"], r["history"], r["answer"]
            ]
            refs = evaluator.gold_data[marker]
            sources = [
                " " + norm(s) + " "
                for s in (stories[r["story"]], r["question"], r["history"])
            ]
            missing = lambda a: len(norm(a).split()) >= 2 and not any(
                (" " + norm(a) + " ") in s for s in sources
            )
            groups = ["all"]
            if r["strict"]:
                groups.append("strict")
                if missing(r["answer"]):
                    groups.append("normalized_absent_primary")
                if all(missing(a) for a in refs):
                    groups.append("all_references_absent")
            for g in groups:
                counts[g] += len(refs)
                for arm, e in r["out"].items():
                    sc = evaluator.compute_turn_score(*marker, e["text"])
                    accepted = any(norm(a) == norm(e["text"]) for a in refs)
                    scores[g][arm].append(
                        (sc["em"], sc["f1"], accepted)
                        if e["stopped"]
                        else (0.0, 0.0, False)
                    )
        out[name] = {}
        for g, vals in scores.items():
            n = len(vals["writer"])
            out[name][g] = dict(
                turns=n,
                annotations=counts[g],
                arms={
                    k: dict(
                        official_em=sum(x[0] for x in v) / n,
                        official_f1=sum(x[1] for x in v) / n,
                        any_exact=sum(x[2] for x in v),
                    )
                    for k, v in vals.items()
                },
            )
    dump(
        PUB / "additional_references.json",
        dict(original=sha(raw), scorer=sha(code), results=out),
    )
    print({g: v["arms"]["writer"] for g, v in out["word_code"].items()})


if __name__ == "__main__":
    score()
