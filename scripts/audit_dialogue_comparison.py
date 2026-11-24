"""Recount stopped first-reference results and publish answer hashes for the three controls."""

from collections import Counter
import gzip
import hashlib
import json
from scripts.audit_dialogue_ink import normalize
from scripts.dialogue_ink_data import read_data
from scripts.evaluate_dialogue_ink import exchanges
from scripts.prepare_dialogue_ink import WORK, PUB, dump, sha


def audit():
    data = read_data()
    groups = {g: [r for r in data["rows"] if r[0] == g] for g in ("reserved", "dev")}
    swap = {g: exchanges(rows) for g, rows in groups.items()}
    known = {tuple(normalize(r[5])) for r in data["rows"] if r[0] == "train"}
    scores = {}
    for f in ("token_comparison", "cold_comparison"):
        j = json.loads((PUB / (f + ".json")).read_text())["scores"]
        for g, v in j.items():
            scores.setdefault(g, {}).update(v)
    receipts = {}
    for label, folder in [
        ("subword", "subword_evaluation"),
        ("ink", "matched/ink"),
        ("byte", "matched/byte"),
    ]:
        path = WORK / folder / "decisions.json.gz"
        rows = json.loads(gzip.decompress(path.read_bytes()))
        ids = set()
        compact = []
        counts = Counter()
        for r in rows:
            g, i = r["group"], r["index"]
            src = groups[g][i]
            if (g, i) in ids or src[1:6] != [
                r[k] for k in ("story", "turn", "question", "history", "answer")
            ]:
                raise ValueError("Duplicated or mismatched turn")
            ids.add((g, i))
            reference = r["answer"]
            gold = normalize(reference)
            strict = (
                len(reference.split()) >= 2
                and tuple(gold) not in known
                and all(
                    reference.casefold() not in s.casefold()
                    for s in (data["pages"][src[1]], src[3], src[4])
                )
            )
            if strict != r["strict"] or [
                r["question_control"],
                r["passage_control"],
            ] != [int(x[i]) for x in swap[g]]:
                raise ValueError("Different test population or exchange")
            ledger = []
            for arm in ("writer", "exchanged_question", "exchanged_passage"):
                e = r["out"][arm]
                output = e["text"]
                pred = normalize(output)
                hit = bool(e["stopped"] and pred == gold)
                literal = bool(e["stopped"] and output == reference)
                if hit != e["exact"] or literal != e["pixels"]:
                    raise ValueError("Different exact answer score")
                for k in ["all", "strict"] if strict else ["all"]:
                    counts[g, k, arm] += hit
                ledger.append(
                    [
                        hashlib.sha256(output.encode()).hexdigest(),
                        e["stopped"],
                        hit,
                        literal,
                    ]
                )
            compact.append(
                [
                    g,
                    i,
                    src[1],
                    r["turn"],
                    strict,
                    r["question_control"],
                    r["passage_control"],
                    ledger,
                ]
            )
        if len(ids) != sum(len(group) for group in groups.values()):
            raise ValueError("Missing held turns")
        for g in groups:
            for k in ("all", "strict"):
                expected = scores[g][label][k]
                for arm in ("writer", "exchanged_question", "exchanged_passage"):
                    if (
                        counts[g, k, arm]
                        != expected["exact" if arm == "writer" else arm + "_exact"]
                    ):
                        raise ValueError("Score mismatch")
        file = PUB / (label + "_comparison_decisions.json.gz")
        file.write_bytes(
            gzip.compress(json.dumps(compact, separators=(",", ":")).encode(), mtime=0)
        )
        receipts[label] = dict(
            held=len(rows), streams=3 * len(rows), outputs=sha(path), public=sha(file)
        )
    dump(PUB / "comparison_audit.json", receipts)


if __name__ == "__main__":
    audit()
