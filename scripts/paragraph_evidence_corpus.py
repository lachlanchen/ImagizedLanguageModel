"""Picture-only paragraph routing trials with retained source assignments."""

from collections import defaultdict
from itertools import product
from pathlib import Path
import re

from scripts.grounded_proposition_corpus import noun
from scripts.prepare_screened_span_list import split
from scripts.prepare_visual_facts import ROOT
from scripts.span_reservation import reservation
from scripts.test_page_writer_clause_selection import paired_clauses
from scripts.train_page_writer_requests import VARIED, NEW, load_records

WORK = ROOT / "artifacts/paragraph_evidence_v1"
PUBLIC = ROOT / "references/evidence/paragraph_evidence_20261124"


def section_pair(text):
    # Choose a boundary without using the question or an answer annotation.
    ends = [
        m.end()
        for m in re.finditer(r"[。！？；]", text)
        if 12 <= m.end() <= len(text) - 12
    ]
    if not ends:
        return None
    cut = min(ends, key=lambda n: abs(2 * n - len(text)))
    return [text[:cut], text[cut:]]


def natural(split_name):
    result = []
    for x in split(split_name):
        if x["group"] not in ("cmrc", "drcd"):
            continue
        parts = section_pair(x["p"])
        if parts is None:
            continue
        located = [any(r in p for r in x["refs"]) for p in parts]
        if sum(located) != 1:
            continue
        result.append(
            dict(
                type=x["group"],
                board=x["key"],
                q=x["q"],
                parts=parts,
                right=located.index(True),
                refs=x["refs"],
                year="",
                rival="",
            )
        )
    return result


def dual(data, *, shared=False, independent=False, fresh=False):
    groups = defaultdict(list)
    for x in data:
        if x["task"] == "dual":
            groups[x["board"]].append(x)
    out = []
    guard = reservation()
    for key, vs in groups.items():
        if len(vs) != 4 or vs[0]["p"].count("　") != 1:
            continue
        old = sorted((v for v in vs if v["edit"] == 0), key=lambda v: v["request"])
        if len(old) != 2:
            raise ValueError("Incomplete dual source")
        a, b = old
        if shared and (
            a["r"] == b["r"] or a["o"] == b["o"] or b["s"] in b["o"] or a["s"] in b["o"]
        ):
            continue
        left, right = a["p"].split("　")
        old_year = [a["year"], b["year"]]
        if independent:
            alternatives = [str(int(old_year[0]) + 3), str(int(old_year[1]) + 7)]
            if len(set(old_year + alternatives)) != 4 or any(
                y in a["p"] for y in alternatives
            ):
                continue
            assignments = list(product(*zip(old_year, alternatives)))
        else:
            assignments = [old_year, old_year[::-1]]
        group = []
        for edit, years in enumerate(assignments):
            parts = [
                left.replace(old_year[0] + "年", years[0] + "年", 1),
                right.replace(old_year[1] + "年", years[1] + "年", 1),
            ]
            if shared:
                parts[1] = parts[1].replace(b["s"], a["s"], 1)
            if (shared or independent) and guard.overlaps("　".join(parts)):
                break
            for j, v in enumerate(old):
                name = a["s"] if shared else v["s"]
                expected = (
                    v["y"]
                    .replace(v["s"], name, 1)
                    .replace(v["year"] + "年。", years[j] + "年。")
                )
                # For the original four cases preserve each actual training question.
                original = next(
                    t for t in vs if t["request"] == j and t["edit"] == min(edit, 1)
                )
                qs = (
                    [p.format(noun(name, v["r"])) for p in NEW]
                    if fresh
                    else [original["q"].replace(v["s"], name, 1)]
                )
                for k, q in enumerate(qs):
                    group.append(
                        dict(
                            type="shared"
                            if shared
                            else ("independent" if independent else "dual"),
                            board=key,
                            parts=parts,
                            right=j,
                            q=q,
                            refs=[expected],
                            year=years[j],
                            rival=years[1 - j],
                            wording=k,
                            request=j,
                            edit=edit,
                        )
                    )
        else:
            out.extend(group)
    return out


def clause_cases(held):
    raw = paired_clauses(held)
    originals = {x["key"]: x["p"] for x in held if x["task"] == "clause"}
    out = []
    for i, v in enumerate(raw):
        a, b = [originals[x["key"]] for x in raw[i // 4 * 4 : i // 4 * 4 + 2]]
        parts = [a, b] if i % 4 < 2 else [b, a]
        assert "　".join(parts) == v["p"]
        out.append(
            dict(
                type="clause",
                board=str(i // 4),
                q=v["q"],
                parts=parts,
                right=parts.index(originals[v["key"]]),
                refs=[v["y"]],
                year="",
                rival="",
            )
        )
    return out


def corpus():
    train = load_records(VARIED, "training")
    fresh = load_records(VARIED, "fresh")
    held = load_records(VARIED, "held")
    training = dual(train) + dual(train, shared=True) + natural("train")
    test = (
        dual(fresh, fresh=True)
        + dual(fresh, shared=True, fresh=True)
        + dual(fresh, independent=True, fresh=True)
    )
    test += dual(held) + clause_cases(held) + natural("fresh")
    # The earlier held group used one wording and has separate board identities.
    for v in test:
        if v["type"] == "dual" and v["q"].startswith("请介绍"):
            v["type"] = "earlier"
    return training, test
