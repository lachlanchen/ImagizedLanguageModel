"""Recount strict answers and check the poetry sample against inherited training ink."""

from collections import defaultdict
from functools import lru_cache
import numpy as np
from scripts.prepare_visual_action_meaning import ROOT, WORK, read, dump, sha, table
from scripts.prepare_pronoun_writer import window
from scripts.train_typeface_writer import Face, FACES


def main():
    p = WORK / "evaluation"
    cases = read(p / "cases.json")
    rows = read(p / "lines.json")
    margins = np.load(p / "margins.npz")
    scores = read(p / "result.json")
    recount = {}
    subgroup = {}
    for name, records in cases["cases"].items():
        s = np.zeros(3, int)
        correct_word = np.zeros(3, int)
        parts = defaultdict(lambda: [0, 0])
        for j, (ids, _) in enumerate(records):
            ok = np.ones(3, bool)
            vw = True
            for k in range(4):
                side = k // 2
                d = margins[name][j, k, :, 0]
                for rev in (False, True):
                    ok &= np.where(d == 0, side == int(rev), d > 0)
                r = rows[ids[side]]
                vw &= (
                    r["arc"][1] == r["focus"][1] and r["words"][1] in r["registered"][1]
                )
            s += ok
            correct_word += ok & vw
            if name in ("literal", "webster", "editorial"):
                if "held" not in locals():
                    held = read(
                        ROOT / "references/evidence/pictured_action_lookup_v1/held.json"
                    )
                part = parts[held[ids[0]]["split"]]
                part[0] += 1
                part[1] += int(vw and ok[1])
        assert (
            s.tolist() == scores["groups"][name]["line"]
            and correct_word.tolist() == scores["groups"][name]["written"]
        )
        recount[name] = correct_word.tolist()
        subgroup[name] = dict(parts)
    face = Face(FACES[0])
    glyphs = {}

    @lru_cache(None)
    def cell(c):
        return glyphs.setdefault(face.cell(c), len(glyphs))

    def key(s):
        return tuple(map(cell, s))

    older = read(ROOT / "artifacts/visual_word_senses_v1/train.json")
    allwindows = {
        key(left.rjust(8) + c + right.ljust(8)) for _, _, _, left, c, right, _ in older
    }
    allwindows.update(
        key(r[0]) for r in read(ROOT / "artifacts/pronoun_writer_v1/train.json")
    )
    syntax = read(ROOT / "artifacts/visual_clause_roles_v1/train.json")
    pages = {key(r["page"]) for r in syntax}
    allwindows.update(key(window(r["page"], i)) for r in syntax for i in r["focus"])
    src = ROOT.parent / "ZhJpBook/books/tangshi-sanbai/markdown/wenyan.md"
    book = src.read_text()
    reference = ROOT / "data/raw/visual_action_semantics/tang-translations.txt"
    translated = reference.read_text()
    import re

    poems = {
        n: s for n, s in re.findall(r"(?ms)^(\d{3})\n(.*?)(?=^\d{3}\n|\Z)", translated)
    }
    matches = []
    for r in table("tang.tsv"):
        page = r["line"]
        assert page in book and r["poem"] in poems
        focus = [page.index(r[x]) for x in ["verb", "object"]]
        matches.append(
            dict(
                line=page,
                poem=r["poem"],
                translation_title=poems[r["poem"]].splitlines()[2],
                exact_page_in_source_fit=key(page) in pages,
                source_window_overlap=[
                    key(window(page, i)) in allwindows for i in focus
                ],
            )
        )
    group = scores["groups"]
    promote = (
        all(
            group[n]["written"][1] > max(group[n]["written"][0], group[n]["written"][2])
            for n in ("editorial", "webster", "tang_webster")
        )
        and group["literal"]["written"][1] >= 0.98 * 703
    )

    def clean_row(r):
        return not r["exact_page_in_source_fit"] and not any(r["source_window_overlap"])

    clean_tang = sum(
        clean_row(a) and clean_row(b) for a, b in zip(matches[::2], matches[1::2])
    )
    fresh = read(WORK / "pu_evaluation/result.json")["groups"]
    fresh_writes = [
        fresh["webster_" + arm]["written"] for arm in ("parent", "aligned", "exchanged")
    ]
    fresh_gain = fresh_writes[1] > max(fresh_writes[0], fresh_writes[2])
    dump(
        p / "audit.json",
        dict(
            written=recount,
            by_split=subgroup,
            poetry_training_overlap=matches,
            distinct_poems=len({t["poem"] for t in matches}),
            distinct_lines=len(set(t["line"] for t in matches)),
            chinese_edition_sha256=sha(src),
            english_reference_sha256=sha(reference),
            original_numeric_rule_pass=bool(promote),
            both_tang_lines_clean=clean_tang,
            fresh_writes=fresh_writes,
            promotion_rule_pass=bool(promote and fresh_gain),
            trained_or_inferred_during_recount=False,
        ),
    )


if __name__ == "__main__":
    main()
