"""Measure stopped replies on independent works and the frozen development sets."""

from collections import Counter, defaultdict
import json
import os

from scripts.audit_bilingual_novel import first_divergence
from scripts.audit_interval_teacher import save_image
from scripts.evaluate_bilingual_novel_writer import corpus, generate_archives, recount
from scripts.evaluate_pointing_causal_writer import load_streams
from scripts.reserve_first_glyph_books import PAGES, reserved
from scripts.train_bilingual_novel_writer import WORK as ORDINARY, PUBLIC as OLD_PUBLIC
from scripts.train_causal_glyph_writer import record, sha
from scripts.train_first_glyph_book import WORK, PUBLIC
from scripts.train_span_choice import gpu_owner


def evaluate_cases(cases, path):
    replies, ends = load_streams(path)
    rows = []
    for c, y, end in zip(cases, replies, ends, strict=True):
        first = bool(y) and any(y[:128] == r[:128] for r in (*c.source, *c.canonical))
        exact = bool(c.grade(y, end)[2])
        error = first_divergence(c, y, end)
        assert exact == (error == "exact") and (not exact or first)
        rows.append(
            dict(
                board=c.board,
                id=c.question_id,
                first=first,
                exact=exact,
                stopped=bool(end),
                error=error,
            )
        )
    return rows


def summary(rows):
    first = sum(r["first"] for r in rows)
    exact = sum(r["exact"] for r in rows)
    return dict(
        n=len(rows),
        first=first,
        exact=exact,
        subsequent_misses=first - exact,
        stopped=sum(r["stopped"] for r in rows),
        outcomes=dict(Counter(r["error"] for r in rows)),
    )


def ordinary_archives(groups, fresh, folder):
    old = ORDINARY / "evaluation/bilingual"
    scores = json.loads((OLD_PUBLIC / "result.json").read_text())["scores"]["bilingual"]
    for k in groups.keys() - fresh.keys():
        path = old / (k + ".npz")
        assert sha(path) == scores[k]["sha256"]
        link = folder / path.name
        if not link.exists():
            os.link(path, link)
        assert sha(link) == scores[k]["sha256"]
    generate_archives(fresh, ORDINARY / "bilingual/page", folder)


def examples(groups, scores, records):
    out = PUBLIC / "examples"
    out.mkdir(exist_ok=True)
    selected = {}
    for label, mode in (("complete", True), ("incomplete", False)):
        keys = [k for k, ok in scores["first"]["fresh_by_page"].items() if ok == mode]
        if not keys:
            continue
        selected[label] = keys[0]
    first_hit = next((r for r in records["first"]["fresh_zh"] if r["exact"]), None)
    first_miss = next((r for r in records["first"]["fresh_en"] if not r["exact"]), None)
    for tag, r in (("answer_hit", first_hit), ("answer_miss", first_miss)):
        if r:
            selected[tag] = r["id"]
    for tag, key in selected.items():
        lang = "en" if tag == "answer_miss" else "zh"
        items = groups["fresh_" + lang]
        idx = next(
            i for i, c in enumerate(items) if c.question_id == key or c.board == key
        )
        c = items[idx]
        for kind, raw in (("question", c.q), ("reference", c.source[0])):
            save_image(out / f"{tag}_{kind}.png", raw)
        for arm in scores:
            ys, _ = load_streams(WORK / "evaluation" / arm / ("fresh_" + lang + ".npz"))
            if ys[idx]:
                save_image(out / f"{tag}_{arm}.png", ys[idx])
    return selected


def run():
    groups = corpus()
    fresh = reserved()
    groups.update(fresh)
    titles = {k: v["title"] for k, v in json.loads(PAGES.read_text()).items()}
    scores, records = {}, {}
    for arm in ("ordinary", "first", "later"):
        folder = WORK / "evaluation" / arm
        folder.mkdir(parents=True, exist_ok=True)
        if arm == "ordinary":
            ordinary_archives(groups, fresh, folder)
        else:
            generate_archives(groups, WORK / arm / "page", folder)
        stats, _ = recount(groups, folder)
        reports = {
            key: evaluate_cases(groups[key], folder / (key + ".npz"))
            for key in ("fresh_en", "fresh_zh", "fresh_erased", "book_en", "book_zh")
        }
        records[arm] = reports
        for k, rows in reports.items():
            stats[k].update(summary(rows))
        rows = reports["fresh_en"] + reports["fresh_zh"]
        by_board = defaultdict(list)
        for r in rows:
            by_board[r["board"]].append(r["exact"])
        assert all(len(flags) == 4 for flags in by_board.values())
        four = {k: all(flags) for k, flags in by_board.items()}
        stats["fresh_by_page"] = four
        stats["fresh_four"] = dict(complete=sum(four.values()), boards=len(four))
        stats["fresh_pooled"] = summary(rows)
        stats["fresh_by_work"] = {
            title: dict(
                **summary([r for r in rows if titles[r["board"]] == title]),
                four=sum(v for k, v in four.items() if titles[k] == title),
                pages=sum(titles[k] == title for k in four),
            )
            for title in dict.fromkeys(titles.values())
        }
        scores[arm] = stats
        print(arm, stats["fresh_four"], stats["fresh_pooled"], flush=True)
    candidate = scores["first"]
    criteria = {
        key: candidate[group][key]
        > max(scores[a][group][key] for a in ("later", "ordinary"))
        for group, key in (("fresh_four", "complete"), ("fresh_pooled", "first"))
    }
    criteria.update(
        cmrc=candidate["trial_regular"]["complete"] >= 14,
        historical=candidate["archive_familiar"]["complete"] >= 99,
    )
    record(PUBLIC / "case_scores.json", records)
    record(
        PUBLIC / "result.json",
        dict(
            scores=scores,
            criteria=criteria,
            selected=all(criteria.values()),
            examples=examples(groups, scores, records),
        ),
    )


if __name__ == "__main__":
    with gpu_owner():
        run()
