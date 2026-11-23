"""Score free image answers, including bilingual questions on six reserved works."""

from collections import defaultdict
import json
import os

from scripts.evaluate_bilingual_novel_writer import corpus, generate_archives, recount
from scripts.evaluate_first_glyph_books import evaluate_cases, summary
from scripts.evaluate_pointing_causal_writer import load_streams
from scripts.mine_stopped_book_replies import WORK, PUBLIC, PARENT
from scripts.reserve_first_glyph_books import reserved as previous_cases
from scripts.reserve_preference_books import literary_cases
from scripts.train_causal_glyph_writer import record, sha
from scripts.train_first_glyph_book import WORK as PREVIOUS, PUBLIC as PREVIOUS_PUBLIC
from scripts.train_span_choice import gpu_owner
from scripts.train_typeface_writer import Face, FACES

ARMS = ("ordinary", "replay", "preference")


def evaluate():
    groups = corpus()
    previous = previous_cases()
    groups.update({"prior_" + k.removeprefix("fresh_"): v for k, v in previous.items()})
    fresh = literary_cases()
    groups.update(fresh)
    old = json.loads((PREVIOUS_PUBLIC / "result.json").read_text())["scores"][
        "ordinary"
    ]
    titles = {
        k: v["title"]
        for k, v in json.loads((PUBLIC / "reserved_books.json").read_text()).items()
    }
    scores, detail = {}, {}
    reviewed = json.loads((PUBLIC / "label_review.json").read_text())
    excluded = {tuple(k) for k in reviewed["exclude"]}
    face = Face(FACES[0])
    training_wrongs = {
        face(v["wrong"])
        for v in json.loads((WORK / "accepted.json").read_text())
        if (v["id"], v["lang"]) not in excluded
    }
    for arm in ARMS:
        folder = WORK / "evaluation" / arm
        folder.mkdir(exist_ok=True, parents=True)
        if arm == "ordinary":
            for name in groups.keys() - fresh.keys():
                prev_name = (
                    name.replace("prior_", "fresh_", 1)
                    if name.startswith("prior_")
                    else name
                )
                previous_path = PREVIOUS / "evaluation/ordinary" / (prev_name + ".npz")
                dest = folder / (name + ".npz")
                assert sha(previous_path) == old[prev_name]["sha256"]
                if not dest.exists():
                    os.link(previous_path, dest)
                assert sha(dest) == old[prev_name]["sha256"]
            generate_archives(fresh, PARENT, folder)
        else:
            generate_archives(groups, WORK / arm / "page", folder)
        stats, _ = recount(groups, folder)
        detail[arm] = {}
        for prefix in ("fresh", "prior"):
            boards = defaultdict(list)
            for k in ("zh", "en", "erased"):
                name = prefix + "_" + k
                cases = groups[name]
                path = folder / (name + ".npz")
                rows = evaluate_cases(cases, path)
                detail[arm][name] = rows
                stats[name].update(summary(rows))
                if k == "erased":
                    continue
                ys, _ = load_streams(path)
                stats[name]["miss_matching_supervised_rival"] = sum(
                    not r["exact"] and y in training_wrongs
                    for r, y in zip(rows, ys, strict=True)
                )
                for r in rows:
                    boards[r["board"]].append(r["exact"])
            assert all(len(flags) == 4 for flags in boards.values())
            stats[prefix + "_four"] = dict(
                complete=sum(all(v) for v in boards.values()), boards=len(boards)
            )
            if prefix == "fresh":
                stats["by_work"] = {}
                for title in dict.fromkeys(titles.values()):
                    selected = [v for k, v in boards.items() if titles[k] == title]
                    if selected:
                        stats["by_work"][title] = dict(
                            exact=sum(sum(v) for v in selected),
                            complete=sum(all(v) for v in selected),
                            boards=len(selected),
                        )
        scores[arm] = stats
        print(
            arm,
            stats["fresh_four"],
            stats["fresh_zh"]["exact"],
            stats["fresh_en"]["exact"],
            flush=True,
        )
    candidate = scores["preference"]
    criteria = dict(
        new_books=candidate["fresh_four"]["complete"]
        > max(scores[a]["fresh_four"]["complete"] for a in ("ordinary", "replay")),
        cmrc=candidate["trial_regular"]["complete"] >= 14,
        historical=candidate["archive_familiar"]["complete"] >= 99,
    )
    record(PUBLIC / "case_scores.json", detail)
    record(
        PUBLIC / "result.json",
        dict(scores=scores, criteria=criteria, selected=all(criteria.values())),
    )


if __name__ == "__main__":
    with gpu_owner():
        evaluate()
