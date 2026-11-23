"""Compare measured training preferences and free answers on independent works."""

from collections import Counter, defaultdict
import json

from scripts.audit_interval_teacher import save_image
from scripts.evaluate_pointing_causal_writer import load_streams
from scripts.evaluate_stopped_book_preference import ARMS
from scripts.mine_stopped_book_replies import PUBLIC, WORK
from scripts.reserve_preference_books import PAGES, literary_cases
from scripts.train_causal_glyph_writer import in_cells, record, sha
from scripts.train_typeface_writer import FACES, Face


def prepare():
    scores = json.loads((PUBLIC / "result.json").read_text())["scores"]
    gaps = json.loads((PUBLIC / "training_gaps.json").read_text())
    series = []
    for arm in ARMS:
        groups = gaps[arm]["groups"]
        series.append([round(100 * groups[g]["preferred"] / groups[g]["n"], 1)
                       for g in ("all", "shorter", "equal", "longer")])
    table = [["Every answer correct", "Passages", "Released", "Gold replay", "Preference"]]
    for label, group in (("Six new works", "fresh_four"), ("Four earlier works", "prior_four"),
                         ("BiPaR novels", "book_four"), ("XQuAD", "xquad_regular_four"),
                         ("CMRC", "trial_regular"), ("Historical forms", "archive_familiar")):
        table.append([label, str(scores["ordinary"][group]["boards"])]
                     + [str(scores[a][group]["complete"]) for a in ARMS])
    cases = literary_cases()
    pages = json.loads(PAGES.read_text())
    face = Face(FACES[0])
    alphabet = defaultdict(set)
    for page in pages.values():
        for c in page["page"]:
            alphabet[face.cell(c)].add(c)
    examples = PUBLIC / "examples"
    examples.mkdir(exist_ok=True)
    strips, records, transitions = [], {}, {}
    for lang, desired, tag in (("zh", True, "success"), ("en", False, "failure")):
        group = "fresh_" + lang
        streams = {}
        for arm in ARMS:
            path = WORK / "evaluation" / arm / (group + ".npz")
            assert sha(path) == scores[arm][group]["sha256"]
            ys, stops = load_streams(path)
            streams[arm] = list(zip(ys, stops, strict=True))
        counts = {arm: Counter() for arm in ARMS[:-1]}
        chosen = None
        for i, case in enumerate(cases[group]):
            candidate = streams["preference"][i]
            exact = bool(case.grade(*candidate)[2])
            for arm in counts:
                baseline = streams[arm][i]
                was_exact = case.grade(*baseline)[2]
                status = "gained" if exact and not was_exact else "lost" if was_exact and not exact else "both_exact" if exact else "both_miss"
                counts[arm][status] += 1
                if status == "both_miss" and baseline == candidate:
                    counts[arm]["same_wrong_pixels_and_stop"] += 1
            if chosen is None and exact == desired and candidate[0]:
                chosen = (case, candidate, i)
        transitions[lang] = counts
        if chosen is None:
            continue
        case, (y, stop), i = chosen
        k, n = case.question_id.split("/")
        v = pages[k]
        q = v["questions"][int(n)]
        ref = next((r for r in q["refs"] if face(r) == y), q["refs"][0])
        start = max(0, min(v["page"].index(ref)-6, len(v["page"])-24))
        context = v["page"][start:start + max(24, len(ref)+6)]
        for suffix, raw, label in (
            ("excerpt", case.p[start*128:(start+len(context))*128], tag.capitalize() + " · excerpt"),
            ("question", case.q, "Pictured question"), ("output", y, "Writer reply"),
            ("reference", face(ref), "Accepted reply")):
            if desired and suffix == "reference":
                continue
            name = tag + "_" + suffix
            save_image(PUBLIC / (name + ".png"), raw)
            strips.append([name, label])
        for name, raw in (("page", case.p), ("question", case.q), ("output", y), ("reference", face(ref))):
            save_image(examples / (tag + "_" + name + ".png"), raw)
        options = [sorted(alphabet[x]) for x in in_cells(y)]
        text = "".join(opts[0] for opts in options) if all(len(opts)==1 for opts in options) else None
        records[tag] = dict(id=case.question_id, title=v["title"], question=q[lang],
                            refs=q["refs"], reply_if_unique=text, stopped=bool(stop), exact=desired,
                            source_excerpt=context)
    record(PUBLIC / "transitions.json", transitions)
    record(PUBLIC / "figure.json", dict(series=series, table=table, strips=strips, examples=records))


if __name__ == "__main__":
    prepare()
