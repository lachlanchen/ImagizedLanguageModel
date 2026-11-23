"""Recount the end-to-end error decomposition from the five physical reply archives."""

from collections import Counter, defaultdict
from functools import lru_cache
import json
from pathlib import Path
import shutil

from ilm.visual_lm.causal_glyph_writer import write_strip
from scripts.evaluate_page_evidence_writer import tensor
from scripts.evaluate_pointing_causal_writer import load_streams
from scripts.paragraph_evidence_corpus import WORK, PUBLIC
from scripts.prepare_visual_facts import Face, FONTS
from scripts.train_causal_glyph_writer import record, sha
from scripts.train_paragraph_evidence import records


def audit(folder=WORK, destination=PUBLIC):
    out = folder / "evaluation"
    measurements = json.loads((out / "result.json").read_text())
    choices = json.loads((out / "routing.json").read_text())
    assert sha(out / "routing.json") == measurements["source_choices_sha256"]
    rows = records(folder, "held")
    face = lru_cache(maxsize=10000)(Face((FONTS[0], 2)))
    destination.mkdir(parents=True, exist_ok=True)
    audited = {}
    samples = {}
    for branch, prefix in (("writer", ""), ("extractor", "extract_")):
        streams = {}
        for arm, result in measurements[branch].items():
            file = out / (prefix + arm + ".npz")
            assert sha(file) == result["streams"]
            streams[arm] = list(zip(*load_streams(file), strict=True))
        ids = [
            i
            for i, v in enumerate(rows)
            if branch == "writer" or v["type"] in ("drcd", "cmrc")
        ]
        assert all(len(s) == len(ids) for s in streams.values())
        stats = defaultdict(Counter)
        successes = defaultdict(list)
        invariance = defaultdict(list)
        for j, i in enumerate(ids):
            v = rows[i]
            group = v["type"]
            right = v["right"]
            ch = choices[i]
            assert right == ch["oracle"] and 1 - right == ch["rival"]
            refs = {face(r) for r in v["refs"]}
            passed = {
                arm: bool(stop and image in refs)
                for arm, stream in streams.items()
                for image, stop in [stream[j]]
            }
            for arm in streams:
                successes[group, v["board"], arm].append(passed[arm])
                stats[group][arm + "_correct"] += int(passed[arm])
            for arm in ("literal", "learned"):
                assert (
                    streams[arm][j]
                    == streams["oracle" if right == ch[arm] else "rival"][j]
                )
            c = stats[group]
            c["n"] += 1
            c["wrong_paragraph"] += int(ch["learned"] != right)
            c["oracle_errors"] += int(not passed["oracle"])
            c["neither_right"] += int(not passed["oracle"] and not passed["rival"])
            c["correct_paragraph_wrong_reply"] += int(
                ch["learned"] == right and not passed["learned"]
            )
            c["wrong_paragraph_wrong_reply"] += int(
                ch["learned"] != right and not passed["learned"]
            )
            c["learned_only_selects_right"] += int(
                ch["learned"] == right and ch["literal"] != right
            )
            c["literal_only_selects_right"] += int(
                ch["literal"] == right and ch["learned"] != right
            )
            if group == "independent":
                invariance[v["board"], v["request"], v["wording"], v["year"]].append(
                    streams["learned"][j]
                )
            if branch == "writer":
                fixed = not passed["whole"] and passed["learned"]
                fail = not passed["oracle"] and ch["learned"] == right
                key = (
                    "recovered"
                    if group == "dual" and fixed
                    else "residual"
                    if group == "dual" and fail
                    else None
                )
                if key and key not in samples:
                    samples[key] = dict(
                        row=i,
                        board=v["board"],
                        right=right,
                        question=v["q"],
                        reference=v["refs"][0],
                        selected=ch["learned"],
                        stop={a: s[j][1] for a, s in streams.items()},
                    )
                    for name, raw in [
                        ("whole", streams["whole"][j][0]),
                        ("learned", streams["learned"][j][0]),
                        ("reference", face(v["refs"][0])),
                    ]:
                        if raw:
                            write_strip(
                                tensor(raw).numpy(),
                                destination / (key + "_" + name + ".png"),
                            )
        for g, c in stats.items():
            assert (
                c["n"] - c["learned_correct"]
                <= c["wrong_paragraph"] + c["oracle_errors"]
            )
            assert (
                c["n"] - c["learned_correct"]
                == c["correct_paragraph_wrong_reply"] + c["wrong_paragraph_wrong_reply"]
            )
            for arm, data in measurements[branch].items():
                score = data["scores"][g]
                groups = [
                    s for (name, _, a), s in successes.items() if name == g and a == arm
                ]
                assert (
                    score["correct"],
                    score["questions"],
                    score["boards"],
                    score["complete"],
                ) == (c[arm + "_correct"], c["n"], len(groups), sum(map(all, groups)))
        if invariance:
            assert all(len(s) == 2 for s in invariance.values())
            stats["independent"]["distractor_pairs"] = len(invariance)
            stats["independent"]["unchanged_output"] = sum(
                a == b for a, b in invariance.values()
            )
        audited[branch] = dict(stats)
    record(destination / "audit.json", audited)
    record(destination / "samples.json", samples)
    for path in (folder / "fit.json", folder / "corpus.json", out / "result.json"):
        shutil.copyfile(path, destination / path.name)
    print(json.dumps(audited, indent=2))


if __name__ == "__main__":
    audit()
