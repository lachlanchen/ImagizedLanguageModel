"""Prepare native chart values and original unseen-book question and response pixels."""

import json
from scripts.prepare_book_participant_questions import PUBLIC, WORK
from scripts.train_typeface_writer import Face, FACES
from scripts.figure_role_change_writer import png
from scripts.train_causal_glyph_writer import record, sha


def main():
    read = lambda p: json.loads(p.read_text())
    keys = [w + "/" + l for w in ("ha_li_bo_te", "san_ti") for l in ("zh", "en")]

    def counts(filename):
        return read(PUBLIC / (filename + ".json"))["counts"]

    series = []
    for arm in ("literal", "independent", "aligned"):
        d = counts(arm + "_reserved")
        series.append(
            [round(100 * d[k]["restricted_both"] / d[k]["pages"], 1) for k in keys]
        )
    proposals = counts("person_reserved")
    table = [
        ["Writer", "Chinese · one", "Chinese · both", "English · one", "English · both"]
    ]
    for label, file, individual, both in [
        ("Earlier clause writer", "benchmark_previous", "exact", "both"),
        ("All page strips", "aligned_reserved", "open_individual", "open_both"),
        (
            "Proposals + control",
            "person_reserved",
            "independent_correct",
            "independent_both",
        ),
        ("Proposals + reader", "person_reserved", "aligned_correct", "aligned_both"),
        ("Text Qwen3-8B", "benchmark_text", "exact", "both"),
    ]:
        d = counts(file)
        vals = []
        for l in ("zh", "en"):
            c = d["all/" + l]
            vals.extend(
                [f"{c[individual]} / {c['questions']}", f"{c[both]} / {c['pages']}"]
            )
        table.append([label] + vals)
    rows = read(WORK / "reserved.json")
    path = WORK / "person_proposal_replies.json"
    assert sha(path) == read(PUBLIC / "person_reserved.json")["sha256"]
    outs = read(path)
    face = Face(FACES[0])
    strips = []
    for i, w in enumerate(("ha_li_bo_te", "san_ti")):
        idx = next(j for j, r in enumerate(rows) if r["work"] == w)
        r, case = rows[idx], outs[idx]
        assert r["page_sha"] == case["key"]
        for h in (0, 1):
            label = f"example{i}_{h}"
            q = r["questions"][h]["zh"]
            answer = r["names"][h]
            png(face(q), PUBLIC / (label + "_q.png"))
            png(face(answer), PUBLIC / (label + "_gold.png"))
            raw = bytes.fromhex(case["replies"][h]["images"]["aligned"])
            png(raw, PUBLIC / (label + "_out.png"))
            strips.append(
                dict(
                    base=label,
                    work=["Harry Potter", "Three-Body"][i],
                    correct=raw == face(answer),
                )
            )
    successes = [
        i
        for i, c in enumerate(outs)
        if all(v["correct"]["aligned"] for v in c["replies"][:2])
    ]
    if successes:
        i = successes[0]
        for h in (0, 1):
            base = f"success{h}"
            q = rows[i]["questions"][h]
            png(face(q["zh"]), PUBLIC / (base + "_q.png"))
            png(face(rows[i]["names"][h]), PUBLIC / (base + "_gold.png"))
            png(
                bytes.fromhex(outs[i]["replies"][h]["images"]["aligned"]),
                PUBLIC / (base + "_out.png"),
            )
            strips.append(dict(base=base, work="First full success", correct=True))
    record(PUBLIC / "figure.json", dict(table=table, series=series, strips=strips))


if __name__ == "__main__":
    main()
