"""Export literal values and all four original replies for the first Zuo excerpt."""

from scripts.figure_role_change_writer import png
from scripts.prepare_classical_argument_reader import PUBLIC, WORK, read, sha, record
from scripts.train_typeface_writer import Face, FACES


def main():
    methods = [
        ("original", "Previous reader"),
        ("independent", "Independent names"),
        ("aligned", "Assigned names"),
        ("without_question", "Generic question"),
        ("text", "Text Qwen3-8B"),
    ]
    series = []
    table = [
        [
            "Reader",
            "Zuo · Chinese",
            "Zuo · English",
            "Shishuo · Chinese",
            "Shishuo · English",
        ]
    ]
    for arm, label in methods:
        c = read(PUBLIC / (arm + "_scores.json"))["counts"]
        keys = [g + "/" + l for g in ("zuo", "shishuo") for l in ("zh", "en")]
        if arm in ("original", "independent", "aligned"):
            series.append(
                [round(100 * c[k]["individual"] / c[k]["questions"], 1) for k in keys]
            )
        table.append([label, *[f"{c[k]['whole']} / {c[k]['scenes']}" for k in keys]])
    raw = read(WORK / "reserved.json")[0]
    face = Face(FACES[0])
    path = WORK / "aligned_responses.json"
    assert sha(path) == read(PUBLIC / "aligned_scores.json")["response_sha"]
    saved = {
        (r["world"], r["h"]): bytes.fromhex(r["reply"])
        for r in read(path)
        if r["key"] == raw["key"] and r["lang"] == "zh"
    }
    strips = []
    for w, page in enumerate(raw["pages"]):
        png(face(page), PUBLIC / f"page{w}.png")
        for h, q in enumerate(raw["qs"]):
            base = f"reply{w}_{h}"
            png(face(q["zh"]), PUBLIC / (base + "_q.png"))
            png(face(raw["qs"][h ^ w]["answer"]), PUBLIC / (base + "_gold.png"))
            png(saved[w, h], PUBLIC / (base + "_out.png"))
            strips.append(dict(base=base, work=("As printed", "Names exchanged")[w]))
    record(PUBLIC / "figure.json", dict(table=table, series=series, strips=strips))


if __name__ == "__main__":
    main()
