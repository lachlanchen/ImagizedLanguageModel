"""Publish the literal test responses and the first complete Chinese Guoyu example."""

import gzip
import json

from scripts.figure_role_change_writer import png
from scripts.train_optical_role_adapter import PUBLIC, WORK, read, record, sha
from scripts.train_typeface_writer import FACES, Face


def main():
    arms = ("aligned", "independent", "parent", "generic", "text")
    ledger = []
    aligned = {}
    for arm in arms:
        src = WORK / (arm + "_responses.json")
        assert sha(src) == read(PUBLIC / (arm + "_scores.json"))["response_sha"]
        for r in read(src):
            if r["group"] == "novels":
                continue
            ledger.append({"reader": arm, **r})
            if arm == "aligned" and r["group"] == "guoyu" and r["lang"] == "zh":
                aligned.setdefault(r["key"], []).append(r)
    dest = PUBLIC / "classical_answers.jsonl.gz"
    with (
        dest.open("wb") as f,
        gzip.GzipFile(filename="", mode="wb", fileobj=f, mtime=0) as compressed,
    ):
        for r in ledger:
            compressed.write(
                (
                    json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n"
                ).encode()
            )
    face = Face(FACES[0])
    folder = PUBLIC / "example"
    folder.mkdir(exist_ok=True)
    for item in read(WORK / "guoyu.json"):
        rows = aligned[item["key"]]
        assert {(r["world"], r["h"]) for r in rows} == {(0, 0), (0, 1), (1, 0), (1, 1)}
        if not all(r["correct"] for r in rows):
            continue
        detail = []
        for row in rows:
            w, h = row["world"], row["h"]
            page = item["pages"][w]
            question = item["qs"][h]["zh"]
            target = item["qs"][w ^ h]["answer"]
            actual = bytes.fromhex(row["reply"])
            assert actual == face(target)
            n = len(actual) // 128
            matching = sorted(
                {
                    page[j : j + n]
                    for j in range(len(page) - n + 1)
                    if face(page[j : j + n]) == actual
                }
            )
            detail.append(
                dict(
                    world=w,
                    question=question,
                    source=target,
                    output_pixel_matches=matching,
                )
            )
            png(face(page), folder / f"page{w}.png")
            png(face(question), folder / f"question{h}.png")
            png(actual, folder / f"answer{w}_{h}.png")
        record(
            folder / "selection.json",
            dict(
                selection="First complete Chinese case in frozen source order",
                key=item["key"],
                pages=item["pages"],
                outputs=detail,
            ),
        )
        break
    record(
        PUBLIC / "classical_answers.json",
        dict(
            file=dest.name,
            sha256=sha(dest),
            rows=len(ledger),
            format="One JSON object per response. For image readers reply is packed black ink, hexadecimal, 128 bytes per 32 by 32 cell in row-major order (MSB first). The text reference has a literal text reply.",
            groups=["guoyu", "zuo", "shishuo"],
            arms=list(arms),
        ),
    )


if __name__ == "__main__":
    main()
