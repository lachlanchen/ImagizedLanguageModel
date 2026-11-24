"""Render authored counterfactuals and reproduce held outputs from the physical checkpoint."""

from hashlib import sha256
import numpy as np
import torch
from scripts.write_open_exam import load, infer
from scripts.write_visual_discourse import Writer
from scripts.evaluate_visual_discourse import draw, unpack
from ilm.visual_lm.portable_writer import read_image, read_text, DEFAULT_FONT
from scripts.prepare_open_exam import MODEL, PUB, WORK, dump
from scripts.train_open_exam import corpus

CASES = (
    (
        "travel",
        "小陈今天要去北京，小林明天回上海。两个人在车站告别。",
        "小陈要去哪里？",
        "北京",
        "北京",
        "广州",
        "广州",
    ),
    (
        "purchase",
        "店员说词典三十元，图册五十元。小唐最后买了词典。",
        "小唐买了什么？",
        "词典",
        "买了词典",
        "买了图册",
        "图册",
    ),
    (
        "mirror",
        "闭馆前，馆员把铜镜从东柜搬到西柜。拓片留在东柜。",
        "闭馆时铜镜放在哪个柜子？",
        "西柜",
        "从东柜搬到西柜",
        "从西柜搬到东柜",
        "东柜",
    ),
    (
        "speech",
        "讲义上写着：言，意思是说话。水，意思是河流。",
        "言是什么意思？",
        "说话",
        "说话",
        "言语",
        "言语",
    ),
    (
        "weekday",
        "小林说今天是星期四，明天他们还要上课。",
        "明天是星期几？",
        "星期五",
        "星期四",
        "星期二",
        "星期三",
    ),
    (
        "hour",
        "通知说，讲座上午九点开始，展览下午三点开始。",
        "讲座几点开始？",
        "上午九点",
        "上午九点",
        "上午十点",
        "上午十点",
    ),
)


def main():
    torch.set_num_threads(4)
    writer = Writer()
    readink = lambda s: read_text(s, DEFAULT_FONT)
    rows, tiles, n = corpus("held")
    counts = {}
    for arm in ("parent", "exchanged", "matched"):
        net = load(arm, "cpu")
        total = both = 0
        for name, page, question, expected, old, new, changed in CASES:
            success = []
            for version, p, e in (
                ("original", page, expected),
                ("changed", page.replace(old, new), changed),
            ):
                dest = MODEL / "examples" / name / version
                dest.mkdir(parents=True, exist_ok=True)
                qi, pi = readink(question), readink(p)
                if arm == "parent":
                    draw(dest, "question", qi)
                    draw(dest, "passage", pi)
                actual, stop = infer(
                    writer,
                    net,
                    read_image(dest / "question.png", len(qi)),
                    read_image(dest / "passage.png", len(pi)),
                )
                out = dest / arm
                out.mkdir(exist_ok=True)
                draw(out, "answer", actual)
                (out / "answer.bin").write_bytes(actual.tobytes())
                exact = stop and np.array_equal(actual, readink(e))
                success.append(exact)
                total += exact
                alpha = {bytes(v): c for v, c in zip(writer.palette, writer.transcript)}
                for v, c in zip(np.concatenate([qi, pi]), question + p, strict=True):
                    alpha.setdefault(bytes(v), c)
                dump(
                    out / "result.json",
                    dict(
                        question=question,
                        passage=p,
                        reference=e,
                        question_cells=len(qi),
                        page_cells=len(pi),
                        cells=len(actual),
                        stopped=stop,
                        exact=exact,
                        transcript="".join(alpha.get(bytes(v), "�") for v in actual),
                    ),
                )
            both += all(success)
        counts[arm] = dict(exact=total, both=both, questions=12, pairs=6)
    dump(PUB / "authored.json", counts)
    if (WORK / "matched-responses.json.gz").exists():
        records = [
            r
            for r in unpack(WORK / "matched-responses.json.gz")
            if r["condition"] == "intact"
        ]
        by = {r["id"]: r for r in rows}
        samples = (
            records[:2]
            + [r for r in records if r["exact"]][:2]
            + [r for r in records if r["exact"] and r["absent_multi"]][:2]
        )
        receipts = []
        for r in {r["id"]: r for r in samples}.values():
            row = by[r["id"]]
            dest = WORK / "physical-replay" / r["id"]
            dest.mkdir(parents=True, exist_ok=True)
            q, p = tiles[row["qi"]], tiles[row["pi"]]
            draw(dest, "question", q)
            draw(dest, "passage", p)
            pix, stop = infer(
                writer,
                net,
                read_image(dest / "question.png", len(q)),
                read_image(dest / "passage.png", len(p)),
            )
            if (
                sha256(pix.tobytes()).hexdigest() != r["output_sha256"]
                or stop != r["stop"]
            ):
                raise ValueError("Actual physical output differs in CPU replay")
            draw(dest, "answer", pix)
            receipts.append(
                dict(id=r["id"], hash=r["output_sha256"], exact=r["exact"], stop=stop)
            )
        dump(PUB / "cpu-replay.json", receipts)


if __name__ == "__main__":
    main()
