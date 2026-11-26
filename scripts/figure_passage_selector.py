"""Draw the whole-word test and show actual output crops with their selection rules."""

from io import BytesIO
from zipfile import ZipFile, ZIP_DEFLATED
import numpy as np
from PIL import Image
from pptx.chart.data import CategoryChartData
from pptx.enum.chart import XL_CHART_TYPE
from pptx.dml.color import RGBColor
from pptx.util import Pt
from reportlab.lib.colors import HexColor
from reportlab.lib.utils import ImageReader

from scripts.figure_book_order_prior import Plate, PAPER
from scripts.evaluate_passage_selector import ARCHIVE, read_ink, read_array
from scripts.evaluate_book_order_prior import emit
from scripts.prepare_passage_selector import (
    PUB,
    WORK,
    base,
    fiction,
    normalize,
    lex,
    WORD,
)


def join_words(words):
    parts = []
    for w in words:
        if len(w):
            if parts:
                parts.append(np.zeros((1, 128), np.uint8))
            parts.append(w)
    return np.concatenate(parts)


def main():
    receipt = base.read(PUB / "scores.json")
    assert base.sha(ARCHIVE) == receipt["archive"]
    panel = Plate("Learned_passage_words", 231)
    panel.text(0, 0, "a   Exact next word on unseen passages", 11, True)
    names = ["Fiction", "PG-19", "LAMBADA"]
    groups = [
        receipt["results"][s]["strata"]["all"]
        for s in ("fiction", "gutenberg", "lambada")
    ]
    colors = ["608987", "C5854B", "233D59"]
    series = {}
    for i, (k, title) in enumerate(
        [
            ("salience", "Page frequency × IDF"),
            ("local_counts", "Local counts"),
            ("intact", "Visual predictor"),
        ]
    ):
        series[title] = [100 * g["correct"][k] / g["n"] for g in groups]
        panel.text(76 + i * 145, 18, title, 8, True, colors[i])
    pdf = panel.pdf
    for j, n in enumerate(names):
        pdf.setFont("Helvetica-Bold", 9)
        pdf.setFillColor(HexColor("#183631"))
        pdf.drawString(4, panel.height - 48 - 25 * j, n)
        for i, val in enumerate(series.values()):
            y = 34 + 25 * j + i * 6.5
            x = 81
            pdf.setFillColor(HexColor("#" + colors[i]))
            pdf.rect(x, panel.height - y - 5, 18 * val[j], 4.5, fill=1, stroke=0)
            pdf.setFont("Helvetica", 6.6)
            pdf.drawString(x + 18 * val[j] + 3, panel.height - y - 5, f"{val[j]:.1f}%")
    data = CategoryChartData()
    data.categories = names
    for k, v in series.items():
        data.add_series(k, v)
    chart = panel.slide.shapes.add_chart(
        XL_CHART_TYPE.BAR_CLUSTERED, Pt(4), Pt(28), Pt(482), Pt(83), data
    ).chart
    chart.has_legend = False
    chart.value_axis.minimum_scale = 0
    chart.value_axis.maximum_scale = 20
    chart.value_axis.tick_labels.font.size = Pt(7)
    chart.category_axis.tick_labels.font.size = Pt(8)
    chart.category_axis.reverse_order = True
    for s, c in zip(chart.series, colors, strict=True):
        s.format.fill.solid()
        s.format.fill.fore_color.rgb = RGBColor.from_string(c)
    face = base.Ink()
    forms = base.read(WORK / "forms.json")
    example = PAPER / "evidence/learned_passage_examples.zip"
    clips = []
    with ZipFile(ARCHIVE) as z, np.load(WORK / "fiction.npz") as a:
        src, marks, ys, ends = [a[k] for k in ("source", "marks", "target", "ends")]
        books = base.read(PUB / "data.json")["books"]["fiction"]
        groups = a["group"]

        def crop(i):
            book_index = books[groups[i]]["index"]
            text = normalize(
                next(raw for idx, raw, _ in fiction() if idx == book_index)
            )
            tokens = list(WORD.finditer(text))
            expected = [forms[k] for k in src[i] if k]
            spans = [
                k
                for k, t in enumerate(tokens)
                if t[0] == forms[ys[i, 0]]
                and k >= len(expected)
                and [u[0] for u in tokens[k - len(expected) : k]] == expected
            ]
            assert len(spans) == 1
            k = spans[0]
            prefix = text[tokens[k - len(expected)].start() : tokens[k].start()]
            _, gaps = lex(prefix)
            assert np.array_equal(gaps, marks[i][src[i] != 0])
            ids = face.encode(prefix[-32:])
            return face.array()[ids]

        out = read_ink(z, "fiction/intact")
        gold = read_ink(z, "fiction/gold0")
        i = next(
            i
            for i in range(len(src))
            if ys[i, 0] not in src[i] and np.array_equal(out[i], gold[i])
        )
        clips.append((i, crop(i), gold[i], out[i]))
        selection = read_array(z, "fiction/stream-rows")
        streams = [read_ink(z, "fiction/stream" + str(j)) for j in range(3)]
        targets = [read_ink(z, "fiction/gold" + str(j)) for j in range(3)]
        at = next(
            k
            for k, i in enumerate(selection)
            if all(np.array_equal(streams[j][k], targets[j][i]) for j in (0, 1))
            and not ends[i, :2].any()
        )
        i = int(selection[at])
        clips.append(
            (
                i,
                crop(i),
                join_words([v[i] for v in targets]),
                join_words([v[at] for v in streams]),
            )
        )
    with ZipFile(example, "w", ZIP_DEFLATED) as z:
        for i, src_img, ref, pred in clips:
            for k, x in [("input", src_img), ("reference", ref), ("output", pred)]:
                emit(z, f"{i}/{k}", x)
    panel.text(0, 119, "b   First exact word absent from its pictured prefix", 10, True)
    panel.text(0, 175, "c   First two consecutive exact words under feedback", 10, True)
    from hashlib import sha256

    with ZipFile(example) as z:
        for y, (i, *_) in zip((133, 189), clips, strict=True):
            for k, text in [
                ("input", "Final input cells"),
                ("reference", "Published"),
                ("output", "Model output"),
            ]:
                member = f"{i}/{k}.png"
                blob = z.read(member)
                with Image.open(BytesIO(blob)) as im:
                    w, h = im.size
                factor = min(0.32, 374 / w)
                panel.text(7, y, text, 8)
                panel.pdf.drawImage(
                    ImageReader(BytesIO(blob)),
                    101,
                    panel.height - y - h * factor,
                    w * factor,
                    h * factor,
                )
                panel.slide.shapes.add_picture(
                    BytesIO(blob), Pt(101), Pt(y), Pt(w * factor), Pt(h * factor)
                )
                panel.sources.append(
                    dict(
                        archive=example.name,
                        member=member,
                        sha256=sha256(blob).hexdigest(),
                    )
                )
                y += 13
    panel.finish()
    base.dump(
        PUB / "figure.json",
        dict(
            file=panel.name,
            items=panel.sources,
            examples=[c[0] for c in clips],
            archive=receipt["archive"],
            clips=base.sha(example),
            categories=names,
            series=series,
        ),
    )
    print("examples", [(i, len(x), len(y)) for i, _, x, y in clips])


if __name__ == "__main__":
    main()
