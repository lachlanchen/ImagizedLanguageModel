"""Build the cold image/byte learning curves and same-exposure subword comparison."""

import argparse
import fcntl
import json
import build_dialogue_ink_article as preceding
from build_pictured_relation_article import build
from inspect_grounded_proposition import inspect

paper, base, previous, prior = (
    preceding.paper,
    preceding.base,
    preceding.previous,
    preceding.prior,
)
study = "dialogue_efficiency"
work = paper.ROOT / "artifacts/dialogue_efficiency_publication_v1"
paper.WORK = previous.WORK = work
paper.STAGE = previous.STAGE = base.STAGE = work / "article"
previous.BASELINE = "a685287e"
prior.SPECS[study] = (
    "Figure_129_dialogue_efficiency",
    "Dialogue_efficiency",
    "fig:dialogue-efficiency",
)
prior.configure(study, "v1")
base.FIGURES.append((paper.FIG, study + "_caption.tex", prior.SPECS[study][2]))
literal = base.literal_figures
base.literal_figures = lambda: literal().replace(
    "build_dialogue_ink_article.py", "build_dialogue_efficiency_article.py"
)


def figure_data():
    p = paper.ROOT / "references/evidence/dialogue_ink_v1"
    curves = [
        json.loads((p / f"{mode}_curve.json").read_text()) for mode in ("ink", "byte")
    ]
    series = [
        [round(100 * r["writer_exact"] / r["turns"], 2) for r in curve["dev"].values()]
        for curve in curves
    ]
    comp = json.loads((p / "token_comparison.json").read_text())["scores"]["dev"]
    controlled = json.loads((p / "cold_comparison.json").read_text())["scores"]["dev"]
    table = [["Model", "Fit energy (kJ)", "First answer", "Four-ref. F1"]]
    for label, file, score in [
        ("Visual, inherited", "dialogue_ink", comp["direct"]["all"]),
        ("Subword Transformer", "dialogue_subword", comp["subword"]["all"]),
        ("Visual, cold", "dialogue_surfaces/ink", controlled["ink"]["all"]),
        ("Byte, cold", "dialogue_surfaces/byte", controlled["byte"]["all"]),
    ]:
        meta = json.loads((paper.ROOT / "models" / file / "weights.json").read_text())
        cost = meta["fit"] if "fit" in meta else meta["checkpoints"]["8000"]
        table.append(
            [
                label,
                f"{cost['energy']['joules']/1000:.2f}",
                f"{score['exact']:,} / {score['n']:,}",
                f"{score['official_f1']*100:.1f}%",
            ]
        )
    return dict(series=series, table=table)


def verify():
    if json.loads((paper.SOURCE / "figure.json").read_text()) != figure_data():
        raise ValueError("Different measured values")
    return prior.verify(study, "v1")


def promote():
    expected = verify()

    def check(table):
        if table != expected:
            raise ValueError("Different native table")

    paper.promote(check, "dialogue_efficiency_article_build.json")


if __name__ == "__main__":
    actions = dict(
        baseline=previous.baseline,
        figure=lambda: prior.export(study, "v1"),
        build=build,
        inspect=lambda: inspect(base, verify_existing_pages=True),
        promote=promote,
        verify=verify,
        captions=lambda: (paper.HERE / "complete_continuation_figures.tex").write_text(
            base.literal_figures()
        ),
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=actions)
    work.mkdir(exist_ok=True)
    with (paper.ROOT / "artifacts/publication_cpu.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        actions[parser.parse_args().action]()
