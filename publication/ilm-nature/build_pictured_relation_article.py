"""Build Figure 94 and verify the pictured-relation study in all three editions."""

import argparse
import fcntl
import json
import sys

import build_role_binding_article as preceding
from inspect_grounded_proposition import inspect

prior = preceding.prior

paper, base, previous = prior.paper, prior.base, prior.previous
work = paper.ROOT / "artifacts/pictured_relation_publication_v1"
paper.WORK = previous.WORK = work
paper.STAGE = previous.STAGE = base.STAGE = work / "article"
previous.BASELINE = "11c4f101"
study = "pictured_relation"
prior.SPECS[study] = (
    "Figure_94_pictured_relation",
    "Pictured_relation",
    "fig:pictured-relation",
)
prior.configure(study, "20261123")
base.FIGURES.append((paper.FIG, study + "_caption.tex", prior.SPECS[study][2]))
literal = base.literal_figures
base.literal_figures = lambda: literal().replace(
    "build_role_binding_article.py", "build_pictured_relation_article.py"
)


def build():
    """Compile serially, giving each edition its own time budget and retaining prior attempts."""
    base.preserved()
    stage = paper.STAGE
    receipt = stage / "build_receipt.json"
    if receipt.exists():
        raise FileExistsError(receipt)
    env = previous.environment(stage)
    sources = {p.name: base.sha(p) for p in paper.HERE.glob("*.tex")}
    jobs = []
    for folder, name in base.DOCUMENTS[:2]:
        out = stage / folder
        out.mkdir(exist_ok=True)
        jobs.append((name, ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error",
                           "-outdir=" + str(out), name + ".tex"]))
    jobs.append(("redline", [sys.executable, str(paper.HERE / "build_motor_diagnostics_redline.py")]))
    attempts = stage / "compilation"
    attempts.mkdir(exist_ok=True)
    cost_file = stage / "child_costs.json"
    if cost_file.exists() and not list(attempts.glob("*.json")):
        (attempts / "initial.json").write_bytes(cost_file.read_bytes())
    for name, command in jobs:
        number = len(list(attempts.glob(name + "-*.log"))) + 1
        log = attempts / f"{name}-{number}.log"
        try:
            base.run(command, log, env, previous.costs())
        finally:
            if cost_file.exists():
                log.with_suffix(".json").write_bytes(cost_file.read_bytes())
    batches = [json.loads(p.read_text()) for p in sorted(attempts.glob("*.json"))]
    cost_file.write_text(json.dumps(dict(
        child_seconds=sum(v["child_seconds"] for v in batches),
        minimum_available_host_gib=min(v["minimum_available_host_gib"] for v in batches),
        peak_child_tree_rss_gib=max(v["peak_child_tree_rss_gib"] for v in batches),
        children=[item for v in batches for item in v["children"]],
    ), indent=2) + "\n")
    pdfs = {}
    for folder, name in base.DOCUMENTS:
        log = (stage / folder / (name + ".log")).read_text()
        errors = [line for line in log.splitlines() if any(t in line for t in (
            "Overfull", "Float too large", "undefined references", "undefined citations",
            "multiply defined", "LaTeX Error"))]
        if errors:
            raise ValueError((name, errors))
        pdfs[name] = base.sha(stage / folder / (name + ".pdf"))
    if sources != {p.name: base.sha(p) for p in paper.HERE.glob("*.tex")}:
        raise ValueError("Manuscript inputs changed during compilation")
    receipt.write_text(json.dumps(dict(documents=pdfs, sources=sources, baseline=previous.BASELINE,
        redline_baseline=base.load(stage / "redline_build.json")["baseline_commit"]), indent=2) + "\n")


def promote():
    expected = prior.verify(study, "20261123")

    def verify(table):
        if table != expected:
            raise ValueError("Revision table changed")

    paper.promote(verify, "pictured_relation_article_build.json")


if __name__ == "__main__":
    actions = dict(
        baseline=previous.baseline,
        figure=lambda: prior.export(study, "20261123"),
        build=build,
        inspect=lambda: inspect(base, verify_existing_pages=True),
        promote=promote,
        captions=lambda: (paper.HERE / "complete_continuation_figures.tex").write_text(
            base.literal_figures()
        ),
    )
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=actions)
    with (paper.ROOT / "artifacts/publication_cpu.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        actions[p.parse_args().action]()
