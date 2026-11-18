"""Publish exact passage-writing results and source-order changes from the measured outputs."""
import shutil

import numpy as np

from scripts.distil_glyph_planner import OUT as TRANSFER
from scripts.report_passage_diversity import changed_samples
from scripts.train_glyph_planner import OUT, ROOT, load, verified_split, write_new

DEST = ROOT/"references/evidence/glyph_planner_20261118"


def main():
    first = load(OUT/"evaluation.json")
    search = load(OUT/"search/result.json")
    transfer = load(TRANSFER/"evaluation.json")
    DEST.mkdir()
    for name, source in {
        "qa_coverage": OUT/"qa_coverage.json", "qa_glyph_support": OUT/"qa_glyph_support.json",
        "evaluation": OUT/"evaluation.json", "search": OUT/"search/result.json", "transfer": TRANSFER/"evaluation.json",
        "first_holdout": OUT/"fresh/preparation.json", "transfer_holdout": TRANSFER/"fresh/preparation.json",
        "train_resources": OUT/"train_resources.json", "eval_resources": OUT/"evaluate_resources.json",
        "transfer_train_resources": TRANSFER/"train_resources.json", "transfer_eval_resources": TRANSFER/"evaluate_resources.json",
        "aligned_fit": OUT/"aligned/fit.json", "shuffled_fit": OUT/"shuffled/fit.json",
        "distilled_fit": TRANSFER/"distilled/fit.json", "continued_fit": TRANSFER/"reference/fit.json",
    }.items():
        shutil.copyfile(source, DEST/(name+".json"))
    examples = changed_samples(first, verified_split(OUT/"fresh"), np.load(OUT/"fresh/pixels.npy"),
                               OUT/"fresh",DEST,"uniform","aligned")
    table = [["Writer or baseline",f"First test ({first['rows']:,})", f"Transfer test ({transfer['rows']:,})"]]
    table.extend([[label, str(a) if a is not None else "—", str(b) if b is not None else "—"] for label,a,b in [
        ("Local counts",first["local_exact"],transfer["local_exact"]),
        ("Previous uniform writer",first["arms"]["uniform"]["exact"],transfer["arms"]["teacher"]["exact"]),
        ("Shuffled planner, one history",first["arms"]["shuffled"]["exact"],None),
        ("Aligned planner, one history",first["arms"]["aligned"]["exact"],None),
        ("Aligned planner, four histories",search["arms"]["aligned"]["exact"],transfer["arms"]["start"]["exact"]),
        ("Four extra passes on source answers",None,transfer["arms"]["reference"]["exact"]),
        ("Four extra passes with teacher",None,transfer["arms"]["distilled"]["exact"]),
    ]])
    write_new(DEST/"figure.json",dict(table=table,samples=examples))


if __name__ == "__main__":
    main()
