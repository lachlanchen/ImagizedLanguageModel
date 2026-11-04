#!/usr/bin/env python3
"""Generate an explicitly post-hoc missing-transcript table from fixed receipts."""

import hashlib
import json
from decimal import Decimal,ROUND_HALF_UP
from pathlib import Path

from generate_correlated_density_evidence import RUN
from generate_evidence import HERE,ROOT


def interval(values):
    a,b=values
    def number(value):return str(Decimal(str(100*value)).quantize(Decimal(".001"),rounding=ROUND_HALF_UP))
    return number(a) if a==b else number(a)+"--"+number(b)


def main():
    folder=RUN/"vision_transcript_diagnostic"
    original=json.loads((folder/"report.json").read_text());bounds=json.loads((folder/"partial_bounds/report.json").read_text())
    cleanup=json.loads((folder/"cleanup.json").read_text())
    if original["decision"]!="invalid_actor_transcription" or not bounds["post_hoc"] or bounds["model_calls"]!=0:
        raise ValueError("original rejection and post-hoc scope must remain explicit")
    if not original["calibration"]["passed"] or cleanup["remaining_owned_live_pids"] or cleanup["port_open_after_cleanup"]:
        raise ValueError("calibration or owned cleanup differs")
    for name,key in [(RUN/"development/report.json","primary_sha256"),(folder/"report.json","original_diagnostic_sha256"),
                     (folder/"cleanup.json","evaluator_cleanup_sha256")]:
        if hashlib.sha256(name.read_bytes()).hexdigest()!=bounds[key]:raise ValueError("sensitivity provenance changed")
    lines=[r"\begin{center}\small",r"\begin{tabular}{lrrr}\toprule",
        r"Condition & Valid / unknown & Exact text (\%) & Character score (\%)\\\midrule"]
    for route,label in [("k1","K=1"),("k1_blank","K=1, blank Q"),("k8","K=8"),("k8_blank","K=8, blank Q")]:
        row=bounds["measurements"][route]
        if row["valid_examples"]+row["unknown_examples"]!=256:raise ValueError("all cases required")
        lines.append(f"{label} & {row['valid_examples']} / {row['unknown_examples']} & "+
            interval(row["metrics"]["text_exact"]["identification_interval"])+" & "+
            interval(row["metrics"]["bounded_character_accuracy"]["identification_interval"])+r"\\")
    lines.extend([r"\bottomrule\end{tabular}",r"\end{center}"])
    (HERE/"transcript_bounds_table.tex").write_text("\n".join(lines)+"\n")
    paths={"v62_vision_transcript_diagnostic.json":folder/"report.json",
        "v62_transcript_partial_bounds.json":folder/"partial_bounds/report.json",
        "v62_vision_evaluator_cleanup.json":folder/"cleanup.json"}
    hashes={}
    for name,path in paths.items():
        (HERE/"evidence"/name).write_text(path.read_text().replace(str(ROOT)+"/","").replace(str(Path.home())+"/","~/"))
        hashes[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    (HERE/"evidence/transcript_bounds_source_hashes.json").write_text(json.dumps(hashes,indent=2)+"\n")
    print("Generated full-case post-hoc table; original diagnostic remains invalid")


if __name__=="__main__":main()
