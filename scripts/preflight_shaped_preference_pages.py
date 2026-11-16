#!/usr/bin/env python3
"""Registered CPU source-page support check; no student or teacher execution."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from ilm.training_only.shaped_page_renderer import (
    ShapedPageRenderer, MissingGlyphError, page_to_tiles, tiles_to_page, sha,
)
import numpy as np
from PIL import Image

REGISTRATION = ROOT/"references/evidence/shaped_preference_page_registration_20261116.json"


def host_available():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1])/1024**2
    raise RuntimeError("Host RAM cannot be measured")


def budget(start):
    if time.monotonic()-start>120 or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2>1:
        raise RuntimeError("Source preflight exceeded120seconds or1GiB RSS")
    if host_available()<24:
        raise RuntimeError("Host reserve below24GiB")


def run(output):
    started=time.monotonic()
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="" or output.exists():
        raise ValueError("Use CPU-only execution and a new evidence directory")
    if host_available()<48:
        raise RuntimeError("Source preflight needs48GiB available host RAM at entry")
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    os.nice(10)
    registration=json.loads(REGISTRATION.read_text())
    for path,expected in registration["input_sha256"].items():
        if sha(ROOT/path)!=expected:
            raise ValueError("Registered source changed: "+path)
    output.mkdir(parents=True)
    renderer=ShapedPageRenderer()
    controls={}
    one,_,m1,_=renderer.render("中文 言語",verify_expanded=True)
    two,_,m2,_=renderer.render("中文\n言語",verify_expanded=True)
    controls["newline_changes_visible_layout"]=m2["lines"]>m1["lines"] and not np.array_equal(one,two)
    decomposed,_,md,_=renderer.render("a\u0301",verify_expanded=True)
    composed,_,mc,_=renderer.render("\u00e1",verify_expanded=True)
    controls["combining_cluster_matches_composed_form"]=np.array_equal(decomposed,composed)
    controls["raw_combining_source_identity_retained"]=md["source_sha256"]!=mc["source_sha256"]
    _,_,bar,_=renderer.render("|",verify_expanded=True)
    controls["separator_complete_in_expanded_reference"]=bar["expanded_canvas_verified"]
    _,_,rtl,_=renderer.render("العربية",verify_expanded=True)
    controls["right_to_left_shaping_observed"]=any(level%2 for level in rtl["bidi_levels"])
    try:
        renderer.render("\u0378",verify_expanded=True)
    except MissingGlyphError:
        controls["unknown_font_glyph_refused_before_image"]=True
    else:
        controls["unknown_font_glyph_refused_before_image"]=False
    try:
        tiles_to_page(page_to_tiles(one)[:-1],one.shape)
    except ValueError:
        controls["incomplete_tile_stream_refused"]=True
    else:
        controls["incomplete_tile_stream_refused"]=False
    Image.fromarray(two).save(output/"newline_control_source.png")
    Image.fromarray(decomposed).save(output/"combining_control_source.png")
    (output/"controls.json").write_text(json.dumps(controls,indent=2)+"\n")
    budget(started)
    if not all(controls.values()):
        raise RuntimeError("Source-page control failed; preserve outputs and do not alter registration")
    selected={row["row"]:row for row in registration["cases"]}
    originals={}
    for line in (ROOT/registration["pairs_file"]).open():
        row=json.loads(line)
        if row["row"] in selected:
            if row["prospective_partition"]!="train" or row["row_sha256"]!=selected[row["row"]]["row_sha256"]:
                raise ValueError("Support case is not the registered training source")
            originals[row["row"]]=row
    if set(originals)!=set(selected):
        raise ValueError("Missing complete source support cases")
    results=[]
    font_files={}
    with (output/"field_receipts.jsonl").open("x") as journal:
        for number in sorted(selected):
            source=originals[number]
            for field in registration["fields"]:
                budget(started)
                text=source["raw"][field]
                record={"row":number,"field":field,"roles":selected[number]["roles"],
                        "source_row_sha256":source["row_sha256"],
                        "source_text_sha256":hashlib.sha256(text.encode()).hexdigest()}
                try:
                    rgb,binary,meta,debug=renderer.render(text,verify_expanded=True)
                except (MissingGlyphError,OverflowError) as error:
                    record.update(success=False,error=str(error),source_preserved=True)
                else:
                    rgb_tiles=page_to_tiles(rgb);binary_tiles=page_to_tiles(binary)
                    if not np.array_equal(tiles_to_page(rgb_tiles,rgb.shape),rgb) or not np.array_equal(tiles_to_page(binary_tiles,binary.shape),binary):
                        raise RuntimeError("Physical tile stream changed source-page pixels")
                    stem=f"row_{number:05d}_{field}"
                    paths={"rgb":output/(stem+".png"),"binary":output/(stem+"_binary.png"),
                           "tiles":output/(stem+"_tiles.npz"),"layout":output/(stem+"_layout.json")}
                    Image.fromarray(rgb).save(paths["rgb"])
                    Image.fromarray(binary).save(paths["binary"])
                    np.savez_compressed(paths["tiles"],rgb=rgb_tiles,binary=binary_tiles,page_shape=np.array(rgb.shape,dtype=np.int64))
                    paths["layout"].write_bytes(debug)
                    with Image.open(paths["rgb"]) as image:
                        if not np.array_equal(np.asarray(image),rgb):
                            raise RuntimeError("RGB PNG changed source bytes")
                    for font in meta["selected_fonts"]:
                        path=font["file"]
                        if sha(path)!=font["checksum"]:
                            raise RuntimeError("Selected font file differs from native checksum")
                        font_files[path]=font["checksum"]
                    record.update(success=True,metadata=meta,physical_tiles=len(rgb_tiles),
                                  rgb_and_binary_tile_roundtrips_exact=True,
                                  files={key:{"file":p.name,"sha256":sha(p)} for key,p in paths.items()})
                results.append(record)
                journal.write(json.dumps(record,ensure_ascii=False)+"\n");journal.flush()
            print(json.dumps({"completed_row":number,"fields":len(registration["fields"])}),flush=True)
    budget(started)
    for path,expected in registration["input_sha256"].items():
        if sha(ROOT/path)!=expected:
            raise RuntimeError("Registered source changed during execution: "+path)
    for path,expected in font_files.items():
        if sha(path)!=expected:
            raise RuntimeError("Actual selected font bytes changed during execution")
    forbidden=[name for name in sys.modules if name.split('.')[0] in {"torch","transformers","vllm"}]
    if forbidden:
        raise RuntimeError("A neural package entered source preparation")
    successes=[r for r in results if r['success']]
    report={"schema":"shaped-preference-source-page-preflight-v1","registration_sha256":sha(REGISTRATION),
        "controls":controls,"selected_training_rows":len(selected),"all_registered_roles":registration["roles"],
        "complete_field_cases":len(results),"successful_fields":len(successes),
        "rejected_fields":len(results)-len(successes),"source_preflight_passed":len(results)==len(successes),
        "field_receipts_sha256":sha(output/"field_receipts.jsonl"),"actual_selected_font_files_sha256":font_files,
        "maximum_physical_tiles":max((r['physical_tiles'] for r in successes),default=0),
        "all_successful_expanded_canvas_checks_passed":True,"all_successful_RGB_and_binary_tile_roundtrips_exact":True,
        "complete_corpus_qualified":False,"binary_projection_legibility_qualified":False,
        "semantic_labels_qualified":False,"new_model_calls":0,"new_optimizer_updates":0,
        "preference_fit_admitted":False,"source_strings_or_existing_font_files_changed":False,
        "elapsed_seconds":time.monotonic()-started,"peak_process_rss_gib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,
        "ceilings":registration["ceilings"],"neural_packages_imported":forbidden,
        "whole_model_or_language_quality_claim":False,"goal_achieved":False}
    (output/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    run(parser.parse_args().output)
