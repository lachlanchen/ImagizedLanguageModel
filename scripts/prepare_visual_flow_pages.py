#!/usr/bin/env python3
"""Prepare complete input/answer image pairs for native-resolution training.

Input pages stay byte-identical. Answer strips are placed at their original
coordinates on white 1024-square canvases, without cropping or rescaling.
No reference strings are included in the resulting training image manifest.
"""
import argparse
import json
import os
from pathlib import Path
import time

from ilm.image_edit.baseline import ROOT, save_json, sha

PAGES = ROOT / "artifacts/image_edit_learning/pages_v2"
SOURCE = ROOT / "data/visual_qa/cmrc2018_short/train.jsonl"
MANIFEST_SHA = "e2d6aec134a99e079a9bd42f217559a04e32727bdbc4b09d131b0fb22cf4dfa6"
SOURCE_SHA = "742a473db53d1a55e319f0b52befc8da5fa3d884df395b37e66581e882d1496c"


def main(output):
    from PIL import Image
    import numpy as np
    if sha(PAGES / "pairs.jsonl") != MANIFEST_SHA or sha(SOURCE) != SOURCE_SHA:
        raise ValueError("complete registered source corpus/manifest required")
    original = [json.loads(x) for x in (PAGES / "pairs.jsonl").read_text().splitlines()]
    accepted = {r["id"]: {a["text"] for a in r["references"]}
        for r in map(json.loads, SOURCE.read_text().splitlines())}
    counts, articles, qa_pairs = {}, {"train": set(), "validation": set()}, {"train": {}, "validation": {}}
    if len(original) != 448 or len({r["id"] for r in original}) != 448:
        raise ValueError("all original 448 pairs required, without output-based selection")
    for r in original:
        key = r["split"] + "/" + r["task"]
        counts[key] = counts.get(key, 0) + 1
        if r["task"] == "qa":
            if r["reference_text"] not in accepted[r["id"]]:
                raise ValueError("canonical answer not accepted by the original QA source")
            articles[r["split"]].add(r["article"])
            qa_pairs[r["split"]].setdefault(r["pair"], []).append(r["id"])
    if counts != {"train/copy":128, "validation/copy":32, "train/qa":256, "validation/qa":32}:
        raise ValueError("complete registered train/validation counts differ")
    if articles["train"] & articles["validation"] or any(len(v) != 2 for d in qa_pairs.values() for v in d.values()):
        raise ValueError("article-disjoint complete question pairs required")
    output.mkdir(parents=True, exist_ok=False); (output / "targets").mkdir()
    rows = []; started = time.perf_counter()
    for i, r in enumerate(original):
        source_input, source_target = PAGES / r["input"], PAGES / r["target"]
        if sha(source_input) != r["input_sha256"] or sha(source_target) != r["target_sha256"]:
            raise ValueError("original rendered image bytes changed")
        with Image.open(source_input) as image:
            if image.size != (1024,1024) or image.mode != "RGB": raise ValueError("complete RGB page required")
        with Image.open(source_target) as image:
            if image.size != (512,256) or image.mode != "RGB": raise ValueError("original RGB answer strip required")
            canvas = Image.new("RGB", (1024,1024), "white"); canvas.paste(image, (0,0))
            if not np.array_equal(np.array(canvas.crop((0,0,512,256))),np.array(image)):
                raise AssertionError("answer pixels changed during white padding")
        array=np.array(canvas)
        if not (array[256:]==255).all() or not (array[:256,512:]==255).all():
            raise AssertionError("padding must contain only white pixels")
        target=output/"targets"/f"{i:04}.png";canvas.save(target)
        row=dict(id=r["id"],split=r["split"],task=r["task"],language=r["language"],
            input=os.path.relpath(source_input,output),input_sha256=r["input_sha256"],
            target=str(target.relative_to(output)),target_sha256=sha(target),
            original_target_sha256=r["target_sha256"])
        if r["task"]=="qa":row.update(article=r["article"],pair=r["pair"])
        rows.append(row)
    manifest=output/"image_pairs.jsonl"
    manifest.write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in rows))
    result=dict(implementation_sha256=sha(__file__),original_manifest_sha256=MANIFEST_SHA,
        original_QA_source_sha256=SOURCE_SHA,image_manifest_sha256=sha(manifest),complete_pairs=len(rows),
        counts=counts,qa_articles={k:len(v) for k,v in articles.items()},qa_pairs={k:len(v) for k,v in qa_pairs.items()},
        QA_articles_disjoint=True,all_QA_targets_accepted_by_original_source=True,
        input_pages_byte_identical=True,target_original_region_pixel_identical=True,padding_only_white=True,
        input_and_target_geometry=[1024,1024],original_target_geometry=[512,256],
        target_resize_or_rerender_performed=False,reference_strings_in_image_manifest=False,
        source_prep_has_answer_access=True,models_loaded=False,neural_forward_performed=False,optimizer_updates=0,
        new_codec_reconstruction_quality_validated=False,language_learning_established=False,
        elapsed_seconds=time.perf_counter()-started,general_ilm_goal_complete=False)
    save_json(output/"report.json",result)
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",type=Path,required=True)
    args=p.parse_args();main(args.output.resolve())
