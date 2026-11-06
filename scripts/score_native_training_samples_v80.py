#!/usr/bin/env python3
"""Score and display every saved native training answer without text repair."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch

from ilm.visual_lm.complete_writing_data import CompleteWritingImageData
from scripts.cache_semantic_images_v72 import ROOT, TRAIN, TRAIN_SHA, atomic_json, sha
from scripts.verify_codec_language_v68 import physical_scores


def raster(images):
    if (images.device.type != "cpu" or images.dtype != torch.float32 or images.ndim != 4
            or images.shape[1:] != (1,32,64) or len(images)>128
            or not bool(((images==-1)|(images==1)).all())):
        raise ValueError("unaltered physical answer blocks required")
    return ((torch.cat(list(images),dim=-1)[0]+1)*127.5).to(torch.uint8).numpy() if len(images) else np.empty((32,0),np.uint8)


def check_answer(result):
    images = result["images"]
    strip = raster(images)
    stopped, capped = result["stopped"], result["capped"]
    probabilities = result["stop_probabilities"]
    if (type(stopped) is not bool or type(capped) is not bool or stopped==capped
            or (capped and len(images)!=128) or len(probabilities)!=len(images)+1
            or not all(math.isfinite(v) and 0<=v<=1 for v in probabilities)
            or any(v>=.5 for v in probabilities[:-1]) or (probabilities[-1]>=.5)!=stopped
            or not result["native_context_kv_retained"] or not result["feedback_from_emitted_pixels"]
            or len(result["feedback_sha256"])!=len(images)):
        raise ValueError("native physical-feedback or stop trace differs")
    return strip


def contact_sheet(rows, destination):
    width = 1024
    panels = []
    for row in rows:
        for label, pixels in (("TARGET", row["target"]), ("ACTUAL OUTPUT", row["actual"])):
            lines = max(1, math.ceil(pixels.shape[1]/width))
            panel = Image.new("RGB", (width+32, 40+lines*40), "white")
            draw = ImageDraw.Draw(panel)
            draw.text((12,10), f"{row['position']:02} {label} | {row['source']} | {pixels.shape[1]//64} blocks", fill="black")
            for line in range(lines):
                segment = pixels[:,line*width:(line+1)*width]
                if segment.size:
                    panel.paste(Image.fromarray(segment).convert("RGB"), (12,32+line*40))
            panels.append(panel)
    image = Image.new("RGB", (width+32, sum(p.height for p in panels)), "#e8e8e8")
    offset=0
    for panel in panels:
        image.paste(panel,(0,offset));offset+=panel.height
    image.save(destination)


def run(run_dir):
    identity=json.loads((run_dir/"identity.json").read_text())
    if identity["purpose"]!="native_chinese_full_context" or identity["train_image_report_sha256"]!=TRAIN_SHA:
        raise ValueError("registered native Chinese training required")
    for name, expected in identity["sources"].items():
        if sha(ROOT/name)!=expected:
            raise ValueError("registered training implementation changed")
    data=CompleteWritingImageData(TRAIN,purpose="train",expected_report_sha256=TRAIN_SHA)
    indices={name:i for i,name in enumerate(data.names)}
    reports=[]
    for folder in sorted((run_dir/"diagnostics").glob("update_*")):
        sources=sorted(folder.glob("[0-9][0-9]/source.json"))
        if len(sources)!=4:
            print(f"Not yet complete: {folder.name}",flush=True);continue
        rows=[];views=[]
        for source_file in sources:
            source=json.loads(source_file.read_text()); answer_dir=source_file.parent
            if not source["training_example"] or source["source"] not in indices:
                raise ValueError("owning admitted training reference required")
            if sha(answer_dir/"answer.pt")!=source["output"]["raw_sha256"]:
                raise ValueError("saved raw training answer changed")
            result=torch.load(answer_dir/"answer.pt",map_location="cpu",weights_only=True)
            actual=check_answer(result)
            target_tensor=data.target(indices[source["source"]])[0,0,:,:-128]
            target=((target_tensor+1)*127.5).to(torch.uint8)
            if len(result["images"]):
                if sha(answer_dir/"answer.png")!=source["output"]["answer_png_sha256"] or not np.array_equal(
                        np.asarray(Image.open(answer_dir/"answer.png")),actual):
                    raise ValueError("training preview differs from unmodified raw output")
            else:
                if (answer_dir/"answer.png").exists():
                    raise ValueError("empty output has a stale preview")
            score=physical_scores(actual<128,[target],result["stopped"])
            rows.append(dict(source=source["source"],update=source["update"],
                raw_sha256=sha(answer_dir/"answer.pt"),target_blocks=target.shape[1]//64,**score))
            views.append(dict(position=int(answer_dir.name),source=source["source"],target=target.numpy(),actual=actual))
        contact_sheet(views,folder/"all_outputs_and_targets.png")
        report=dict(update=rows[0]["update"],cases=4,exact=sum(r["exact"] for r in rows),
            mean_foreground_f1=float(np.mean([r["foreground_f1"] for r in rows])),
            stopped=sum(r["stopped"] for r in rows),rows=rows,training_examples_only=True,
            cohort_varies_with_replay_pool=True,checkpoint_forward_independently_replayed=False,
            raw_pixel_scoring=True,no_output_repair=True,generalization_claim=False,
            contact_sheet_sha256=sha(folder/"all_outputs_and_targets.png"),
            training_identity_sha256=sha(run_dir/"identity.json"),scorer_sha256=sha(Path(__file__)))
        atomic_json(folder/"pixel_scores.json",report);reports.append(report)
        print(json.dumps({k:report[k] for k in ("update","cases","exact","mean_foreground_f1","stopped")}),flush=True)
    return reports


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,required=True)
    run(parser.parse_args().run.resolve())
