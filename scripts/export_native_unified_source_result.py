#!/usr/bin/env python3
"""Score and preserve complete native source controls, with no ILM credit."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import unicodedata

import numpy as np
from PIL import Image
import torch

from ilm.native_unified.showo2_source import sha


def normal(text,language):
    text=unicodedata.normalize("NFC",text)
    return "".join(text.split()) if language=="zh" else " ".join(text.split())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run",type=Path,default=Path("artifacts/native_unified_probe"))
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args();root=args.run
    load=lambda name:json.loads((root/name).read_text())
    inputs=load("prepare_v1/inputs.json");references=load("prepare_v1/references.json")
    reading=load("reading_v1/outputs.json");writing=load("writing_v1/outputs.json");text=load("text_control_v1/outputs.json")
    if len(inputs)!=12 or len(references)!=12 or len(reading)!=20 or len(writing)!=8 or len(text)!=12:
        raise ValueError("complete original observations required")
    if [x["id"] for x in text]!=[x["id"] for x in inputs]:raise ValueError("text control order differs")
    if load("cache_replay_v1/report.json")["equal_calls"]!=20:raise ValueError("full cache replay missing")
    args.output.mkdir(parents=True,exist_ok=False);files={}
    def retain(path,name,digest=None):
        actual=sha(path)
        if digest is not None and actual!=digest:raise ValueError("evidence bytes changed")
        dst=args.output/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dst)
        if sha(dst)!=actual:raise ValueError("copy differs")
        files[name]=actual;return name
    def pixels(folder,name):
        value=torch.load(folder/"pixels.pt",map_location="cpu",weights_only=True)
        if value.ndim==5:value=value.squeeze(2)
        array=((value[0].float().permute(1,2,0).numpy()+1)*127.5).round().clip(0,255).astype(np.uint8)
        if not np.array_equal(array,np.asarray(Image.open(folder/"image.png").convert("RGB"))):
            raise ValueError("PNG not equal to original decoder tensor")
        return retain(folder/"image.png",name)
    def ocr(path):
        completed=subprocess.run(["tesseract",str(path),"stdout","-l","chi_sim+eng","--psm","6"],
            env=dict(os.environ,OMP_THREAD_LIMIT="1"),capture_output=True,text=True,check=True)
        return completed.stdout
    cases=[]
    for i,(row,ref,txt) in enumerate(zip(inputs,references,text)):
        image_answer=next(x for x in reading if x["id"]==row["id"] and x["kind"]=="answer")
        target=root/"prepare_v1"/ref["target"]
        calibration=ocr(target)
        entry=dict(id=row["id"],task=row["task"],language=ref["language"],expected=ref["expected"],
            source_image_answer=image_answer["text"],source_image_stopped=image_answer["stopped"],
            source_text_answer=txt["text"],source_text_stopped=txt["stopped"],reference_ocr=calibration)
        for key,value in (("source_image",image_answer["text"]),("source_text",txt["text"]),("reference_ocr",calibration)):
            entry[key+"_exact"]=normal(value,ref["language"])==normal(ref["expected"],ref["language"])
        entry["input"]=retain(root/"prepare_v1"/row["input"],f"raw/{i:02}_input.png",row["input_sha256"])
        entry["reference"]=retain(target,f"raw/{i:02}_reference.png",ref["target_sha256"])
        entry["input_codec"]=pixels(root/"reading_v1"/f"{i:02}"/"input_codec",f"raw/{i:02}_input_codec.png")
        if i<8:
            generated=writing[i]
            if generated["id"]!=row["id"] or not generated["assisted_string_supplied"]:raise ValueError("assisted writing cohort differs")
            entry["assisted_writing"]=pixels(root/"writing_v1"/f"{i:02}",f"raw/{i:02}_assisted.png")
            transcript=ocr(root/"writing_v1"/generated["path"])
            entry["assisted_ocr"]=transcript
            entry["assisted_ocr_exact"]=normal(transcript,ref["language"])==normal(ref["expected"],ref["language"])
        cases.append(entry)
    for folder in ("reading_v1","writing_v1","text_control_v1","cache_replay_v1"):
        for name in ("outputs.json","results.json","report.json"):
            path=root/folder/name
            if path.exists():retain(path,"reports/"+folder+"_"+name)
    for name in ("inputs.json","references.json","preparation.json","assisted_writing.json"):
        retain(root/"prepare_v1"/name,"reports/"+name)
    summary=dict(source_image_exact=sum(x["source_image_exact"] for x in cases),
        source_text_exact=sum(x["source_text_exact"] for x in cases),
        reference_ocr_exact=sum(x["reference_ocr_exact"] for x in cases),
        assisted_writing_ocr_exact=sum(x.get("assisted_ocr_exact",False) for x in cases),
        image_cases=12,text_cases=12,assisted_writing_cases=8)
    result=dict(cases=cases,summary=summary,files_sha256=files,exporter_sha256=sha(__file__),
        all_20_generated_pngs_match_retained_postprocessed_tensors=True,
        copyright="Four CMRC 2018 passage images and references: CC-BY-SA-4.0; Cui et al., EMNLP-IJCNLP 2019. Other copy pages are locally prepared synthetic strings. Native model/source: NUS Show Lab, Apache-2.0.",
        no_ocr_repair=True,optimizer_updates=0,student_language_test=False,language_quality_qualified=False,general_ilm_goal_complete=False)
    (args.output/"result.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(summary,indent=2))


if __name__=="__main__":main()
