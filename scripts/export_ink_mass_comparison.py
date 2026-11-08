#!/usr/bin/env python3
"""Authenticate both complete arms and export every raw page before review."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from ilm.image_edit.baseline import ROOT, save_json, sha
from scripts.export_balanced_autonomous_writer import tensor_png


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    load = lambda p: json.loads(p.read_text())
    lines = lambda p: [json.loads(s) for s in p.read_text().splitlines() if s]
    before_path = ROOT / "references/evidence/balanced_autonomous_writer_result"
    before = load(before_path / "result.json")
    manifest = ROOT / "artifacts/image_edit_learning/strip_pages/pairs.jsonl"
    inputs_path = ROOT / "artifacts/image_edit_learning/full_strip_inputs.json"
    inputs = load(inputs_path)
    manifest_by_id = {r["id"]: r for r in lines(manifest)}
    expected_ids = [f"copy/{s}/{i:03}" for s in ("train", "validation") for i in range(32)]
    if before["complete_outputs"] != 64 or any([r["id"] for r in a] != expected_ids for a in (before["rows"],inputs)):
        raise ValueError("complete original-order 64 cases required")
    for name, digest in before["reports"].items():
        if sha(before_path / name) != digest:
            raise ValueError("starting evidence changed")
    sources = {}
    arm_data = {}
    schedules = []
    for arm in ("control", "ink_mass"):
        training = args.run / (arm + "_training")
        inference = args.run / (arm + "_inference")
        score_path = args.run / (arm + "_ocr/report.json")
        report = load(training / "report.json"); meta = report["metadata"]
        if (report["stage"] != "completed" or report["updates"] != 32 or meta["arm"] != arm
                or meta["initial_sha256"] != before["final_checkpoint_sha256"]
                or not meta["optimizer_restored"] or meta["starting_optimizer_step"] != 160
                or meta["manifest_sha256"] != sha(manifest) or meta["inputs_sha256"] != sha(inputs_path)
                or any(sha(ROOT / p) != v for p,v in meta["implementation"].items())
                or meta["physical_coefficient"] != .4035679875746343):
            raise ValueError("registered complete arm differs: " + arm)
        checkpoint = training / "checkpoint.pt"
        if sha(checkpoint) != report["checkpoint_sha256"]:
            raise ValueError("final weights differ")
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if (saved["metadata"] != meta or saved["step"] != 32 or not saved["optimizer"]["state"]
                or len(saved["optimizer"]["state"]) != len(saved["adapters"])
                or any(int(v["step"]) != 192 for v in saved["optimizer"]["state"].values())):
            raise ValueError("all optimizer states must reach 192")
        last = saved["last_entry"]; del saved
        order = torch.Generator().manual_seed(20268602); seeds = torch.Generator().manual_seed(20268603)
        schedule = [dict(update=j+1,epoch=1,id=meta["training_ids"][index],
            noise_seed=int(torch.randint(2**31-1,(1,),generator=seeds)))
            for j,index in enumerate(torch.randperm(32,generator=order).tolist())]
        journal = lines(training / "updates.jsonl")
        if (load(training / "schedule.json") != schedule or len(journal) != 32 or journal[-1] != last
                or set(Counter(r["id"] for r in schedule).values()) != {1}
                or hashlib.sha256(json.dumps(schedule,sort_keys=True).encode()).hexdigest() != meta["schedule_sha256"]):
            raise ValueError("complete registered permutation does not reproduce")
        schedules.append(schedule)
        for row, scheduled in zip(journal,schedule):
            if (any(row[k] != v for k,v in scheduled.items()) or not np.isfinite(row["loss"])
                    or not np.isfinite(row["gradient_norm"]) or row["gradient_norm"] <= 0):
                raise ValueError("invalid actual update")
            folder = training / f"update_{row['update']:03}_before"
            tensor_png(folder/"pixels.pt",folder/"answer.png",row["autonomous_page"]["pixels_sha256"],row["autonomous_page"]["png_sha256"])
        if (not journal[0]["actual_differentiable_forward_native_pixel_exact"]
                or set(journal[0]["trajectory_prediction_gradient_norms"]) != {"0","24","49"}
                or not all(v > 0 and np.isfinite(v) for v in journal[0]["trajectory_prediction_gradient_norms"].values())
                or not report["changed_adapter_tensors"]):
            raise ValueError("actual gradient/native checks missing")
        replays = load(training / "initial_replays.json")
        if [r["id"] for r in replays] != meta["training_ids"][:2] or not all(r["native_pixels_bit_exact"] for r in replays):
            raise ValueError("initial same-checkpoint replay missing")
        for r in replays:
            old = next(v for v in before["rows"] if v["id"] == r["id"])
            folder = training / ("starting_"+old["language"])
            tensor_png(folder/"pixels.pt",folder/"answer.png",r["output"]["pixels_sha256"],old["png_sha256"])
        native = load(inference / "report.json")
        if (native["complete_generations"] != 64 or native["checkpoint_sha256"] != report["checkpoint_sha256"]
                or native["source"] != meta["source"] or native["condition_sha256"] != meta["condition_sha256"]
                or native["inputs_sha256"] != sha(inputs_path) or native["steps"] != 50 or native["guidance"] != 1.
                or native["output_canvas"] != [512,256] or not native["inference_source_files_unchanged"]
                or native["runner_sha256"] != sha(ROOT/"scripts/infer_paired_image_cohort_v2.py")
                or any(native[k] for k in ("training_manifest_loaded","reference_answer_used","question_specific_text_condition","output_repair"))):
            raise ValueError("unassisted native inference differs")
        score = load(score_path); prior_score = load(before_path/"output_ocr_report.json")
        if (not score["complete_cohort"] or score["reference_calibration_sha256"] != sha(before_path/"reference_ocr_report.json")
                or any(score[k] != prior_score[k] for k in ("engine","implementation","manifest_sha256","cohort"))):
            raise ValueError("independent observation/calibration differs")
        outputs = load(inference/"outputs.json")
        if any([r["id"] for r in a] != expected_ids for a in (outputs,score["rows"])):
            raise ValueError("incomplete/reordered outputs")
        arm_data[arm] = dict(training=report, inference=native, scores=score, outputs=outputs)
        for name,p in (("training_report.json",training/"report.json"),("training_schedule.json",training/"schedule.json"),
                ("training_updates.jsonl",training/"updates.jsonl"),("initial_replays.json",training/"initial_replays.json"),
                ("inference_report.json",inference/"report.json"),("output_ocr_report.json",score_path)):
            sources[arm+"_"+name] = p
    if schedules[0] != schedules[1]:
        raise ValueError("two-arm exposure/noise mismatch")
    args.output.mkdir(parents=True,exist_ok=False)
    rows = []
    for i,(old,given) in enumerate(zip(before["rows"],inputs)):
        item = manifest_by_id[old["id"]]
        files = {}; hashes = {}; observed = {}
        pages = [("input",manifest.parent/item["input"],item["input_sha256"]),
            ("reference",manifest.parent/item["target"],item["target_sha256"]),
            ("starting",before_path/old["files"]["answer"],old["png_sha256"])]
        for arm,d in arm_data.items():
            actual,score = d["outputs"][i],d["scores"]["rows"][i]
            if (actual["seed"] != old["seed"] or actual["seed"] != given["seed"]
                    or actual["input_sha256"] != item["input_sha256"] or actual["input_sha256"] != given["input_sha256"]
                    or item["reference_text"] != score["expected"] or score["expected"] != old["expected"]):
                raise ValueError("source/seed/target differs")
            p = args.run/(arm+"_inference")/actual["path"]
            if sha(p) != actual["sha256"]: raise ValueError("output index differs")
            tensor_png(p.parent/"pixels.pt",p,actual["result"]["pixels_sha256"],score["png_sha256"])
            pages.append((arm,p,actual["sha256"]))
            observed[arm] = dict(score,neural_tensor_sha256=actual["result"]["pixels_sha256"],png_equals_quantized_neural_tensor=True)
        for key,p,digest in pages:
            if sha(p) != digest: raise ValueError("source image differs")
            target=args.output/f"{i:03}_{key}.png";shutil.copyfile(p,target);files[key]=target.name;hashes[key]=digest
        rows.append(dict(id=old["id"],split=old["split"],language=old["language"],expected=old["expected"],
            seed=old["seed"],starting_ocr_exact=old["ocr_exact"],observations=observed,files=files,files_sha256=hashes))
    summary = {arm:{s:dict(cases=32,ocr_exact=sum(r["observations"][arm]["ocr_exact"] for r in rows if r["split"]==s))
        for s in ("train","validation")} for arm in arm_data}
    language = {arm:{s+"/"+l:sum(r["observations"][arm]["ocr_exact"] for r in rows if r["split"]==s and r["language"]==l)
        for s in ("train","validation") for l in ("zh","en")} for arm in arm_data}
    if any(summary[a] != arm_data[a]["scores"]["summary"] for a in arm_data):raise ValueError("recomputed counts differ")
    gates = dict(Chinese=language["ink_mass"]["validation/zh"]>=max(6,language["control"]["validation/zh"]+2),
        seen=summary["ink_mass"]["train"]["ocr_exact"]>=max(13,summary["control"]["train"]["ocr_exact"]),
        English=language["ink_mass"]["validation/en"]>=max(13,language["control"]["validation/en"]))
    worker=load(args.run/"comparison.json")
    if worker["exploratory_gates"] != gates or any(worker["scores"][a]["summary"] != summary[a] for a in arm_data):
        raise ValueError("registered worker result differs")
    changes = {s:dict(treatment_only=[r["id"] for r in rows if r["split"]==s and r["observations"]["ink_mass"]["ocr_exact"] and not r["observations"]["control"]["ocr_exact"]],
        control_only=[r["id"] for r in rows if r["split"]==s and not r["observations"]["ink_mass"]["ocr_exact"] and r["observations"]["control"]["ocr_exact"]]) for s in ("train","validation")}
    font=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",11);galleries=[]
    for first in range(0,64,16):
        canvas=Image.new("RGB",(2048,4192),"white");draw=ImageDraw.Draw(canvas)
        for j,r in enumerate(rows[first:first+16]):
            x,y=j%4*512,j//4*1048;draw.rectangle((x,y,x+511,y+23),fill="#eeeeee")
            draw.text((x+4,y+5),r["id"]+" | target / starting / control / ink-mass",font=font,fill="black")
            for k,key in enumerate(("reference","starting","control","ink_mass")):
                with Image.open(args.output/r["files"][key]) as im:canvas.paste(im.convert("RGB"),(x,y+24+k*256))
        name=f"all_cases_{first:02}_{first+15:02}.png";canvas.save(args.output/name);galleries.append(name)
    sources["reference_ocr_report.json"]=before_path/"reference_ocr_report.json"
    sources["worker_comparison.json"]=args.run/"comparison.json"
    reports={}
    for name,p in sources.items():shutil.copyfile(p,args.output/name);reports[name]=sha(p)
    save_json(args.output/"result.json",dict(rows=rows,summary=summary,language_summary=language,paired_changes=changes,
        reports=reports,galleries=galleries,complete_cases=64,complete_final_neural_outputs=128,
        complete_training_pngs_verified=64,all_final_pngs_equal_neural_tensors=True,
        initial_checkpoint_sha256=before["final_checkpoint_sha256"],final_checkpoint_sha256={a:d["training"]["checkpoint_sha256"] for a,d in arm_data.items()},
        registered_schedule_reproduced=True,optimizer_steps_all_192=True,exploratory_gates=gates,
        exploratory_rule_met=all(gates.values()),raw_review_completed=False,writing_qualification=False,
        exporter_sha256=sha(__file__),tensor_verifier_sha256=sha(ROOT/"scripts/export_balanced_autonomous_writer.py"),
        semantic_QA_evaluated=False,output_repair=False,independent_neural_replay=False,general_ilm_goal_complete=False))
    print(json.dumps(dict(summary=summary,language_summary=language,gates=gates),indent=2))


if __name__ == "__main__":
    main()
