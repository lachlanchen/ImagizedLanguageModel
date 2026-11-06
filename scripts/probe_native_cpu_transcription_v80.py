#!/usr/bin/env python3
"""Calibrate CPU OCR before measuring saved native training-answer transcripts."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
from PIL import Image, ImageOps
import torch

from scripts.audit_native_pixel_transcripts_v61 import normalize, scores
from scripts.cache_semantic_images_v72 import ROOT, TRAIN, atomic_json, committed_identity, sha
from scripts.score_native_training_samples_v80 import check_answer
from scripts.train_native_chinese_v80 import ChineseImages

PLAN = "references/native_cpu_transcription_v80_protocol.md"
SOURCES = ("scripts/probe_native_cpu_transcription_v80.py", PLAN,
    "scripts/audit_native_pixel_transcripts_v61.py", "scripts/score_native_training_samples_v80.py",
    "scripts/train_native_chinese_v80.py", "ilm/visual_lm/complete_writing_data.py",
    "ilm/visual_lm/semantic_writing_data.py")
LABELS = dict(cmrc=ROOT/"data/visual_qa/cmrc2018_short/train.jsonl",
    dolly=ROOT/"data/general_instructions/dolly15k/train.jsonl")


def anonymous_view(pixels, directory):
    if pixels.dtype != np.uint8 or pixels.ndim != 2 or pixels.shape[0] != 32:
        raise ValueError("complete physical writing strip required")
    if not np.isin(pixels, (0,255)).all() or not pixels.shape[1]:
        raise ValueError("nonempty binary writing image required")
    image = Image.fromarray(pixels).resize((pixels.shape[1]*4,128), Image.Resampling.NEAREST)
    image = ImageOps.expand(image,border=16,fill=255)
    # Verify that enlargement and padding preserve every original physical bit.
    assert np.array_equal(np.asarray(image)[16:-16:4,16:-16:4],pixels)
    buffer = io.BytesIO(); image.save(buffer,format="PNG"); raw=buffer.getvalue()
    digest=hashlib.sha256(raw).hexdigest(); path=directory/(digest+".png")
    if path.exists() and path.read_bytes()!=raw:
        raise ValueError("anonymous image hash collision")
    path.write_bytes(raw)
    return digest


def prepare(run,out):
    data=ChineseImages(); training=json.loads((run/"identity.json").read_text())
    for name,digest in training["sources"].items():
        if sha(ROOT/name)!=digest: raise ValueError("original training source changed")
    labels={}
    for source,path in LABELS.items():
        if sha(path)!=data.images.report["source_sha256"][source]:
            raise ValueError("source labels differ from the authenticated image preparation")
        labels[source]={r["id"]:r for r in map(json.loads,path.read_text().splitlines())}
    admitted=sorted(data.codes.recipe["names"][int(Path(n).stem)] for n in data.codes.names)
    if len(admitted)!=12976 or len(set(admitted))!=12976:
        raise ValueError("complete admitted image-code inventory required")
    groups={"cmrc":{},"dolly":{}}
    for name in admitted:
        source=Path(name).parts[0]; record=labels[source][Path(name).stem]
        group=record["article_title"] if source=="cmrc" else record["category"]
        groups[source].setdefault(group,[]).append(name)
    rng=np.random.default_rng(20268010); chosen=[]
    articles=sorted(groups["cmrc"])
    for i in rng.choice(len(articles),128,replace=False):
        candidates=groups["cmrc"][articles[i]]
        chosen.append(candidates[int(rng.integers(len(candidates)))])
    if len(groups["dolly"])!=8: raise ValueError("all eight English categories required")
    for category in sorted(groups["dolly"]):
        candidates=groups["dolly"][category]
        chosen.extend(candidates[int(i)] for i in rng.choice(len(candidates),16,replace=False))
    inventory={p.parent.name:sha(p) for p in sorted((run/"diagnostics").glob("update_*/pixel_scores.json"))}
    if not inventory: raise ValueError("at least one completed saved diagnostic required")
    binary=shutil.which("tesseract")
    if not binary: raise FileNotFoundError("the existing CPU recognizer is required")
    weights=Path("/usr/share/tesseract-ocr/5/tessdata")
    identity=dict(**committed_identity(SOURCES),training_identity_sha256=sha(run/"identity.json"),
        image_report_sha256=data.images.report_sha256,code_report_sha256=data.codes.report_sha256,
        source_labels_sha256={s:sha(p) for s,p in LABELS.items()},snapshot_scores_sha256=inventory,
        executable=binary,executable_sha256=sha(Path(binary)),
        version=subprocess.check_output([binary,"--version"],text=True,stderr=subprocess.STDOUT).splitlines()[0],
        weights_sha256={s:sha(weights/(s+".traineddata")) for s in ("chi_sim","eng")},
        languages="chi_sim+eng",tessdata_dir=str(weights),oem=1,psm=7,timeout_seconds=45,workers=2,openmp_threads=1,
        request_receives_question=False,request_receives_reference=False,actor_updated=False)
    if (out/"identity.json").exists():
        previous=json.loads((out/"identity.json").read_text())
        if {k:v for k,v in previous.items() if k!="source_commit"}!={k:v for k,v in identity.items() if k!="source_commit"}:
            raise ValueError("partial diagnostic inputs or implementation changed")
        identity=previous
    elif out.exists() and any(out.iterdir()):
        raise ValueError("preserve the unrelated nonempty output directory")
    out.mkdir(parents=True,exist_ok=True); atomic_json(out/"identity.json",identity)
    folder=out/"anonymous_calibration";folder.mkdir(exist_ok=True)
    index={n:i for i,n in enumerate(data.images.names)}; calibration=[]
    for name in chosen:
        source=Path(name).parts[0]; record=labels[source][Path(name).stem]
        target=data.images.target(index[name])[0,0,:,:-128]
        pixels=((target+1)*127.5).to(torch.uint8).numpy()
        text=record["references"][0]["text"] if source=="cmrc" else record["answer_flat"]
        calibration.append(dict(kind=source,source=name,image_sha256=anonymous_view(pixels,folder),
            source_tensor_sha256=sha(TRAIN/name),expected_text=text))
    noise=np.random.default_rng(20268011)
    for n in range(1,17):
        for kind,pixels in (("blank",np.full((32,n*64),255,np.uint8)),
                ("random_ink",np.where(noise.random((32,n*64))<.15,0,255).astype(np.uint8))):
            calibration.append(dict(kind=kind,source=None,image_sha256=anonymous_view(pixels,folder),expected_text=""))
    if len(calibration)!=288: raise ValueError("the complete fixed calibration is required")
    manifest=out/"private_calibration.json"
    if manifest.exists() and json.loads(manifest.read_text())!=calibration:
        raise ValueError("preserve the earlier calibration")
    atomic_json(manifest,calibration)
    atomic_json(out/"preparation.json",dict(identity_sha256=sha(out/"identity.json"),
        calibration_manifest_sha256=sha(manifest),references=dict(cmrc=128,dolly=128),blank=16,random_ink=16,
        selected_training_only=True,development_examples=0,gpu_calls=0,cuda_initialized=torch.cuda.is_initialized(),
        source_pixels_preserved=True))
    return identity,calibration,labels


def recognize(path,out,identity):
    key=sha(path); destination=out/"ocr_responses"/(key+".json")
    destination.parent.mkdir(parents=True,exist_ok=True)
    request_sha=sha(out/"identity.json")
    if destination.exists():
        result=json.loads(destination.read_text())
        if result["image_sha256"]!=key or result["identity_sha256"]!=request_sha:
            raise ValueError("cached OCR identity differs")
        return result
    command=[identity["executable"],str(path),"stdout","--tessdata-dir",identity["tessdata_dir"],
        "-l",identity["languages"],"--oem","1","--psm","7","tsv"]
    response=None
    try:
        response=subprocess.run(command,capture_output=True,text=True,timeout=45,
            env=dict(os.environ,OMP_THREAD_LIMIT="1"))
        if response.returncode: raise RuntimeError(response.stderr)
        reader=csv.DictReader(io.StringIO(response.stdout),delimiter="\t")
        if not {"text","conf"}<=set(reader.fieldnames or []): raise ValueError("invalid OCR TSV")
        words=[r["text"] for r in reader if r.get("text","").strip() and float(r["conf"])>=0]
        result=dict(valid_process=True,text=" ".join(words),raw_tsv=response.stdout,stderr=response.stderr)
    except (subprocess.TimeoutExpired,RuntimeError,ValueError,OSError) as error:
        def decoded(value): return value.decode(errors="replace") if isinstance(value,bytes) else (value or "")
        result=dict(valid_process=False,text=None,error=repr(error),
            raw_tsv=decoded(getattr(error,"stdout",None) or getattr(response,"stdout",None)),
            stderr=decoded(getattr(error,"stderr",None) or getattr(response,"stderr",None)))
    result.update(image_sha256=key,identity_sha256=request_sha)
    atomic_json(destination,result)
    return result


def run(args):
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="" or torch.cuda.is_initialized():
        raise ValueError("explicit CPU-only execution required")
    torch.set_num_threads(2)
    root=args.run.resolve(); out=(args.out or root/"cpu_transcription_v80").resolve()
    if not out.is_relative_to(root) or out==root: raise ValueError("use a separate output directory inside the run")
    if (out/"report.json").exists(): raise FileExistsError("completed diagnostic is immutable")
    begun=time.perf_counter(); identity,calibration,labels=prepare(root,out)
    if args.prepare_only:
        print(json.dumps(json.loads((out/"preparation.json").read_text()),indent=2));return
    unique={r["image_sha256"]:out/"anonymous_calibration"/(r["image_sha256"]+".png") for r in calibration}
    def batch(images):
        with ThreadPoolExecutor(max_workers=2) as pool:
            return dict(zip(sorted(images),pool.map(lambda k:recognize(images[k],out,identity),sorted(images))))
    responses=batch(unique); rows=[]
    for row in calibration:
        value=responses[row["image_sha256"]]
        rows.append(dict(**row,**{k:value[k] for k in ("valid_process","text")},
            exact=bool(value["valid_process"] and normalize(value["text"])==normalize(row["expected_text"]))))
    atomic_json(out/"calibration_scores.json",rows)
    measured={kind:dict(cases=sum(r["kind"]==kind for r in rows),
        exact=sum(r["kind"]==kind and r["exact"] for r in rows),
        valid=sum(r["kind"]==kind and r["valid_process"] for r in rows)) for kind in ("cmrc","dolly","blank","random_ink")}
    controls=measured["blank"]["exact"]==16 and measured["random_ink"]["exact"]>=15 and all(r["valid_process"] for r in rows)
    qualified={source:controls and measured[source]["exact"]>=122 for source in ("cmrc","dolly")}
    result=dict(identity_sha256=sha(out/"identity.json"),calibration=measured,qualified_sources=qualified,
        actor_images_transcribed=0,training_snapshots_only=True,primary_gates_unchanged=True,
        semantic_accuracy_claim=False,general_ilm_goal_complete=False)
    if qualified["cmrc"]:
        pending=[]; actor_images={}; folder=out/"anonymous_actor_images";folder.mkdir(exist_ok=True)
        for name,digest in identity["snapshot_scores_sha256"].items():
            directory=root/"diagnostics"/name
            if sha(directory/"pixel_scores.json")!=digest: raise ValueError("original snapshot scores changed")
            report=json.loads((directory/"pixel_scores.json").read_text())
            if report["cases"]!=4 or len(report["rows"])!=4 or not report["training_examples_only"]:
                raise ValueError("all four original training diagnostic cases are required")
            for i,row in enumerate(report["rows"]):
                source=directory/f"{i:02}"; header=json.loads((source/"source.json").read_text())
                if (sha(source/"answer.pt")!=row["raw_sha256"] or row["raw_sha256"]!=header["output"]["raw_sha256"]
                        or header["source"]!=row["source"] or not row["source"].startswith("cmrc/")
                        or header["update"]!=report["update"] or row["update"]!=report["update"]
                        or not header["training_example"]):
                    raise ValueError("original actor output changed")
                value=torch.load(source/"answer.pt",map_location="cpu",weights_only=True); pixels=check_answer(value)
                key=anonymous_view(pixels,folder) if pixels.shape[1] else None
                if key is not None:
                    if sha(source/"answer.png")!=header["output"]["answer_png_sha256"] or not np.array_equal(np.asarray(Image.open(source/"answer.png")),pixels):
                        raise ValueError("preview differs from actual output pixels")
                    actor_images[key]=folder/(key+".png")
                elif (source/"answer.png").exists():
                    raise ValueError("empty writing has a stale preview")
                pending.append(dict(update=report["update"],source=row["source"],image_sha256=key))
        recognized=batch(actor_images); scored=[]
        for row in pending:
            value=recognized[row["image_sha256"]] if row["image_sha256"] else dict(valid_process=True,text="")
            record=labels["cmrc"][Path(row["source"]).stem]
            scored.append(dict(**row,valid_process=value["valid_process"],text=value["text"],
                scores=scores(value["text"],[r["text"] for r in record["references"]]) if value["valid_process"] else None))
        atomic_json(out/"actor_transcripts.json",scored)
        result.update(actor_images_transcribed=len(actor_images),actor_cases=len(scored),
            complete_actor_transcription=all(r["valid_process"] for r in scored))
        if result["complete_actor_transcription"]:
            result["snapshots"]={str(update):{metric:float(np.mean([r["scores"][metric] for r in scored if r["update"]==update]))
                for metric in ("text_exact","normalized_character_error","bounded_character_accuracy")}
                for update in sorted({r["update"] for r in scored})}
    result.update(elapsed_seconds=time.perf_counter()-begun,cuda_initialized=torch.cuda.is_initialized(),
        decision="calibrated_auxiliary_transcripts" if result.get("complete_actor_transcription") else "no_qualified_actor_transcript_score")
    atomic_json(out/"report.json",result);print(json.dumps(result,indent=2),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,required=True);parser.add_argument("--out",type=Path)
    parser.add_argument("--prepare-only",action="store_true")
    run(parser.parse_args())
