#!/usr/bin/env python3
"""V69 verification with FP64 independent accumulation of the mean control.

The source-frozen first verifier remains unchanged. This correction changes
only independent reference summation precision, not producer outputs, scores
or acceptance tolerances. See references/conditional_readout_v69_verification.md.
"""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from ilm.visual_lm.codec_block_writer import CodecBlockWriter, load_qualified_writing_codec
from scripts.probe_conditional_readout_v69 import (ROOT, atomic_json, cpu_only,
    examples, identity, load_features, sha)


def normalize(x):
    x = x.astype(np.float64)
    return (x - x.mean(-1, keepdims=True)) / np.sqrt(x.var(-1, keepdims=True) + 1e-5)


def inputs(h, y, kind):
    if kind == "shared":
        return h.reshape(-1,4096).astype(np.float64), y.reshape(-1,768).astype(np.float64)
    return h.reshape(-1,8192).astype(np.float64), y.reshape(-1,1536).astype(np.float64)


def verify(out):
    cpu_only()
    begun = time.perf_counter()
    report = json.loads((out / "fit_report.json").read_text())
    if report["sources"] != identity()[1] or report["capture_report_sha256"] != sha(out/"capture_report.json"):
        raise ValueError("fixed fit/capture source identity changed")
    for name, key in (("heads.pt","heads_sha256"), ("solve_certificates.pt","solve_certificates_sha256"),
                      ("predictions.pt","predictions_sha256")):
        if sha(out/name) != report[key]:
            raise ValueError("fitted artifact changed")
    rows, cached = load_features(out)
    originals = examples()
    for index, (row, original, cache) in enumerate(zip(rows, originals, cached)):
        routes={"normal","blank"} if row["split"]=="development" else {"normal"}
        if (cache["index"]!=index or cache["sources"]!=report["sources"]
                or cache["pixels_sha256"]!=row["pixels_sha256"] or set(cache["inputs"])!=routes
                or cache["target_code"].shape!=(2,768) or not torch.isfinite(cache["target_code"]).all()):
            raise ValueError("captured input/label boundary differs")
        for key in ("split","subset","source","id","source_file","article","pixels_sha256"):
            if row[key] != original[key]:
                raise ValueError("source record changed")
        for key in ("prompt","blank","reference"):
            if (row[key] is None) != (original[key] is None):
                raise ValueError("source image missing")
            if row[key] is not None and not torch.equal(row[key], original[key]):
                raise ValueError("prepared source image changed")
        for route, value in cache["inputs"].items():
            if (value["states"].shape != (2,4096) or value["original_code"].shape != (2,768)
                    or not torch.isfinite(value["states"]).all() or not torch.isfinite(value["original_code"]).all()):
                raise ValueError("invalid actual contextual measurement")
    h = np.stack([v["inputs"]["normal"]["states"].numpy() for v in cached[:512]])
    y = np.stack([v["target_code"].numpy() for v in cached[:512]])
    fitting = [i for i,r in enumerate(rows[:512]) if r["subset"]=="fit"]
    calibration = [i for i,r in enumerate(rows[:512]) if r["subset"]=="calibration"]
    saved = torch.load(out/"heads.pt",map_location="cpu",weights_only=True)
    certs = torch.load(out/"solve_certificates.pt",map_location="cpu",weights_only=True)
    calibration_checks, equation_residuals, materialization_errors = {}, {}, {}
    grid = (1e-6,1e-4,1e-2,1.,100.)
    for kind, head in saved["heads"].items():
        x, labels = inputs(h[fitting],y[fitting],kind)
        check_x, check_y = inputs(h[calibration],y[calibration],kind)
        mu_x, mu_y = x.mean(0),labels.mean(0)
        scale = np.sqrt(np.mean((x-mu_x)**2)); x=(x-mu_x)/scale; labels=labels-mu_y
        gram = x@x.T/x.shape[1]
        eigenvalues, eigenvectors = np.linalg.eigh(gram)
        qy = eigenvectors.T@labels
        scores=[]
        for regularization in grid:
            alpha=eigenvectors@(qy/(eigenvalues[:,None]+regularization))
            weight=(x.T@alpha/x.shape[1]).astype(np.float32)
            # Match the declared FP32 inference materialization; the independent
            # eigen solve and NumPy prediction are not the author's Torch solve.
            z=((check_x.astype(np.float32)-mu_x.astype(np.float32))/np.float32(scale))@weight+mu_y.astype(np.float32)
            scores.append(float(np.mean((normalize(normalize(z.reshape(-1,2,768)))-normalize(y[calibration]))**2)))
        selected=grid[min(range(len(scores)),key=scores.__getitem__)]
        if selected!=head["regularization"] or not np.allclose(scores,report["selections"][kind]["calibration_mse_grid"],atol=2e-5,rtol=0):
            raise ValueError("training-only calibration choice differs")
        calibration_checks[kind]=dict(selected=selected, independent_mse=scores)
        x,labels=inputs(h,y,kind);mu_x=x.mean(0);mu_y=labels.mean(0)
        scale=np.sqrt(np.mean((x-mu_x)**2));x=(x-mu_x)/scale;labels=labels-mu_y
        alpha=certs[kind]["alpha"].numpy();system=x@x.T/x.shape[1]+selected*np.eye(len(x))
        residual=float(np.max(np.abs(system@alpha-labels)))
        weight=x.T@alpha/x.shape[1]
        error=float(np.max(np.abs(weight.astype(np.float32)-head["weight"].numpy())))
        if residual>1e-7 or error>1e-6:
            raise ValueError("ridge equation or FP32 materialization differs")
        if not np.allclose(mu_x,head["mean_x"].numpy(),atol=1e-5,rtol=1e-7):
            raise ValueError("head feature centering differs")
        if (not np.allclose(mu_y,head["mean_y"].numpy(),atol=1e-6,rtol=1e-7)
                or not np.isclose(scale,float(head["scale"]),atol=1e-7,rtol=1e-7)):
            raise ValueError("head training-only output mean or scale differs")
        equation_residuals[kind]=residual;materialization_errors[kind]=error
    if not np.allclose(normalize(y.astype(np.float64).mean(0)),saved["mean_code"].numpy(),atol=2e-6,rtol=0):
        raise ValueError("prompt-free mean control differs from training labels")
    writer=CodecBlockWriter(load_qualified_writing_codec(
        ROOT/"artifacts/continuous_glyph_codec_v34_20260814/checkpoint_latest.pt",torch.device("cpu")))
    target_error=0.
    with torch.no_grad():
        for start in range(0,len(rows),32):
            codes=writer.encode_targets(torch.cat([r["reference"] for r in rows[start:start+32]]))
            expected=torch.stack([v["target_code"] for v in cached[start:start+32]])
            target_error=max(target_error,float((codes-expected).abs().max()))
    if target_error>1e-4:
        raise ValueError("target code replay differs from source images")
    emitted=torch.load(out/"predictions.pt",map_location="cpu",weights_only=True)
    if len(emitted)!=3584 or len(report["measurements"])!=3584:
        raise ValueError("all four readouts and 896 inputs required")
    measured=[];numpy_prediction_error=0.;pixel_replay_error=0.
    for start in range(0,len(emitted),4):
        block=emitted[start:start+4]; index=block[0]["index"];route=block[0]["route"]
        if [v["head"] for v in block]!=["original","mean","shared","joint"]:
            raise ValueError("readout ordering changed")
        value=cached[index]["inputs"][route];state=value["states"].numpy()
        predictions=[]
        for item in block:
            kind=item["head"]
            if kind=="original": expected=value["original_code"].numpy()
            elif kind=="mean":expected=saved["mean_code"].numpy()
            else:
                head=saved["heads"][kind]
                a=state.reshape(-1,4096) if kind=="shared" else state.reshape(1,-1)
                z=((a.astype(np.float64)-head["mean_x"].numpy())/float(head["scale"]))@head["weight"].numpy()+head["mean_y"].numpy()
                expected=normalize(z.reshape(2,768))
            numpy_prediction_error=max(numpy_prediction_error,float(np.max(np.abs(expected-item["code"].numpy()))))
            predictions.append(item["code"])
        with torch.no_grad(): replay=writer.predict(torch.stack(predictions))
        for item, pixels in zip(block,replay):
            delta=float((pixels-item["pixels"]).abs().max());pixel_replay_error=max(pixel_replay_error,delta)
            if delta>1e-6 or not torch.equal(pixels<0,item["pixels"]<0):
                raise ValueError("physical decoder replay differs")
            a=item["pixels"].numpy()<0;b=rows[index]["reference"][0].numpy()<0
            denominator=int(a.sum()+b.sum())
            measured.append(dict(index=index,route=route,head=item["head"],
                foreground_f1=2*int((a&b).sum())/denominator if denominator else 1.,
                exact=bool(np.array_equal(a,b)),blank=not bool(a.any()),
                code_mse=float(np.mean((normalize(item["code"].numpy())-normalize(cached[index]["target_code"].numpy()))**2))))
    if numpy_prediction_error>1e-4:
        raise ValueError("independent linear-head prediction differs")
    for actual,original in zip(measured,report["measurements"]):
        if any(actual[k]!=original[k] for k in ("index","route","head","exact","blank")):
            raise ValueError("physical identity score differs")
        if abs(actual["foreground_f1"]-original["foreground_f1"])>1e-14 or abs(actual["code_mse"]-original["code_mse"])>2e-6:
            raise ValueError("physical F1 or code MSE differs")
    for group, original in report["summaries"].items():
        split,source,route,kind=group.split("/")
        selected=[m for m in measured if m["route"]==route and m["head"]==kind
            and rows[m["index"]]["split"]==split and rows[m["index"]]["source"]==source]
        if len(selected)!=original["count"]:
            raise ValueError("summary sample count differs")
        for key in ("foreground_f1","exact","blank","code_mse"):
            if abs(float(np.mean([m[key] for m in selected]))-original[key])>2e-6:
                raise ValueError("summary differs")
    for kind, original in report["question_effects"].items():
        pairs={}
        for m in measured:
            row=rows[m["index"]]
            if m["head"]==kind and row["split"]=="development" and row["source"]=="cmrc":
                pairs.setdefault(row["article"],{}).setdefault(m["index"],{})[m["route"]]=m
        cluster=np.array([np.mean([v["normal"]["foreground_f1"]-v["blank"]["foreground_f1"]
            for v in pairs[key].values()]) for key in sorted(pairs)])
        both=np.mean([all(v["normal"]["exact"] for v in item.values()) for item in pairs.values()])
        draws=np.random.default_rng(20266903).integers(0,64,(2000,64))
        interval=np.quantile(cluster[draws].mean(1),[.025,.975])
        if (abs(cluster.mean()-original["mean"])>1e-14 or not np.allclose(interval,original["ci95"],atol=1e-14,rtol=0)
                or both!=original["both_first_blocks_exact"]):
            raise ValueError("paired first-block evidence differs")
    result=dict(fit_report_sha256=sha(out/"fit_report.json"),verifier_sha256=sha(Path(__file__)),
        source_cases_reloaded=704,physical_predictions_verified=3584,independent_calibration=calibration_checks,
        normal_equation_residual_max=equation_residuals,fp32_materialization_error_max=materialization_errors,
        target_encoder_replay_error_max=target_error,numpy_head_prediction_error_max=numpy_prediction_error,
        decoder_replay_error_max=pixel_replay_error,offline_checks_pass=True,
        independent_full_actor_forward_replayed=False,first_block_only=True,stopping_evaluated=False,
        mean_control_reference_accumulation="FP64 NumPy; producer uses FP32 Torch",
        original_verifier_preserved=True,
        language_quality_claim=False,general_ilm_goal_complete=False,elapsed_seconds=time.perf_counter()-begun)
    atomic_json(out/"verification.json",result)
    atomic_json(out/"status.json",dict(stage="completed_verified",pid=None,language_quality_claim=False))
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--out",type=Path,required=True)
    verify(parser.parse_args().out)
