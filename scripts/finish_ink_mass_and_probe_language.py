#!/usr/bin/env python3
"""Wait for the exact comparison owner, export it, then probe real full pages.

All model stages are sequential. This worker never retries or chooses weights
from held-out results. It leaves both experiment and preflight review pending.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from ilm.image_edit.baseline import ROOT, save_json, sha
from scripts.finish_balanced_writer import owner


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-run",type=Path,required=True)
    parser.add_argument("--comparison-pid",type=int,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    comparison=args.comparison_run.resolve();output=args.output.resolve()
    identity=owner(args.comparison_pid)
    if identity is None or "scripts/run_ink_mass_comparison.py" not in identity["command"] or str(comparison) not in identity["command"]:
        raise ValueError("exact currently live comparison coordinator required")
    paths=("scripts/finish_ink_mass_and_probe_language.py","scripts/finish_balanced_writer.py",
        "scripts/export_ink_mass_comparison.py","scripts/export_balanced_autonomous_writer.py",
        "scripts/probe_native_paired_language.py","references/paired_language_native_preflight_protocol.md",
        "scripts/infer_paired_image_cohort_v2.py","scripts/score_paired_writing.py",
        "scripts/evaluate_image_edit_baseline.py","ilm/image_edit/baseline.py","ilm/image_edit/paired.py",
        "ilm/image_edit/autonomous.py","ilm/image_edit/recognition.py")
    if subprocess.check_output(["git","status","--porcelain","--",*paths],cwd=ROOT,text=True):
        raise ValueError("commit exact completion/probe sources first")
    code={p:sha(ROOT/p) for p in paths}
    output.mkdir(parents=True,exist_ok=False)
    phase,child="waiting_comparison",None
    started=time.time()

    def status(**extra):
        save_json(output/"runtime.json",dict(phase=phase,
            coordinator_pid=None if phase in ("failed","ready_for_raw_review") else os.getpid(),
            child_pid=None if child is None or child.poll() is not None else child.pid,
            comparison_owner=identity,started_unix=started,code_sha256=code,gpu=0,novnc_url=None,
            general_ilm_goal_complete=False,**extra))

    def interrupted(signum,frame):raise InterruptedError("probe worker signal "+str(signum))
    for signum in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):signal.signal(signum,interrupted)

    def execute(name,script,arguments,gpu):
        nonlocal phase,child
        if any(sha(ROOT/p)!=v for p,v in code.items()):raise ValueError("bound stage source changed")
        phase=name
        command=[str(Path.home()/".local/bin/with-current-nvidia"),"env",
            "CUDA_DEVICE_ORDER=PCI_BUS_ID","CUDA_VISIBLE_DEVICES="+("0" if gpu else ""),
            "PYTHONNOUSERSITE=1","OMP_NUM_THREADS=4","OPENBLAS_NUM_THREADS=2","MKL_NUM_THREADS=4",
            "PYTHONPATH=.",str(ROOT.parent/"LocalLLM/.local/image-generation/venv/bin/python"),script,*map(str,arguments)]
        with (output/(name+".log")).open("x") as log:
            child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            status(child_command=command)
            try:
                if child.wait():raise RuntimeError(name+" failed; retain artifacts without automatic restart")
            finally:
                if child.poll() is None:
                    os.killpg(child.pid,signal.SIGTERM)
                    try:child.wait(timeout=10)
                    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        if any(sha(ROOT/p)!=v for p,v in code.items()):raise ValueError("bound stage source changed")

    try:
        status()
        while owner(args.comparison_pid)==identity:
            status();time.sleep(15)
        terminal=json.loads((comparison/"runtime.json").read_text())
        if terminal["phase"]!="ready_for_export_and_raw_review" or terminal["child_pid"] is not None:
            raise RuntimeError("comparison exited without its complete registered assessment")
        execute("export_comparison","scripts/export_ink_mass_comparison.py",
            ["--run",comparison,"--output",comparison/"export"],False)
        execute("codec","scripts/probe_native_paired_language.py",["--stage","codec","--output",output/"codec"],True)
        execute("backward","scripts/probe_native_paired_language.py",["--stage","backward","--output",output/"backward"],True)
        controls=ROOT/"artifacts/paired_language_controls_v1"
        snapshot=ROOT/"artifacts/external/image_edit_learning/hub/models--black-forest-labs--FLUX.2-klein-base-4B/snapshots/a3b4f4849157f664bdbc776fd7453c2783562f4d"
        indexes=dict(normal=controls/"preflight_normal_inputs.json",question_erased=controls/"preflight_question_erased_inputs.json",
            passage_erased=controls/"preflight_passage_erased_inputs.json")
        for name,index in indexes.items():
            execute("infer_"+name,"scripts/infer_paired_image_cohort_v2.py",
                ["--snapshot",snapshot,"--condition",ROOT/"artifacts/image_edit_baseline/condition/condition.pt",
                 "--inputs",index,"--checkpoint",ROOT/"artifacts/balanced_autonomous_writer_v1/training/checkpoint.pt",
                 "--output",output/(name+"_inference")],True)
            execute("score_"+name,"scripts/score_paired_writing.py",
                ["--stage","score","--task","qa","--train-limit","32",
                 "--manifest",ROOT/"artifacts/image_edit_learning/pages_v2/pairs.jsonl",
                 "--calibration",controls/"reference_ocr/report.json",
                 "--generations",output/(name+"_inference/outputs.json"),"--output",output/(name+"_ocr")],False)
        # No more GPU children. Authenticate the actual tensors on CPU, including
        # the identical-input/identical-seed erased-question pair.
        os.environ["CUDA_VISIBLE_DEVICES"]=""
        import torch
        from scripts.export_balanced_autonomous_writer import tensor_png
        reports={}
        for name,index in indexes.items():
            folder=output/(name+"_inference")
            rows=json.loads((folder/"outputs.json").read_text());given=json.loads(index.read_text())
            report=json.loads((folder/"report.json").read_text())
            if ([r["id"] for r in rows]!=[r["id"] for r in given] or len(rows)!=(4 if name=="normal" else 2)
                    or report["checkpoint_sha256"]!="fc031796540a4b6ca4c86774a5033e94ae32b415e825b6a221128756748d3575"
                    or any(report[k] for k in ("reference_answer_used","training_manifest_loaded","question_specific_text_condition","output_repair"))):
                raise ValueError("fixed independent input-only preflight cohort differs")
            for row,original in zip(rows,given):
                if row["seed"]!=original["seed"] or row["input_sha256"]!=original["input_sha256"]:raise ValueError("case binding differs")
                p=folder/row["path"];tensor_png(p.parent/"pixels.pt",p,row["result"]["pixels_sha256"],row["sha256"])
            if name=="question_erased":
                a,b=[torch.load((folder/r["path"]).parent/"pixels.pt",map_location="cpu",weights_only=True) for r in rows]
                if not torch.equal(a,b) or rows[0]["sha256"]!=rows[1]["sha256"]:
                    raise ValueError("identical erased inputs/seeds produced unequal actual native outputs")
            reports[name]=dict(inference_report_sha256=sha(folder/"report.json"),
                score_report_sha256=sha(output/(name+"_ocr/report.json")),
                summary=json.loads((output/(name+"_ocr/report.json")).read_text())["summary"])
        save_json(output/"assessment.json",dict(reports=reports,actual_native_generations=8,
            all_eight_pngs_equal_retained_neural_tensors=True,erased_question_pair_outputs_bit_exact=True,
            discarded_optimizer_updates=2,full_page_reconstructions=4,raw_review_completed=False,
            semantic_learning_validated=False,language_quality_qualified=False,general_ilm_goal_complete=False))
        phase="ready_for_raw_review";status(owned_children_exited=True)
    except BaseException as error:
        phase="failed";status(reason=str(error),owned_children_exited=child is None or child.poll() is not None);raise


if __name__=="__main__":main()
