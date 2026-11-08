#!/usr/bin/env python3
"""After the exact QA coordinator exits, package all outputs for raw review."""
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
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run",type=Path,required=True)
    p.add_argument("--coordinator-pid",type=int,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--export-result",action="store_true",help="also authenticate and export the complete result locally, pending raw review")
    args=p.parse_args();run=args.run.resolve();output=args.output.resolve()
    expected=owner(args.coordinator_pid)
    if expected is None or "scripts/run_native_paired_qa.py" not in expected["command"] or run.name not in expected["command"]:
        raise ValueError("exact currently live QA coordinator required")
    paths=("scripts/finish_native_qa_review.py","scripts/finish_balanced_writer.py",
        "scripts/prepare_blind_image_review.py","scripts/export_balanced_autonomous_writer.py","ilm/image_edit/baseline.py")
    if args.export_result:paths=(*paths,"scripts/export_native_paired_qa.py")
    if subprocess.check_output(["git","status","--porcelain","--",*paths],cwd=ROOT,text=True):
        raise ValueError("commit the exact review completion sources first")
    code={p:sha(ROOT/p) for p in paths}
    output.mkdir(parents=True,exist_ok=False);phase,child="waiting_QA",None;started=time.time()

    def status(**extra):
        save_json(output/"runtime.json",dict(phase=phase,
            worker_pid=None if phase in ("failed","ready_for_transcription_review") else os.getpid(),
            child_pid=None if child is None or child.poll() is not None else child.pid,
            QA_owner=expected,started_unix=started,code_sha256=code,gpu=None,novnc_url=None,
            general_ilm_goal_complete=False,**extra))

    def interrupted(signum,frame):raise InterruptedError("review worker signal "+str(signum))
    for signum in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):signal.signal(signum,interrupted)

    def execute(script,arguments,log_name):
        nonlocal child
        if any(sha(ROOT/p)!=h for p,h in code.items()):raise ValueError("bound CPU completion source changed")
        command=[str(Path.home()/".local/bin/with-current-nvidia"),"env","CUDA_VISIBLE_DEVICES=",
            "PYTHONNOUSERSITE=1","OMP_NUM_THREADS=2","OPENBLAS_NUM_THREADS=2","MKL_NUM_THREADS=2","PYTHONPATH=.",
            str(ROOT.parent/"LocalLLM/.local/image-generation/venv/bin/python"),script,*map(str,arguments)]
        with (output/log_name).open("x") as log:
            child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            status(child_command=command)
            try:
                if child.wait():raise RuntimeError("CPU completion stage failed; preserve partial evidence")
            finally:
                if child.poll() is None:
                    os.killpg(child.pid,signal.SIGTERM)
                    try:child.wait(timeout=10)
                    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        if any(sha(ROOT/p)!=h for p,h in code.items()):raise ValueError("bound CPU completion source changed")
    try:
        while owner(args.coordinator_pid)==expected:
            status();time.sleep(15)
        terminal=json.loads((run/"runtime.json").read_text())
        if terminal["phase"]!="ready_for_raw_review" or terminal["child_pid"] is not None:
            raise RuntimeError("QA ended without its complete registered assessment")
        assessed=json.loads((run/"assessment.json").read_text())
        if assessed["actual_qa_assessment_calls"]!=256 or assessed["actual_final_copying_calls"]!=64 or not assessed["all_training_and_assessment_pngs_equal_actual_tensors"]:
            raise ValueError("complete actual neural assessment required before review preparation")
        if any(sha(ROOT/p)!=h for p,h in code.items()):raise ValueError("review preparation source changed")
        phase="preparing_review"
        arguments=[]
        for arm in ("initial","final"):
            for variant in ("normal","question_erased","passage_erased"):
                arguments.extend(["--index",f"{arm}_{variant}={run/(arm+'_'+variant+'_inference/outputs.json')}"])
        arguments.extend(["--index",f"final_copy={run/'final_copy_inference/outputs.json'}",
            "--index",f"initial_copy={ROOT/'artifacts/ink_mass_comparison_v1/ink_mass_inference/outputs.json'}"])
        execute("scripts/prepare_blind_image_review.py",[*arguments,"--seed","20269001","--output",str(output/"review")],"prepare_review.log")
        receipt=json.loads((output/"review/receipt.json").read_text())
        if receipt["cases"]!=384 or receipt["source_groups"]!=8 or not receipt["all_source_pngs_copied_verbatim"]:
            raise ValueError("all 320 new and 64 reused original answer pages required")
        exported_sha=None
        if args.export_result:
            phase="exporting_complete_result"
            execute("scripts/export_native_paired_qa.py",["--stage","export","--run",run,"--output",output/"export"],"export_result.log")
            exported=json.loads((output/"export/result.json").read_text())
            if exported["complete_output_images"]!=384 or exported["complete_training_images"]!=128 or exported["raw_review_completed"]:
                raise ValueError("complete local evidence pending raw review required")
            exported_sha=sha(output/"export/result.json")
        phase="ready_for_transcription_review"
        status(assessment_sha256=sha(run/"assessment.json"),review_receipt_sha256=sha(output/"review/receipt.json"),
            owned_child_exited=True,raw_annotations_completed=False,new_model_generations=0,
            complete_export_sha256=exported_sha,publication_or_push_performed=False)
    except BaseException as error:
        phase="failed";status(reason=str(error),owned_child_exited=child is None or child.poll() is not None);raise


if __name__=="__main__":main()
