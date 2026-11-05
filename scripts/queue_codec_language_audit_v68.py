#!/usr/bin/env python3
"""Wait for the same live trainer, then run its fixed audit and verification.

This coordinator holds no model or CUDA context. It never restarts training,
and it never launches evaluation while the trainer still owns its process.
"""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
WATCHED = ("scripts/queue_codec_language_audit_v68.py", "scripts/eval_codec_language_v68.py",
    "scripts/verify_codec_language_v68.py", "ilm/visual_lm/codec_language_audit.py",
    "ilm/visual_lm/codec_language_runtime.py", "references/codec_language_v68_autonomous_audit.md")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trainer-pid", type=int, required=True)
    parser.add_argument("--run", type=Path, default=ROOT / "artifacts/codec_language_v68_20261105")
    args = parser.parse_args()
    if args.trainer_pid <= 1:
        raise ValueError("a specific observed trainer process is required")
    lock = (args.run / "audit_queue.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    expected = {name:sha(ROOT / name) for name in WATCHED}
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    for name, digest in expected.items():
        original = subprocess.check_output(["git", "show", revision + ":" + name], cwd=ROOT)
        if hashlib.sha256(original).hexdigest() != digest:
            raise ValueError("commit the queued evaluation implementation before launch")
    preflight = args.run / "audit_preflight/attempt01/report.json"
    data_check = json.loads(preflight.read_text())
    if not all(data_check["checks"].values()) or data_check["model_calls"] != 0:
        raise ValueError("actual CPU evaluator preflight must pass first")
    for name, digest in data_check["sources"].items():
        if sha(ROOT / name) != digest:
            raise ValueError("CPU evaluator preflight no longer matches its implementation")
    expected_preflight = sha(preflight)
    last_printed = None

    def status(stage, **fields):
        nonlocal last_printed
        value = dict(stage=stage, pid=os.getpid(), trainer_pid=args.trainer_pid,
            source_commit=revision, sources=expected, observed_at=time.time(), **fields)
        temporary = args.run / "evaluation_queue.tmp"
        temporary.write_text(json.dumps(value, indent=2) + "\n")
        temporary.replace(args.run / "evaluation_queue.json")
        # Persist every observation, but do not flood the console with polls.
        message = (stage, fields.get("trainer_update"))
        if message != last_printed:
            print(json.dumps(value), flush=True)
            last_printed = message

    def same_trainer_alive():
        proc = Path("/proc") / str(args.trainer_pid)
        try:
            command = (proc / "cmdline").read_bytes().split(b"\0")
            cwd = (proc / "cwd").resolve(strict=True)
        except FileNotFoundError:
            return False
        return cwd == ROOT and any(arg.endswith(b"scripts/train_codec_language_v68.py") for arg in command)

    def sources_unchanged():
        if expected != {name:sha(ROOT / name) for name in WATCHED} or sha(preflight) != expected_preflight:
            raise ValueError("queued evaluator or its preflight changed")

    try:
        while True:
            sources_unchanged()
            try:
                alive = same_trainer_alive()
            except OSError as error:
                status("retrying_process_observation", error=repr(error))
                time.sleep(10)
                continue
            state = json.loads((args.run / "status.json").read_text())
            if alive:
                status("waiting_for_same_trainer", trainer_live=True,
                    trainer_update=state.get("update"), trainer_stage=state["stage"])
                time.sleep(10)
                continue
            if not (args.run / "training_report.json").exists():
                raise RuntimeError("observed trainer is gone without a completed training report; preserve its checkpoint")
            training = json.loads((args.run / "training_report.json").read_text())
            if training["update"] != 10000 or state["stage"] != "training_complete":
                raise ValueError("full terminal training is required before evaluation")
            try:
                used = int(subprocess.check_output(["nvidia-smi", "-i", "0", "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits"], text=True, timeout=15).strip())
            except (OSError, subprocess.SubprocessError, ValueError) as error:
                status("retrying_gpu_observation", error=repr(error))
                time.sleep(10)
                continue
            if used > 256:
                status("waiting_for_idle_gpu", used_mib=used)
                time.sleep(10)
                continue
            break
        sources_unchanged()
        environment = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES="0",
            CUBLAS_WORKSPACE_CONFIG=":4096:8", OPENBLAS_NUM_THREADS="4", OMP_NUM_THREADS="4", PYTHONPATH=str(ROOT))

        def child(command, log_name, stage, environment):
            with (args.run / log_name).open("a") as log:
                process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
                while process.poll() is None:
                    status(stage, child_pid=process.pid)
                    time.sleep(10)
                if process.returncode:
                    raise RuntimeError(stage + " exited with code " + str(process.returncode))

        if not (args.run / "development/report.json").exists():
            child([sys.executable, "scripts/eval_codec_language_v68.py", "--run", str(args.run)],
                "autonomous_audit_console.log", "evaluating_final_actor", environment)
        sources_unchanged()
        receipt = args.run / "development/independent_verification.json"
        if not receipt.exists():
            child([sys.executable, "scripts/verify_codec_language_v68.py", "--run", str(args.run), "--out", str(receipt)],
                "autonomous_verification_console.log", "verifying_raw_outputs", dict(environment, CUDA_VISIBLE_DEVICES=""))
        verification = json.loads(receipt.read_text())
        if (not verification["all_verification_checks_pass"]
                or verification["report_sha256"] != sha(args.run / "development/report.json")):
            raise ValueError("independent final verification failed")
        status("complete", bounded_chinese_development_pass=verification["bounded_chinese_development_pass"],
            overall_ilm_goal_complete=False)
    except BaseException as error:
        status("failed", error=repr(error), overall_ilm_goal_complete=False)
        raise
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


if __name__ == "__main__":
    main()
