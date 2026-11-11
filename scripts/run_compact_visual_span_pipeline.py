#!/usr/bin/env python3
"""Run teacher extraction, full-sequence fitting and evaluation on one GPU."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from ilm.image_edit.baseline import ROOT, save_json, sha
from scripts.score_pixel_language_first_blocks import host_resources


SCRIPTS = {
    "teacher": ROOT / "scripts/cache_visual_sequence_teacher_states.py",
    "fit": ROOT / "scripts/train_compact_visual_span.py",
    "infer": ROOT / "scripts/infer_compact_visual_span.py",
    "review": ROOT / "scripts/review_compact_visual_span.py",
}
GPU_LOCK = ROOT / "artifacts/ilm_gpu0.lock"


def load(path):
    return json.loads(path.read_text())


def live_start_ticks(process):
    try:
        fields = (process / "stat").read_text().rsplit(")", 1)[1].split()
    except (FileNotFoundError, ProcessLookupError):
        return None
    return fields[19] if fields[0] not in {"Z", "X"} else None


def stage_progress(stage, path):
    if stage == "teacher":
        return len(list(path.glob("shard_*/report.json")))
    if stage == "fit":
        saved = path / "checkpoint_latest.pt"
        return saved.stat().st_mtime_ns if saved.is_file() else 0
    return 0


def host_ready_for_stage(stage, host):
    if stage in {"teacher", "fit"}:
        # These workers repeat full_reader_gpu admission, including the owned
        # process audit. That established path records high global swap while
        # admitting only with the RAM reserve and low actual memory stalls.
        return (host["available_gib"] >= 48 and host["some_pressure_avg10"] <= .5
                and host["full_pressure_avg10"] <= .1)
    # Standalone inference currently uses gpu_zero's original RAM/swap gate.
    return host["available_gib"] >= 24 and host["swap_fraction"] <= .75


class Pipeline:
    def __init__(self, directory, identity):
        self.directory = directory
        self.identity = identity
        self.began = time.monotonic()
        directory.mkdir(parents=True, exist_ok=True)
        self.lock = (directory / "owner.lock").open("a")

    def status(self, phase, **details):
        save_json(self.directory / "status.json", {
            "phase": phase, "pid": os.getpid(), "elapsed_seconds": time.monotonic() - self.began, **details,
        })

    def wait_for_predecessor(self, pid):
        process = Path("/proc") / str(pid)
        start_ticks = live_start_ticks(process)
        if start_ticks is None:
            return
        try:
            workdir = (process / "cwd").resolve(strict=True)
        except (FileNotFoundError, ProcessLookupError):
            return
        if not workdir.is_relative_to(ROOT):
            raise ValueError("The predecessor must be a process in this project")
        # The start time distinguishes the original process from a reused PID.
        while live_start_ticks(process) == start_ticks:
            self.status("waiting_for_predecessor", predecessor_pid=pid, predecessor_start_ticks=start_ticks)
            time.sleep(30)

    def wait_for_gpu(self, stage):
        while True:
            host = host_resources()
            free = int(subprocess.check_output([
                "nvidia-smi", "-i", "0", "--query-gpu=memory.free", "--format=csv,noheader,nounits",
            ], text=True).strip())
            with GPU_LOCK.open("a") as lock:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    idle = True
                except BlockingIOError:
                    idle = False
            # Match the stage's existing admission checks. The child repeats
            # them and owns the mutex throughout its use of the device.
            if idle and free >= 20 * 1024 and host_ready_for_stage(stage, host):
                return
            self.status("waiting_for_gpu", stage=stage, free_vram_mib=free, gpu_mutex_free=idle, resources=host)
            time.sleep(30)

    def child(self, stage, command, *, gpu=True):
        if gpu:
            self.wait_for_gpu(stage)
        for name, digest in self.identity["sources_sha256"].items():
            if sha(ROOT / name) != digest:
                raise ValueError(f"Queued source changed before execution: {name}")
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="0" if gpu else "", PYTHONPATH=str(ROOT))
        env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        path = self.directory / f"{stage}_{time.time_ns()}.log"
        with path.open("x") as stream:
            child = subprocess.Popen([sys.executable, "-u", *map(str, command)], cwd=ROOT, env=env,
                                     stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                self.status("running", stage=stage, child_pid=child.pid, log=path.name)
                return child.wait(), path
            except BaseException:
                if child.poll() is None:
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait()
                raise

    def resumable_stage(self, stage, command, directory, terminal_report):
        while not terminal_report.is_file():
            before = stage_progress(stage, directory)
            code, log = self.child(stage, command)
            if code == 0 and terminal_report.is_file():
                return
            # Only resource deadlines with a newly committed checkpoint or
            # shard are retried. Keep the same recipe and all previous logs.
            with log.open("rb") as stream:
                stream.seek(max(0, log.stat().st_size - 8192))
                tail = stream.read().decode(errors="replace")
            if (stage_progress(stage, directory) <= before
                    or not any(reason in tail for reason in ("stage time cap reached", "pressure wait exhausted"))):
                raise RuntimeError(f"{stage} did not complete; inspect {log.name}")

    def execute(self, args):
        manifest = args.evaluation / "inputs/manifest.json"
        if sha(manifest) != self.identity["evaluation_manifest_sha256"]:
            raise ValueError("Queued evaluation input manifest changed")
        cache = args.teacher / "cache"
        self.resumable_stage("teacher", [SCRIPTS["teacher"], "--stage", "cache", "--run", args.teacher],
                             cache, cache / "report.json")
        self.resumable_stage("fit", [SCRIPTS["fit"], "--teacher-run", args.teacher, "--run", args.fit],
                             args.fit, args.fit / "report.json")
        fit = load(args.fit / "report.json")
        checkpoints = sorted(args.fit.glob("image_heads_step_*.pt"))
        if (len(checkpoints) != 5 or checkpoints[0].name != "image_heads_step_0000000.pt"
                or checkpoints[-1].name != fit["final_image_heads"]
                or sha(checkpoints[-1]) != fit["final_image_heads_sha256"]):
            raise ValueError("Fit does not contain the five registered checkpoint snapshots")
        runs = []
        for index, checkpoint in enumerate(checkpoints):
            if sha(manifest) != self.identity["evaluation_manifest_sha256"]:
                raise ValueError("Queued evaluation input manifest changed")
            output = self.directory / checkpoint.stem
            runs.append(output)
            if not (output / "report.json").is_file():
                command = [SCRIPTS["infer"], "--manifest", manifest,
                           "--checkpoint", checkpoint, "--output", output]
                if 0 < index < len(checkpoints) - 1:
                    command.append("--normal-only")
                code, log = self.child(checkpoint.stem, command)
                if code or not (output / "report.json").is_file():
                    raise RuntimeError(f"Checkpoint evaluation failed; inspect {log.name}")
            if load(output / "request.json")["checkpoint_sha256"] != sha(checkpoint):
                raise ValueError("Generated answers use a different checkpoint")
        review = self.directory / "review"
        if not (review / "mapping.json").is_file():
            command = [SCRIPTS["review"], "--evaluation", args.evaluation,
                       "prepare-review", "--output", review, "--runs", *runs]
            code, log = self.child("review", command, gpu=False)
            if code or not (review / "mapping.json").is_file():
                raise RuntimeError(f"Blind review preparation failed; inspect {log.name}")
        self.status("awaiting_blind_readings", review=str(review), training_updates=fit["updates"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    artifacts = ROOT / "artifacts"
    parser.add_argument("--teacher", type=Path, default=artifacts / "visual_sequence_teacher_v1")
    parser.add_argument("--fit", type=Path, default=artifacts / "compact_visual_span_fit_v1")
    parser.add_argument("--evaluation", type=Path, default=artifacts / "compact_visual_span_evaluation_v1")
    parser.add_argument("--pipeline", type=Path, default=artifacts / "compact_visual_span_pipeline_v2")
    parser.add_argument("--after-pid", type=int)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0" or os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
        parser.error("Set CUDA_DEVICE_ORDER=PCI_BUS_ID and CUDA_VISIBLE_DEVICES=0")
    if args.after_pid is not None and args.after_pid <= 0:
        parser.error("The predecessor PID must be positive")
    args.teacher, args.fit, args.evaluation, args.pipeline = (
        path.resolve() for path in (args.teacher, args.fit, args.evaluation, args.pipeline)
    )
    preparation = args.teacher / "preparation.json"
    evaluation = args.evaluation / "preparation.json"
    registered = load(evaluation)
    if load(preparation)["data_report_sha256"] != registered["visual_corpus_report_sha256"]:
        raise ValueError("Teacher and held-out evaluation preparations bind different visual corpora")
    manifest_sha = registered["input_manifest_sha256"]
    if sha(args.evaluation / "inputs/manifest.json") != manifest_sha:
        raise ValueError("Registered evaluation input manifest changed")
    sources = [str(path.relative_to(ROOT)) for path in SCRIPTS.values()]
    sources.append("scripts/run_compact_visual_span_pipeline.py")
    sources.extend(f"ilm/visual_lm/compact_visual_span_{name}.py" for name in ("model", "training", "weights"))
    identity = {"teacher_preparation_sha256": sha(preparation), "evaluation_preparation_sha256": sha(evaluation),
                "evaluation_manifest_sha256": manifest_sha,
                "teacher": str(args.teacher.resolve()), "fit": str(args.fit.resolve()),
                "evaluation": str(args.evaluation.resolve()), "sources_sha256": {name: sha(ROOT / name) for name in sources}}
    pipeline = Pipeline(args.pipeline, identity)
    with pipeline.lock:
        fcntl.flock(pipeline.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        request = args.pipeline / "request.json"
        if request.exists() and load(request) != identity:
            raise ValueError("Resume requires the same registered pipeline inputs and sources")
        save_json(request, identity)

        def terminate(_signum, _frame):
            raise SystemExit("Pipeline termination requested")

        signal.signal(signal.SIGTERM, terminate)
        try:
            if args.after_pid is not None:
                pipeline.wait_for_predecessor(args.after_pid)
            pipeline.execute(args)
        except BaseException as error:
            pipeline.status("failed", reason=repr(error))
            raise
        finally:
            status = args.pipeline / "status.json"
            if status.is_file():
                save_json(status, {**load(status), "pid": None})


if __name__ == "__main__":
    main()
