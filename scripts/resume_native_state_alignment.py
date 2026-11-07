#!/usr/bin/env python3
"""Resume an unchanged snapshot with its complete authenticated optimizer log."""
import argparse
import json
import os
from pathlib import Path
import traceback

import torch
from ilm.image_edit.baseline import save_json, sha
from probe_native_unified_source import gpu_slot
from train_native_state_alignment import learn, binding, BUDGET


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--assets", type=Path, default=Path("artifacts/external/native_unified_source"))
    p.add_argument("--prepared", type=Path, default=Path("artifacts/native_state_alignment/prepared_v1"))
    p.add_argument("--cache", type=Path, default=Path("artifacts/native_state_alignment/cache_v1"))
    p.add_argument("--resume", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    saved = torch.load(args.resume, map_location="cpu", weights_only=True)
    count = saved["completed_updates"]
    if saved["metadata"]["implementation"] != binding() or not 0 < count < BUDGET or saved["metadata"]["budget"] != BUDGET:
        raise ValueError("unchanged incomplete main-run checkpoint required")
    journal = args.resume.parent / "updates.jsonl"
    records = journal.read_text().splitlines()
    prefix = records[:count]
    if len(prefix) != count or [json.loads(s)["update"] for s in prefix] != list(range(1, count + 1)):
        raise ValueError("complete contiguous source checkpoint journal required")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "updates.jsonl").write_text("\n".join(prefix) + "\n")
    save_json(args.output / "resume_source.json", dict(checkpoint_sha256=sha(args.resume),
        source_journal_sha256=sha(journal), retained_updates=count,
        uncheckpointed_tail_records_preserved_in_original=len(records) - count,
        wrapper_sha256=sha(__file__), original_directory_preserved=True))
    args.stage = "train"
    save_json(args.output / "status.json", dict(state="running", stage="train", pid=os.getpid()))
    torch.set_num_threads(8)
    try:
        with gpu_slot():
            learn(args)
        save_json(args.output / "status.json", dict(state="completed", stage="train", pid=None))
    except BaseException as error:
        save_json(args.output / "status.json", dict(state="failed", stage="train", pid=None, error=repr(error), traceback=traceback.format_exc()))
        raise


if __name__ == "__main__":
    main()
