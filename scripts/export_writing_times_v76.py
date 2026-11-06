#!/usr/bin/env python3
"""Retain both outcomes of the failed matched writing-time repair."""

import json
import os
import shutil

import torch

from scripts.cache_semantic_images_v72 import ROOT, sha, atomic_json, committed_identity


def main():
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="" or torch.cuda.is_initialized():
        raise ValueError("CPU-only result export required")
    out=ROOT/"artifacts/writing_times_v76_20261106"
    value=json.loads((out/"verification.json").read_text())
    if (not value["all_64_streams_independently_scored"] or not value["matched_common_draws"]
            or value["training_fit_repair_pass"] or value["general_ilm_goal_complete"]):
        raise ValueError("expected the completed rejected matched repair")
    for name,digest in value["identity"]["sources"].items():
        if sha(ROOT/name)!=digest:
            raise ValueError("registered repair implementation changed")
    identity=committed_identity(("scripts/export_writing_times_v76.py",))
    output=ROOT/"references/evidence/writing_times_v76_raw"
    output.mkdir(exist_ok=True)
    raw={}
    training={}
    for law in ("logit_normal","uniform"):
        report=json.loads((out/law/"evaluation.json").read_text())
        training[law]=json.loads((out/law/"training_report.json").read_text())
        if (sha(out/law/"evaluation.json")!=value["source_report_sha256"][law]
                or sha(out/law/"training_report.json")!=report["training_report_sha256"]
                or sha(out/law/"head_latest.pt")!=training[law]["head_sha256"]):
            raise ValueError("verified final head/report changed")
        for i in range(32):
            folder=out/law/"predictions"/f"{i:02}"
            header=json.loads((folder/"header.json").read_text())
            for name,digest in header["files_sha256"].items():
                if sha(folder/name)!=digest:
                    raise ValueError("raw answer changed")
            if i<2:
                for name in ("answer","reference"):
                    target=output/f"{law}_{i:02}_{name}.png"
                    shutil.copy2(folder/(name+".png"),target)
                    raw[str(target.relative_to(ROOT))]=sha(target)
    compact=dict(export_identity=identity,source_run=str(out.relative_to(ROOT)),
        verification=value,verification_sha256=sha(out/"verification.json"),training=training,
        public_raw_files_sha256=raw,raw_answer_pixels_repaired=False,
        language_quality_claim=False,general_ilm_goal_complete=False)
    atomic_json(ROOT/"references/evidence/writing_times_v76_result.json",compact)
    control,uniform=(value["summaries"][law] for law in ("logit_normal","uniform"))
    text=f'''# Better near-clean reconstruction does not repair free answers

The matched writing-head intervention **fails all four prespecified training-fit
criteria**. Both arms still produce **0/32 exact free answers** on the fixed
training questions. Continued logit-normal training reaches
{control['free_f1']*100:.3f}% foreground F1; uniform-time training reaches
{uniform['free_f1']*100:.3f}%, a difference of
{(uniform['free_f1']-control['free_f1'])*100:.3f} points. All answers stop.
The uniform head is rejected; the failed original candidate is not upgraded.

At the constructed target-assisted t=31/32 probe, uniform-time training improves
near-clean first-block F1 to {uniform['near_clean'][0]['foreground_f1']*100:.4f}%
and exact blocks to {uniform['near_clean'][0]['exact']}/32, compared with
{control['near_clean'][0]['foreground_f1']*100:.4f}% and
{control['near_clean'][0]['exact']}/32 for continued logit-normal training.
Improved assisted high-signal reconstruction therefore does not establish a
repair of free answer generation. The time schedule alone is not a sufficient
solution under this fixed intervention. This does not isolate every remaining
cause or rule out other objectives or training budgets.

Each arm starts from the same completed full-corpus V75 checkpoint and trains
only the same 36,403,200-parameter head for 4,000 updates, using all 12,976
training records and 64,000 draws. All causal-condition and stopping parameters
are unchanged. Nine checkpoints of selected examples, positions, noise and
standard-normal time variates match exactly. Both arms sample
{control['sampled_training_blocks']:,} writing blocks. The original law realizes
{control['realized_high_signal_training_blocks']} times at or above 31/32;
the uniform law realizes {uniform['realized_high_signal_training_blocks']:,}.
Exposure and effective loss weighting both change with the law.

Training takes {control['training_seconds']:.3f} and
{uniform['training_seconds']:.3f} active seconds, respectively, excluding
earlier codec, reader, feature preparation and full-corpus writer training.
Evaluation reuses the 32 previously verified prompt-image memories; it makes
no new inherited-reader calls. Free generation receives no target or length,
uses the original 32-step sampler, re-encodes emitted pixels and retains every
output. The near-clean probe explicitly receives noisy target codes and gets
no autonomous credit. These are training cases, not unseen language validation.

Independent CPU scoring reconstructs original references, authenticates all
64 free streams, re-decodes retained near-clean predictions and validates
frozen-body hashes and common-random-number traces. It does not replay the
whole neural sampler. Both failed outcomes and the first two fixed raw
answers per arm are retained without output repair.

See the [registered protocol](writing_time_coverage_v76_protocol.md) and
[source-bound receipt](evidence/writing_times_v76_result.json). Full raw
predictions and head/optimizer states remain under
artifacts/writing_times_v76_20261106/. The general ILM goal remains unachieved.
'''
    (ROOT/"references/writing_times_v76_result.md").write_text(text)
    print(json.dumps(value["summaries"],indent=2))


if __name__=="__main__":
    main()
