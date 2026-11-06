#!/usr/bin/env python3
"""Export the verified source-reader diagnostic without image-output credit."""

import json
import os

import torch

from scripts.cache_semantic_images_v72 import ROOT, sha, atomic_json, committed_identity


def main():
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="" or torch.cuda.is_initialized():
        raise ValueError("CPU-only evidence export required")
    run=ROOT/"artifacts/reader_content_v77_20261106"
    verified=json.loads((run/"verification.json").read_text())
    report=json.loads((run/"generation_report.json").read_text())
    prep=json.loads((run/"preparation.json").read_text())
    if (verified["generation_report_sha256"]!=sha(run/"generation_report.json")
            or verified["preparation_sha256"]!=sha(run/"preparation.json")
            or not verified["offline_checks_pass"] or verified["normal_memory_matches"]!=44
            or report["helper_generations"]!=100 or verified["independent_image_actor_calls"]!=0):
        raise ValueError("entire verified offline diagnostic required")
    for name,digest in report["identity"]["sources"].items():
        if sha(ROOT/name)!=digest:
            raise ValueError("registered diagnostic implementation changed")
    for name,digest in report["raw_files_sha256"].items():
        if sha(run/name)!=digest:
            raise ValueError("raw source helper result changed")
    identity=committed_identity(("scripts/export_reader_content_v77.py",))
    value=dict(export_identity=identity,source_run=str(run.relative_to(ROOT)),
        generation_report_sha256=sha(run/"generation_report.json"),
        preparation_sha256=sha(run/"preparation.json"),verification_sha256=sha(run/"verification.json"),
        generation={k:report[k] for k in ("identity","provenance","elapsed_seconds","setup_seconds",
            "peak_allocated_vram_gib","helper_generations","normal_memory_matches")},verification=verified,
        source_copy_verification_sha256=prep["copy_verification_sha256"],
        source_fit_verification_sha256=prep["fit_verification_sha256"],
        external_pretraining=True,source_helper_only=True,compact_memory_sufficiency_proven=False,
        model_output_pixels_repaired=False,language_quality_claim=False,general_ilm_goal_complete=False)
    atomic_json(ROOT/"references/evidence/reader_content_v77_result.json",value)
    rows=[]
    for key,row in verified["summaries"].items():
        rows.append(f"| {key} | {row['exact']}/{row['cases']} | {row['owning_reference_included']}/{row['cases']} | {row['first_piece_matches_prefix']}/{row['cases']} |")
    text='''# Source-reader content on the same failed image prompts

The offline source decoder and independent image writer share exactly the same
44 normal-image prefill memories. The source decoder nevertheless uses the
full layerwise visual/context KV history for continuation. These results
therefore measure imported source capability and the information boundary;
they do not qualify the ILM or prove the 52 final-layer states are sufficient.

| Source-helper route | Exact canonical answers | Owning reference included | First-piece prefix match |
| --- | ---: | ---: | ---: |
'''+"\n".join(rows)+f'''

The corresponding V75 image writer produces 0/12 exact new-copy answers and
0/32 exact answers on this fixed training cohort. Those original outputs and
scores are unchanged. Helper exactness requires learned source termination and
outer-whitespace-trimmed equality to the canonical reference; quotations,
explanations and alternative wording are not silently removed. Inclusion and
prefix agreement are separate string diagnostics, not semantic accuracy.

There are 100 fixed source-helper generations: normal-image and source-text
controls for all 44 cases, plus fully blanked prompt images for the 12 copy
requests. Every source prompt and reference re-renders to the original physical
strip exactly. On all 44 normal-image calls, the complete 52-state BF16 suffix
matches the failed ILM reader memory byte for byte. The first source projection
uses exactly its last stored BF16 vector; subsequent helper decoding has the
full contextual KV advantage. It uses the authenticated source tokenizer,
token-row lookup and temporary output projection only inside this offline
evaluator. No symbolic decoder or teacher service enters the deployed ILM.

Independent CPU verification reconstructs all 56 source/transformed image
inputs, recompiles controls, checks every retained response and recomputes
literal metrics. It replays {verified['helper_projection_positions']} source
projection decisions from retained hidden states; exact argmax agreements are
{verified['helper_projection_argmax_agreements']}, with
{verified['numerical_near_tie_disagreements']} near-tie disagreements.
It does not replay the full vision/context body. All raw responses remain
unchanged, including wrong, quoted and explanatory outputs. No optimization
or sealed-data access occurs.

The helper stage takes {report['elapsed_seconds']:.3f} seconds, including
{report['setup_seconds']:.3f} seconds setup, at
{report['peak_allocated_vram_gib']:.3f} GiB allocated memory on GPU 0. This cost
excludes all external pretraining and earlier ILM/codec learning. The source
text route and image-only runtime have different interfaces; this is not an
efficiency comparison with a text language model.

The [receipt with all unmodified answers](evidence/reader_content_v77_result.json)
and [registered protocol](reader_content_v77_protocol.md) define the comparison.
Full retained input PNGs, states and decoding traces are under
artifacts/reader_content_v77_20261106/. These sources do not establish a useful
independent image-output language model.
'''
    (ROOT/"references/reader_content_v77_result.md").write_text(text)
    print(json.dumps(verified["summaries"],indent=2))


if __name__=="__main__":
    main()
