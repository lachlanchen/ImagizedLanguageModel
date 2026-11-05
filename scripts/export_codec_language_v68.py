#!/usr/bin/env python3
"""Export the audited complete V68 actor into native, self-contained shards."""

import argparse
import gc
import os
from pathlib import Path
import shutil
import subprocess

import torch

from ilm.visual_lm.codec_image_interface import json_file
from ilm.visual_lm.codec_standalone import (
    ARCHITECTURE, TOTAL_PARAMETERS, actor_configuration, dependencies, empty_actor,
    file_sha, state_schema, write_shards,
)
from scripts.infer_codec_language_v68 import audited_identity

ROOT = Path(__file__).resolve().parents[1]
MODULES = ("codec_image_interface", "codec_standalone", "grounded_image_actor",
    "codec_visual_language_actor", "codec_block_writer", "continuous_glyph_codec", "causal_visual_prefix")


def copy_runtime(bundle):
    package = bundle / "ilm/visual_lm"
    package.mkdir(parents=True, exist_ok=False)
    (bundle / "ilm/__init__.py").write_text("")
    (package / "__init__.py").write_text("")
    for name in MODULES:
        shutil.copy2(ROOT / "ilm/visual_lm" / (name + ".py"), package / (name + ".py"))
    shutil.copy2(ROOT / "scripts/run_codec_bundle.py", bundle / "run.py")
    return {str(p.relative_to(bundle)):file_sha(p) for p in sorted(bundle.rglob("*.py"))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "artifacts/codec_language_v68_20261105")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    identity = audited_identity(args.run)
    if not identity["bounded_chinese_development_pass"]:
        raise RuntimeError("preserve failed language evidence; standalone export requires the original bounded gates")
    if args.out.exists():
        raise FileExistsError("completed or partial export evidence is immutable")
    if (os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID" or os.environ.get("CUDA_VISIBLE_DEVICES") != "0"
            or os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8"):
        raise ValueError("explicit physical GPU 0 and deterministic cuBLAS required")
    from ilm.visual_lm.blind_vision_transcription import check_host_memory, gpu_memory
    from ilm.visual_lm.codec_language_runtime import load_completed_actor
    check_host_memory()
    if gpu_memory() > 256:
        raise RuntimeError("GPU 0 occupied; preserve the current project job")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    names = ["scripts/export_codec_language_v68.py", "scripts/run_codec_bundle.py",
        *["ilm/visual_lm/" + name + ".py" for name in MODULES]]
    sources = {name:file_sha(ROOT / name) for name in names}
    import hashlib
    for name, digest in sources.items():
        if hashlib.sha256(subprocess.check_output(["git", "show", revision + ":" + name], cwd=ROOT)).hexdigest() != digest:
            raise ValueError("commit standalone implementation before export")
    torch.set_num_threads(4)
    torch.manual_seed(20266832)
    actor, training = load_completed_actor(args.run, root=ROOT, device=torch.device("cuda:0"))
    configuration = actor_configuration(actor, training["initialization"]["configurations"])
    skeleton = empty_actor(configuration)
    if state_schema(actor.state_dict()) != state_schema(skeleton.state_dict()):
        raise ValueError("standalone schema differs from the actual trained actor")
    del skeleton
    gc.collect()
    inventory, files = write_shards(actor.state_dict(), args.out)
    files.update(copy_runtime(args.out))
    if sources != {name:file_sha(ROOT / name) for name in sources}:
        raise ValueError("standalone sources changed during export")
    json_file(args.out / "manifest.json", dict(architecture=ARCHITECTURE, identity=identity,
        configuration=configuration, dependencies=dependencies(), total_parameters=TOTAL_PARAMETERS,
        tensors=inventory, files_sha256=files, source_commit=revision, source_sha256=sources,
        external_initialization=training["initialization"], codec_training_checkpoint_sha256=
            "a138c9cb3b0502e43d1227f689c020893d56b468742c32e1840e44d299662f33",
        physical_weight_bytes=sum(row["bytes"] for row in inventory.values()),
        source_archive_required_at_inference=False, optimizer_exported=False,
        vocabulary_exported=False, tokenizer_exported=False, training_data_exported=False,
        standalone_isolation_verified=False, general_language_qualified=False))
    print(args.out)


if __name__ == "__main__":
    main()
