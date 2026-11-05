#!/usr/bin/env python3
"""CPU structural and real-codec checks for the future full native bundle."""

from dataclasses import asdict
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import numpy as np
from PIL import Image
import torch

from ilm.visual_lm.codec_block_writer import load_qualified_writing_codec, CodecBlockWriter
from ilm.visual_lm.codec_image_interface import json_file
from ilm.visual_lm.codec_standalone import (
    ARCHITECTURE, dependencies, empty_actor, file_sha, load_bundle, state_schema, checked_shards, write_shards,
)
from ilm.visual_lm.gguf_language_trunk import load_gguf_reader_source
from ilm.visual_lm.gguf_visual_language import QwenVisualLanguageArchive
from ilm.visual_lm.grounded_image_actor import GroundedImageConfig
from ilm.visual_lm.grounded_image_weights import MODEL_SHA
from scripts.export_codec_language_v68 import MODULES, copy_runtime

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ("references/codec_standalone_plan.md", "scripts/check_codec_standalone.py",
    "scripts/export_codec_language_v68.py", "scripts/run_codec_bundle.py", "scripts/verify_codec_language_bundle.py",
    *["ilm/visual_lm/" + name + ".py" for name in MODULES])

CHILD = r'''
import json,sys
from pathlib import Path
bundle,repository,shared=map(Path,sys.argv[1:])
sys.path.insert(0,str(bundle))
import run
run.install_isolation_guard([repository,shared],[bundle])
import torch
torch.set_num_threads(4)
from ilm.visual_lm.codec_standalone import checked_shards,state_schema
from ilm.visual_lm.continuous_glyph_codec import ContinuousGlyphCodec,ContinuousGlyphCodecConfig
from ilm.visual_lm.codec_block_writer import CodecBlockWriter
fixture=json.loads((bundle/'fixture.json').read_text())
with torch.device('meta'):
    model=ContinuousGlyphCodec(ContinuousGlyphCodecConfig(**fixture['codec_configuration']))
for shard in checked_shards(bundle,fixture['inventory'],fixture['files'],state_schema(model.state_dict())):
    model.load_state_dict(shard,strict=False,assign=True)
if any(p.is_meta for p in model.parameters()): raise ValueError('codec state incomplete')
codes=torch.load(bundle/'probe_codes.pt',map_location='cpu',weights_only=True)
with torch.no_grad(): prediction=CodecBlockWriter(model).predict(codes)
torch.save(prediction,bundle/'child_prediction.pt')
def blocked(call):
    try: call()
    except PermissionError: return True
    raise AssertionError('isolated operation was allowed')
import socket,subprocess
checks=dict(source_access=blocked(lambda:(repository/'ilm/visual_lm/codec_block_writer.py').read_bytes()),
    network=blocked(lambda:socket.socket()),subprocess=blocked(lambda:subprocess.Popen(['true'])),
    local_runtime=all(Path(sys.modules['ilm.visual_lm.'+name].__file__).resolve().is_relative_to(bundle)
        for name in ('codec_standalone','grounded_image_actor','codec_block_writer','codec_visual_language_actor')),
    no_cuda=not torch.cuda.is_initialized())
(bundle/'child_checks.json').write_text(json.dumps(checks))
'''


def rejected(call):
    try:
        call()
    except ValueError:
        return True
    raise AssertionError("malformed standalone state was accepted")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or torch.cuda.is_initialized():
        raise ValueError("CPU-only standalone preflight required")
    args.out.mkdir(parents=True, exist_ok=False)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    sources = {name:file_sha(ROOT / name) for name in SOURCES}
    for name, digest in sources.items():
        if hashlib.sha256(subprocess.check_output(["git", "show", revision + ":" + name], cwd=ROOT)).hexdigest() != digest:
            raise ValueError("commit standalone implementation before CPU checks")
    begun = time.perf_counter()
    torch.set_num_threads(4)
    preflight = ROOT / "artifacts/codec_visual_actor_preflight_20261105/attempt01/report.json"
    if file_sha(preflight) != "960a79cd02eab2b9f084a926dc137bef275356cec3ae9df26a567acc33d7f0ba":
        raise ValueError("original complete-model preflight changed")
    original = json.loads(preflight.read_text())
    codec = load_qualified_writing_codec(ROOT / "artifacts/continuous_glyph_codec_v34_20260814/checkpoint_latest.pt", "cpu")
    configuration = dict(native=original["provenance"]["configurations"],
        grounded=asdict(GroundedImageConfig()), codec=asdict(codec.config))
    actor = empty_actor(configuration)
    schema = state_schema(actor.state_dict())
    checks = dict(complete_native_parameter_counts_and_meta_storage=True)
    blob = ROOT.parent / "LocalLLM/.local/models/ollama/blobs" / ("sha256-" + MODEL_SHA)
    if file_sha(blob) != MODEL_SHA:
        raise ValueError("authenticated source archive changed")
    archive = QwenVisualLanguageArchive(blob, load_gguf_reader_source(
        ROOT.parent / "LocalLLM/.local/tools/llama.cpp/gguf-py"))
    native_names = set()
    for name, shape in archive.shapes().items():
        if name.startswith("visual."):
            target = "actor.vision." + name.removeprefix("visual.")
            dtype = "torch.float32"
        else:
            target = "actor.context." + name.removeprefix("language_model.")
            for projection in ("q_proj", "v_proj", "o_proj"):
                target = target.replace(".self_attn." + projection + ".", ".self_attn." + projection + ".base.")
            dtype = "torch.bfloat16"
        if schema[target]["shape"] != list(shape) or schema[target]["dtype"] != dtype:
            raise ValueError("native bundle layout differs from actual authenticated archive")
        native_names.add(target)
    trainable_names = {name for name,value in actor.named_parameters() if value.requires_grad}
    codec_names = {"actor.writer.codec." + name for name in codec.state_dict()}
    if (native_names & trainable_names or native_names & codec_names or trainable_names & codec_names
            or native_names | trainable_names | codec_names != set(schema)):
        raise ValueError("full state is not exactly native weights, codec and learned adaptation")
    checks["all_748_native_shapes_dtypes_and_complete_state_partition"] = len(native_names) == 748
    codes_path = ROOT / "artifacts/codec_output_v67_20261105/continuous_codes.pt"
    if file_sha(codes_path) != "af093026858db1351ec9050a4b06e27807c4a5fc20fe5ebf7bdc374039d4ffe5":
        raise ValueError("original supplied writing codes changed")
    codes = torch.load(codes_path, map_location="cpu", weights_only=True)[:2].clone()
    with torch.no_grad():
        expected = CodecBlockWriter(codec).predict(codes)
    for index in range(2):
        with Image.open(ROOT / f"artifacts/codec_output_v67_20261105/aligned/{index:03d}.png") as picture:
            if not np.array_equal((expected[index, 0].numpy() >= 0) * 255, np.asarray(picture)):
                raise ValueError("supplied-code physical output differs from the original audit")
    checks["two_original_supplied_code_images_reproduced"] = True
    with tempfile.TemporaryDirectory(prefix="native-codec-", dir=args.out) as temporary:
        temporary = Path(temporary)
        bundle = temporary / "bundle"
        inventory, files = write_shards(codec.state_dict(), bundle, limit_bytes=8 * 1024**2)
        files.update(copy_runtime(bundle))
        fixture = dict(codec_configuration=configuration["codec"], inventory=inventory, files=files)
        json_file(bundle / "fixture.json", fixture)
        torch.save(codes, bundle / "probe_codes.pt")
        empty = temporary / "empty_working_directory"
        empty.mkdir()
        environment = dict(os.environ, PYTHONPATH="", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
            HF_HUB_DISABLE_TELEMETRY="1", HF_HOME=str(temporary / "empty_hub"))
        with (args.out / "isolated_codec_console.log").open("w") as log:
            subprocess.run([sys.executable, "-I", "-B", "-c", CHILD, str(bundle.resolve()), str(ROOT),
                str(ROOT.parent / "LocalLLM")], cwd=empty, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
        recovered = torch.load(bundle / "child_prediction.pt", map_location="cpu", weights_only=True)
        if not torch.equal(expected, recovered):
            raise ValueError("isolated native codec floats differ after exact shard loading")
        child = json.loads((bundle / "child_checks.json").read_text())
        if not all(child.values()):
            raise AssertionError(child)
        checks["isolated_actual_codec_state_and_continuous_output_exact"] = True
        checks["python_source_network_subprocess_guards_and_local_imports"] = True
        rejected(lambda:list(checked_shards(bundle, {k:v for k,v in list(inventory.items())[1:]}, files, state_schema(codec.state_dict()))))
        relative = next(iter(files))
        contents = (bundle / relative).read_bytes()
        altered = bytearray(contents)
        altered[-1] ^= 1
        (bundle / relative).write_bytes(altered)
        rejected(lambda:list(checked_shards(bundle, inventory, files, state_schema(codec.state_dict()))))
        (bundle / relative).write_bytes(contents)
        values = torch.load(bundle / relative, map_location="cpu", weights_only=True)
        values["unexpected_copy"] = next(iter(values.values()))
        torch.save(values, bundle / relative)
        changed_files = dict(files, **{relative:file_sha(bundle / relative)})
        rejected(lambda:list(checked_shards(bundle, inventory, changed_files, state_schema(codec.state_dict()))))
        (bundle / relative).write_bytes(contents)
        checks["missing_corrupted_and_unlisted_shard_state_rejected"] = True
        json_file(bundle / "manifest.json", dict(architecture=ARCHITECTURE, dependencies=dependencies(),
            configuration=configuration, tensors=inventory, files_sha256=files))
        checks["codec_only_bundle_rejected_as_full_actor"] = rejected(lambda:load_bundle(bundle, device="cpu"))
    destination = args.out / "unfinished_export_rejected"
    completed = subprocess.run([sys.executable, "scripts/export_codec_language_v68.py", "--out", str(destination)],
        cwd=ROOT, capture_output=True, text=True)
    if not completed.returncode or "V68 training is unfinished" not in completed.stderr or destination.exists():
        raise ValueError("actual unfinished run was not rejected before export")
    checks["actual_unfinished_export_rejected_without_cuda"] = not torch.cuda.is_initialized()
    if not all(checks.values()):
        raise AssertionError(checks)
    json_file(args.out / "report.json", dict(checks=checks, source_commit=revision, source_sha256=sources,
        complete_actor_parameters=7539239282, complete_trainable_parameters=9351937,
        native_tensors_checked=len(native_names), complete_state_tensors=len(schema),
        complete_weight_storage_bytes=sum(row["bytes"] for row in schema.values()),
        actual_codec_parameters=7423361, decoder_forward_calls=2, supplied_code_blocks=2,
        complete_context_forward_calls=0, foundation_weight_values_materialized=0,
        complete_bundle_created=False, temporary_codec_weight_files_removed=True,
        gpu_allocations=0, optimization_updates=0, model_service_calls=0,
        full_actor_isolation_verified=False, language_quality_claim=False,
        general_ilm_goal_complete=False, elapsed_seconds=time.perf_counter() - begun))
    print(json.dumps(json.loads((args.out / "report.json").read_text()), indent=2))


if __name__ == "__main__":
    main()
