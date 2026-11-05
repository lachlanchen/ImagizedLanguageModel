#!/usr/bin/env python3
"""Bundle-local prompt-image inference; this file is copied as run.py."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def install_isolation_guard(roots, allowed):
    """Reject Python-audited source/data access, network and subprocess calls.

    This is an auditable Python guard, not an operating-system sandbox for
    arbitrary native code. Explicit bundle/prompt/output paths are admitted.
    """
    roots = [Path(p).resolve() for p in roots]
    allowed = [Path(p).resolve() for p in allowed]

    def audit(event, args):
        if event.startswith("socket.") or event == "subprocess.Popen":
            raise PermissionError("standalone replay forbids network and subprocess operations")
        if event == "open" and isinstance(args[0], (str, bytes, os.PathLike)):
            path = Path(os.fsdecode(args[0])).resolve()
            if any(path.is_relative_to(root) for root in roots) and not any(
                    path == item or path.is_relative_to(item) for item in allowed):
                raise PermissionError("standalone replay forbids source/data access: " + str(path))
    sys.addaudithook(audit)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--layout", choices=("page", "strip"), default="page")
    parser.add_argument("--maximum-blocks", type=int, default=128)
    parser.add_argument("--deny-root", type=Path, action="append", default=[])
    args = parser.parse_args()
    bundle = Path(__file__).resolve().parent
    if not (bundle / "manifest.json").is_file():
        raise RuntimeError("run the exported bundle's run.py; source-tree invocation is not deployment")
    if args.out.exists() or not 1 <= args.maximum_blocks <= 128:
        raise ValueError("a new output directory and capacity between 1 and 128 are required")
    if (os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID" or os.environ.get("CUDA_VISIBLE_DEVICES") != "0"
            or os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8"):
        raise ValueError("physical GPU 0 and deterministic cuBLAS required")
    memory = {line.split(":")[0]:int(line.split()[1]) for line in Path("/proc/meminfo").read_text().splitlines()}
    if memory["MemAvailable"] < 24 * 1024**2 or (memory["SwapTotal"] and
            (memory["SwapTotal"] - memory["SwapFree"]) / memory["SwapTotal"] > .75):
        raise RuntimeError("shared workstation host memory limits exceeded")
    used = int(subprocess.check_output(["nvidia-smi", "-i", "0", "--query-gpu=memory.used",
        "--format=csv,noheader,nounits"], text=True).strip())
    if used > 256:
        raise RuntimeError("GPU 0 occupied; preserve its current job")
    # The resource query above is the only child process. Guard before model
    # imports/loading so startup cannot fall back to an external archive.
    if args.deny_root:
        install_isolation_guard(args.deny_root, [bundle, args.image, args.out])
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
    sys.path.insert(0, str(bundle))
    import torch
    from ilm.visual_lm import codec_image_interface, codec_standalone
    for name in ("codec_image_interface", "codec_standalone", "grounded_image_actor",
            "codec_visual_language_actor", "codec_block_writer", "continuous_glyph_codec", "causal_visual_prefix"):
        module = sys.modules["ilm.visual_lm." + name]
        if not Path(module.__file__).resolve().is_relative_to(bundle):
            raise ValueError("runtime module resolved outside its standalone bundle")
    torch.set_num_threads(4)
    prepared = codec_image_interface.prepare_image_prompt(args.image, layout=args.layout)
    receipt = codec_image_interface.save_prepared_prompt(prepared, args.out)
    try:
        torch.manual_seed(20266832)
        torch.cuda.manual_seed_all(20266832)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        begun = time.perf_counter()
        actor, manifest = codec_standalone.load_bundle(bundle, device=torch.device("cuda:0"))
        setup_seconds = time.perf_counter() - begun
        torch.cuda.synchronize()
        begun = time.perf_counter()
        result = actor.generate(prepared.pixels, maximum_blocks=args.maximum_blocks)
        torch.cuda.synchronize()
        generation_seconds = time.perf_counter() - begun
        output = codec_image_interface.save_generation(result, args.out / "generation", maximum_blocks=args.maximum_blocks)
        codec_image_interface.json_file(args.out / "report.json", dict(
            bundle_manifest_sha256=codec_standalone.file_sha(bundle / "manifest.json"),
            final_checkpoint_sha256=manifest["identity"]["checkpoint_sha256"], input=receipt, output=output,
            setup_seconds=setup_seconds, generation_seconds=generation_seconds,
            peak_allocated_vram_gib=torch.cuda.max_memory_allocated() / 1024**3,
            standalone_weights_loaded=True, bundle_local_runtime_modules=True,
            python_isolation_guard_active=bool(args.deny_root),
            operating_system_sandbox=False, model_service_calls=0, ocr_calls=0,
            tokenizer_calls=0, renderer_calls=0, general_language_qualified=False))
        print(json.dumps(dict(stage="generated", blocks=output["blocks"], stopped=output["stopped"])))
    except BaseException as error:
        codec_image_interface.json_file(args.out / "failure.json", dict(error=repr(error)))
        raise


if __name__ == "__main__":
    main()
