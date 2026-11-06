"""Pinned real image-generator components and a fixed offline task condition.

This module does not import a neural framework until a model stage is admitted.
No varying text prompt is accepted by the generation interface.
"""
from contextlib import contextmanager
from importlib.metadata import version
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
REPO = "black-forest-labs/FLUX.2-klein-4B"
REVISION = "e7b7dc27f91deacad38e78976d1f2b499d76a294"
PREDECESSOR = ROOT / "artifacts/decoder_metric_v82r2_20261107"
CACHE = ROOT / "artifacts/external/image_edit_baseline/hub"
SNAPSHOT = CACHE / "models--black-forest-labs--FLUX.2-klein-4B/snapshots" / REVISION
CONDITIONER = ROOT.parent / "LocalLLM/.local/models/image-generation/z-image-turbo-f332072a"
TASK = ("Read and follow the instruction written in the reference image. "
    "Produce only the requested answer as clearly legible black writing on a plain white page. "
    "Match the language of the instruction. "
    "Preserve the exact visual form when copying a glyph is requested.")
FILES = {
    "model_index.json": (446, "51a76cb1cf3ed37423a1128c79c22faee8e6fbe7f5aaeb737f0a258930dbaac0"),
    "scheduler/scheduler_config.json": (486, "067afb012cef64553a763447d1efd93daeffcc0123ca7e25b09f8de20b90762e"),
    "transformer/config.json": (541, "09733c74a3da6d17dd0a0472a091a8950c7c6935889c32c16cc800ede05029de"),
    "vae/config.json": (821, "0d6dfb69ae95a5e2ac9836284bbb63d8b38ce67b25ba2dff380752b2a10ab948"),
    "transformer/diffusion_pytorch_model.safetensors": (7751109744, "9f29f9edcfdae452a653ffb51a534ca4decd389952c225724ff3b94042612a6e"),
    "vae/diffusion_pytorch_model.safetensors": (168120878, "ca70d2202afe6415bdbcb8793ba8cd99fd159cfe6192381504d6c4d3036e0f04"),
}
CONDITIONER_WEIGHTS = {
    "model-00001-of-00003.safetensors": "328a91d3122359d5547f9d79521205bc0a46e1f79a792dfe650e99fc2d651223",
    "model-00002-of-00003.safetensors": "6cd087b316306a68c562436b5492edbcf6e16c6dba3a1308279caa5a58e21ca5",
    "model-00003-of-00003.safetensors": "7ca841ee75b9c61267c0c6148fd8d096d3d21b6d3e161256a9b878154f91fc52",
}
SOURCES = ("ilm/visual_lm/image_edit_baseline.py", "ilm/visual_lm/image_edit_baseline_data.py",
    "scripts/image_edit_language_baseline.py",
    "references/image_edit_baseline_protocol.md")


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def save_json(path, value):
    path = Path(path); temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def identity():
    for name in SOURCES:
        subprocess.run(["git", "ls-files", "--error-unmatch", name], cwd=ROOT,
            check=True, stdout=subprocess.DEVNULL)
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no",
        "--", *SOURCES], cwd=ROOT, text=True)
    if dirty:
        raise ValueError("commit the baseline implementation and protocol before execution")
    return dict(source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"],
        cwd=ROOT, text=True).strip(), sources={name: sha(ROOT / name) for name in SOURCES},
        repo=REPO, revision=REVISION,
        packages={name: version(name) for name in ("torch", "diffusers", "transformers", "huggingface-hub", "pillow", "numpy")})


def inspect_source():
    """Fetch actual pinned metadata/configuration bytes, never model weights."""
    url = f"https://huggingface.co/api/models/{REPO}/revision/{REVISION}?blobs=true"
    with urllib.request.urlopen(url, timeout=30) as response:
        metadata = json.load(response)
    if metadata["sha"] != REVISION or metadata.get("gated"):
        raise ValueError("pinned public source revision differs")
    siblings = {row["rfilename"]: row for row in metadata["siblings"]}
    observed = {}
    for name, (size, digest) in FILES.items():
        row = siblings[name]
        if row["size"] != size:
            raise ValueError("source file size differs: " + name)
        if name.endswith(".safetensors"):
            if row["lfs"]["sha256"] != digest:
                raise ValueError("source LFS digest differs: " + name)
        else:
            with urllib.request.urlopen(f"https://huggingface.co/{REPO}/resolve/{REVISION}/{name}", timeout=30) as response:
                content = response.read(65537)
            if len(content) != size or hashlib.sha256(content).hexdigest() != digest:
                raise ValueError("source configuration bytes differ: " + name)
            json.loads(content)
        observed[name] = dict(bytes=size, sha256=digest, weight_file=name.endswith(".safetensors"))
    return dict(files=observed, new_weight_payload_bytes=sum(size for name, (size, _) in FILES.items()
        if name.endswith(".safetensors")), weights_downloaded=False, model_loaded=False,
        generation_performed=False, language_quality_claim=False)


def predecessor_failure(run=PREDECESSOR):
    """A fallback cannot preempt the current comparison or a fitting actor."""
    run = Path(run)
    status = json.loads((run / "status.json").read_text())
    if status != {"stage": "completed_verified", "pid": None}:
        raise ValueError("finish both current training arms and CPU verification before any fallback model stage")
    verified = json.loads((run / "verification.json").read_text())
    declared = json.loads((run / "identity.json").read_text())
    if (verified["identity"] != declared
            or declared["purpose"] != "matched_state_only_decoder_metric"
            or not verified["all_64_final_outputs_independently_scored"]
            or not verified["all_64_initial_generations_bit_exact"]
            or not verified["matched_case_sampler_and_noise_rng"]):
        raise ValueError("completed original matched verification required")
    for name, digest in declared["sources"].items():
        if sha(ROOT / name) != digest:
            raise ValueError("predecessor source changed: " + name)
    for arm in ("control", "metric"):
        report = json.loads((run / arm / "report.json").read_text())
        if (report["updates"] != 2000 or report["draws"] != 8000
                or report["checkpoint_sha256"] != sha(run / arm / "checkpoint_latest.pt")
                or report["summary"] != verified["summaries"][arm]):
            raise ValueError("predecessor final state differs")
        score = verified["summaries"][arm]
        if score["free_exact"] >= 24 and score["free_f1"] >= .98:
            raise ValueError("an existing arm fits; evaluate its generalization before starting the fallback")
    return dict(run=str(run.relative_to(ROOT)), verification_sha256=sha(run / "verification.json"),
        both_arms_fail_absolute_fit=True)


def verify_snapshot(snapshot, *, codec_only=False):
    snapshot = Path(snapshot)
    chosen = {name: value for name, value in FILES.items()
        if not codec_only or name.startswith("vae/")}
    for name, (size, digest) in chosen.items():
        path = snapshot / name
        if path.stat().st_size != size or sha(path) != digest:
            raise ValueError("pinned component differs: " + name)
    return {name: digest for name, (_, digest) in chosen.items()}


def acquire(cache=CACHE):
    """Acquire just the explicit generator/VAE allowlist, no text encoder."""
    from huggingface_hub import constants, hf_hub_download
    # Prefer a complete verified shared installation if another project has
    # already acquired this exact revision since metadata inspection.
    existing = Path(constants.HF_HUB_CACHE) / "models--black-forest-labs--FLUX.2-klein-4B/snapshots" / REVISION
    if all((existing / name).is_file() for name in FILES):
        verify_snapshot(existing)
        return existing
    cache = Path(cache)
    ancestor = cache
    while not ancestor.exists():
        ancestor = ancestor.parent
    missing_bytes = sum(size for name, (size, _) in FILES.items()
        if not (cache / "models--black-forest-labs--FLUX.2-klein-4B/snapshots" / REVISION / name).is_file())
    if shutil.disk_usage(ancestor).free < 100 * 1024**3 + missing_bytes:
        raise RuntimeError("acquisition must leave 100 GiB free on its destination filesystem")
    snapshot = None
    for name in FILES:
        file = Path(hf_hub_download(REPO, name, revision=REVISION, cache_dir=cache))
        base = file.parent if "/" not in name else file.parent.parent
        if snapshot is not None and snapshot != base:
            raise ValueError("acquired components do not share a source revision")
        snapshot = base
    verify_snapshot(snapshot)
    return snapshot


@contextmanager
def gpu_zero():
    """One actual project model job; preserve other projects and GPU1."""
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise ValueError("explicit CUDA_VISIBLE_DEVICES=0 required for the admitted model stage")
    with (ROOT / "artifacts/ilm_gpu0.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        info = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, value = line.split(":", 1); info[key] = int(value.split()[0])
        if info["MemAvailable"] < 24 * 1024**2 or (
                info["SwapTotal"] and (info["SwapTotal"] - info["SwapFree"]) / info["SwapTotal"] > .75):
            raise RuntimeError("shared workstation RAM/swap threshold reached; inspect owned processes")
        free = subprocess.check_output(["nvidia-smi", "-i", "0", "--query-gpu=memory.free",
            "--format=csv,noheader,nounits"], text=True).strip()
        if int(free) < 20 * 1024:
            raise RuntimeError("GPU0 needs 20 GiB free before cold model loading")
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("the selected runtime cannot use the project GPU")
        torch.cuda.set_device(0); torch.cuda.reset_peak_memory_stats(0)
        torch.set_num_threads(4)
        yield torch.device("cuda:0")


def tensor_sha(value):
    import torch
    bits = value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(bits).hexdigest()


def condition_cache(path):
    import torch
    value = torch.load(path, map_location="cpu", weights_only=True)
    if set(value) != {"positive", "negative", "metadata"}:
        raise ValueError("fixed condition cache structure differs")
    metadata = value["metadata"]
    if (metadata["generic_instruction_sha256"] != hashlib.sha256(TASK.encode()).hexdigest()
            or metadata["hidden_state_layers"] != [9, 18, 27]
            or metadata["question_specific_information"]
            or metadata["source_weight_sha256"] != CONDITIONER_WEIGHTS):
        raise ValueError("the declared generic source condition is required")
    for key in ("positive", "negative"):
        item = value[key]
        if (item.shape != (1, 512, 7680) or item.dtype != torch.bfloat16
                or not bool(torch.isfinite(item).all()) or tensor_sha(item) != metadata[key + "_sha256"]):
            raise ValueError("fixed condition tensor differs: " + key)
    return value


def open_page(path):
    from PIL import Image, ImageOps
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGBA")
        background = Image.new("RGBA", image.size, "white")
        background.alpha_composite(image); image = background.convert("RGB")
    width, height = image.size
    if width % 16 or height % 16 or min(width, height) < 32 or width * height > 1048576:
        raise ValueError("page dimensions must be multiples of 16, at least 32, with area at most 1,048,576")
    return image


def encoder_free_pipeline(snapshot, device):
    import torch
    from diffusers import Flux2KleinPipeline
    pipeline = Flux2KleinPipeline.from_pretrained(str(snapshot), torch_dtype=torch.bfloat16,
        local_files_only=True, text_encoder=None, tokenizer=None).to(device)
    if pipeline.text_encoder is not None or pipeline.tokenizer is not None:
        raise ValueError("a language encoder or tokenizer was loaded into the generator runtime")
    for component in (pipeline.transformer, pipeline.vae):
        if any(isinstance(module, torch.nn.Embedding) for module in component.modules()):
            raise ValueError("unexpected discrete embedding table in the declared image runtime")
    pipeline.set_progress_bar_config(disable=True)
    return pipeline
