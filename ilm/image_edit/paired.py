"""Native paired-image flow learning with an externally pretrained visual body.

Only the fixed, generic conditioning tensor is accepted. Question-specific
information enters through the image branch. This is conventional flow
matching and low-rank adaptation, not a new language-learning theorem.
"""
from pathlib import Path
import json
import math
import subprocess

import torch
from torch import nn
from torch.nn import functional as F

from .baseline import ROOT, condition_cache, open_page, save_json, sha, tensor_sha

REPO = "black-forest-labs/FLUX.2-klein-base-4B"
REVISION = "a3b4f4849157f664bdbc776fd7453c2783562f4d"
FILES = {
    "model_index.json": "de1fef01845f0af3e1e2f2581cd5e86f2dfa4a4776f7f0b26b4c77feac4ee70f",
    "transformer/config.json": "c8a88c1433f24e52c01f46766c9f519e76d91d1c00bfb43fe0db2e4ed3c84d70",
    "transformer/diffusion_pytorch_model.safetensors": "e109674697ffa1a3983126e32512f5428a9442bd8df59f9c95566ee90a473bb6",
    "scheduler/scheduler_config.json": "067afb012cef64553a763447d1efd93daeffcc0123ca7e25b09f8de20b90762e",
    "vae/config.json": "0d6dfb69ae95a5e2ac9836284bbb63d8b38ce67b25ba2dff380752b2a10ab948",
    "vae/diffusion_pytorch_model.safetensors": "ca70d2202afe6415bdbcb8793ba8cd99fd159cfe6192381504d6c4d3036e0f04",
}


def verify_source(snapshot):
    snapshot = Path(snapshot)
    for name, digest in FILES.items():
        if sha(snapshot / name) != digest:
            raise ValueError("base source bytes differ: " + name)
    if json.loads((snapshot / "model_index.json").read_text()).get("is_distilled", False):
        raise ValueError("paired learning requires the undistilled base")
    return dict(repo=REPO, revision=REVISION, files=FILES)


def implementation():
    paths = ("ilm/image_edit/paired.py", "ilm/image_edit/paired_data.py",
             "scripts/train_paired_image_language.py", "references/paired_image_learning_protocol.md")
    return dict(commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
        text=True).strip(), files={name: sha(ROOT / name) for name in paths})


class LowRankLinear(nn.Module):
    """W x + (alpha/r) B A x; the complete native W remains frozen."""

    def __init__(self, base, rank):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.rank = rank
        self.down = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device, dtype=torch.float32))
        self.up = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.down, a=math.sqrt(5))

    def forward(self, value):
        # Autocast supplies BF16 multiplication while FP32 adapter masters and
        # Adam state retain update precision. alpha == rank in this recipe.
        delta = F.linear(F.linear(value, self.down), self.up)
        return self.base(value) + delta.to(value.dtype)


def add_adapters(model, rank):
    if not 1 <= rank <= 128:
        raise ValueError("adapter rank must be in [1,128]")
    model.requires_grad_(False)
    names = [name for name, module in model.named_modules() if isinstance(module, nn.Linear)
        and (name.startswith(("transformer_blocks.", "single_transformer_blocks."))
             or name in ("x_embedder", "context_embedder", "proj_out"))]
    if not names:
        raise ValueError("the actual native visual projection layers were not found")
    for name in names:
        parent_name, _, child = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child, LowRankLinear(getattr(parent, child), rank))
    return names


def adapter_state(model):
    return {name: value.detach().cpu().clone() for name, value in model.named_parameters()
            if value.requires_grad}


def restore_adapters(model, values):
    selected = {name: value for name, value in model.named_parameters() if value.requires_grad}
    if selected.keys() != values.keys():
        raise ValueError("adapter checkpoint architecture differs")
    with torch.no_grad():
        for name, value in selected.items():
            if value.shape != values[name].shape or not bool(torch.isfinite(values[name]).all()):
                raise ValueError("invalid saved adapter: " + name)
            value.copy_(values[name].to(value))


def load_pipeline(snapshot, device, *, codec_only=False):
    from diffusers import Flux2KleinPipeline
    components = dict(transformer=None) if codec_only else {}
    pipe = Flux2KleinPipeline.from_pretrained(str(snapshot), local_files_only=True,
        torch_dtype=torch.bfloat16, text_encoder=None, tokenizer=None, **components).to(device)
    if pipe.config.is_distilled or pipe.text_encoder is not None or pipe.tokenizer is not None:
        raise ValueError("an encoder-free undistilled source pipeline is required")
    if pipe.transformer is not None and any(isinstance(m, nn.Embedding) for m in pipe.transformer.modules()):
        raise ValueError("unexpected vocabulary table in the visual generator")
    pipe.vae.requires_grad_(False).eval()
    pipe.set_progress_bar_config(disable=True)
    return pipe


@torch.no_grad()
def encode_page(pipe, path, device):
    image = open_page(path)
    pixels = pipe.image_processor.preprocess(image, height=image.height, width=image.width)
    pixels = pixels.to(device=device, dtype=pipe.vae.dtype)
    # Call the exact source mode/patchification/BN convention, not a separately
    # invented latent normalization. No resize/crop is permitted here.
    features = pipe._encode_vae_image(pixels, generator=None)
    if features.shape[-2:] != (image.height // 16, image.width // 16):
        raise ValueError("source codec changed page geometry")
    if not bool(torch.isfinite(features).all()):
        raise FloatingPointError("nonfinite source image features")
    return features.detach().cpu()


def read_manifest(path):
    path = Path(path).resolve()
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("nonempty manifest with unique IDs required")
    for row in rows:
        if row["split"] not in ("train", "validation") or row["task"] not in ("copy", "qa"):
            raise ValueError("explicit train/validation and copy/qa labels required")
        for key in ("input", "target"):
            file = (path.parent / row[key]).resolve()
            if sha(file) != row[key + "_sha256"]:
                raise ValueError("manifest image bytes changed: " + row["id"])
            row[key + "_path"] = file
    training = {r["input_sha256"] for r in rows if r["split"] == "train"}
    validation = {r["input_sha256"] for r in rows if r["split"] == "validation"}
    if training & validation:
        raise ValueError("training and validation share an input image")
    return rows


def cache_pairs(pipe, rows, out, device):
    import numpy as np
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    records = []
    for index, row in enumerate(rows):
        pair = {key: encode_page(pipe, row[key + "_path"], device) for key in ("input", "target")}
        name = f"{index:05}.pt"
        torch.save(pair, out / name)
        # Observe the output codec at the actual canvas before optimizing it.
        with torch.inference_mode():
            value = pair["target"].to(device)
            mean = pipe.vae.bn.running_mean.view(1, -1, 1, 1).to(value)
            std = torch.sqrt(pipe.vae.bn.running_var.view(1, -1, 1, 1)
                + pipe.vae.config.batch_norm_eps).to(value)
            native = pipe._unpatchify_latents(value * std + mean)
            decoded = pipe.vae.decode(native, return_dict=False)[0]
            png = pipe.image_processor.postprocess(decoded, output_type="pil")[0]
        decoded_name = f"{index:05}_target_codec.png"; png.save(out / decoded_name)
        original = np.asarray(open_page(row["target_path"]).convert("L")) < 128
        observed = np.asarray(png.convert("L")) < 128
        ink = int(original.sum() + observed.sum())
        f1 = 2 * int(np.logical_and(original, observed).sum()) / ink if ink else 1.
        records.append(dict(id=row["id"], split=row["split"], task=row["task"], file=name,
            sha256=sha(out / name), input_shape=list(pair["input"].shape),
            target_shape=list(pair["target"].shape),
            input_sha256=row["input_sha256"], target_sha256=row["target_sha256"],
            target_codec_png=decoded_name, target_codec_sha256=sha(out / decoded_name),
            target_codec_foreground_f1=f1))
        if index % 16 == 0:
            print(f"Encoded {index + 1}/{len(rows)} complete image pairs", flush=True)
    save_json(out / "manifest.json", dict(source_revision=REVISION, records=records,
        posterior_mode=True, native_patchification_and_batch_normalization=True))
    return records


def load_pair(cache, record, device):
    path = Path(cache) / record["file"]
    if sha(path) != record["sha256"]:
        raise ValueError("cached pair changed")
    pair = torch.load(path, map_location="cpu", weights_only=True)
    if set(pair) != {"input", "target"}:
        raise ValueError("numeric paired-image cache required")
    for key in pair:
        if list(pair[key].shape) != record[key + "_shape"] or not bool(torch.isfinite(pair[key]).all()):
            raise ValueError("cached feature shape/value differs")
    return {key: tensor.to(device=device) for key, tensor in pair.items()}


def flow_loss(pipe, pair, fixed, generator, *, capture=False):
    x, y = pair["input"], pair["target"]
    noise = torch.randn(y.shape, device=y.device, dtype=torch.float32, generator=generator)
    sigma = torch.rand((len(y),), device=y.device, generator=generator)
    mixed = (1 - sigma[:, None, None, None]) * y.float() + sigma[:, None, None, None] * noise
    packed_y = pipe._pack_latents(mixed.to(y.dtype))
    packed_x = pipe._pack_latents(x)
    ids = torch.cat((pipe._prepare_latent_ids(y), pipe._prepare_image_ids([x])), dim=1).to(y.device)
    c = fixed.to(y.device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        prediction = pipe.transformer(hidden_states=torch.cat((packed_y, packed_x), dim=1),
            encoder_hidden_states=c, timestep=sigma,
            img_ids=ids, txt_ids=pipe._prepare_text_ids(c).to(y.device), guidance=None,
            return_dict=False)[0][:, :packed_y.shape[1]]
    velocity = pipe._pack_latents(noise - y.float())
    loss = (prediction.float() - velocity).square().mean()
    observation = dict(sigma=float(sigma[0]), target_patches=packed_y.shape[1],
        input_patches=packed_x.shape[1])
    if capture:
        observation["prediction_sha256"] = tensor_sha(prediction)
    return loss, observation


def save_checkpoint(path, pipe, optimizer, *, step, generator, sampler, metadata):
    temporary = Path(path).with_suffix(".tmp")
    torch.save(dict(adapters=adapter_state(pipe.transformer), optimizer=optimizer.state_dict(),
        step=step, generator_state=generator.get_state(), sampler_state=sampler.get_state(),
        torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(), metadata=metadata), temporary)
    temporary.replace(path)


@torch.inference_mode()
def answer(pipe, input_path, fixed, *, seed, width, height, steps=50, guidance=1.):
    # Only an input image and one fixed condition enter this deployed path.
    page = open_page(input_path)
    if type(steps) is not int or steps < 1 or not math.isfinite(guidance) or guidance < 1:
        raise ValueError("positive native sampling steps and finite guidance at least one required")
    if min(width, height) < 32 or width % 16 or height % 16 or width * height > 1048576:
        raise ValueError("declare a valid output canvas independently of the answer")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        raw = pipe(image=page, prompt_embeds=fixed["positive"].to(pipe.device),
            negative_prompt_embeds=fixed["negative"].to(pipe.device),
            generator=torch.Generator(device=pipe.device).manual_seed(seed),
            width=width, height=height, num_inference_steps=steps,
            guidance_scale=guidance, output_type="pt").images
    if not bool(torch.isfinite(raw).all()):
        raise FloatingPointError("nonfinite autonomous answer image")
    return raw.cpu()


def save_answer(pipe, raw, out):
    # The source pipeline has already postprocessed this PT image into [0,1].
    # Save those exact pixels; no OCR, character lookup or font replacement.
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    torch.save(raw, out / "pixels.pt")
    image = pipe.image_processor.pt_to_pil(raw)[0]
    image.save(out / "answer.png")
    return dict(png_sha256=sha(out / "answer.png"), pixels_sha256=sha(out / "pixels.pt"),
        shape=list(raw.shape), output_repair=False)
