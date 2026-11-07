"""Load pinned Show-o2 source and weights without redundant base checkpoints.

The native neural implementation is Apache-2.0, NUS Show Lab (2025).
This adapter does not change its forward functions. The two timm tuple-helper
imports use torch's equivalent integer/tuple helper, avoiding a torchvision
installation for unused image augmentation. Constructor-only SigLIP hydration
uses its config on meta; every retained tensor is then supplied by the complete
Show-o2 checkpoint, with strict state loading.
"""
import hashlib
import importlib
import json
from pathlib import Path
import sys
import types
from unittest.mock import patch

import torch

SOURCE_COMMIT = "45a5a2de01d1ebd10cd5864d29310a76476cdf23"
MODEL_REVISION = "d3a220ec55feaacbdfcb053847edee14edd4e69a"
MODEL_SHA256 = "58870f9d31bf6be71113301c3402a506938e963b9bd6da7f78aadae6ce812852"
VAE_SHA256 = "38071ab59bd94681c686fa51d75a1968f64e470262043be31f7a094e442fd981"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def source_modules(root):
    """Authenticate pinned source before executing it in an isolated namespace."""
    root = Path(root).resolve()
    receipt = json.loads((root / "source_manifest.json").read_text())
    if receipt["repo"] != "showlab/Show-o" or receipt["commit"] != SOURCE_COMMIT:
        raise ValueError("unrecognized native source revision")
    for record in receipt["files"]:
        raw = (root / "repo" / record["path"]).read_bytes()
        git_blob = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
        if sha(root / "repo" / record["path"]) != record["sha256"] or git_blob != record["git_blob_sha1"]:
            raise ValueError("native source changed: " + record["path"])
    base = root / "repo/show-o2"
    package = "ilm_showo2_native"
    if package in sys.modules:
        if sys.modules[package].__path__ != [str(base)]:
            raise ValueError("another native source tree is already imported")
    else:
        module = types.ModuleType(package); module.__path__ = [str(base)]
        sys.modules[package] = module
        models = types.ModuleType(package + ".models"); models.__path__ = [str(base / "models")]
        sys.modules[models.__name__] = models
        for name in ("modules", "misc"):
            path = base / "models" / (name + ".py")
            words = path.read_text()
            original = "from timm.layers.helpers import to_2tuple"
            if words.count(original) != 1:
                raise ValueError("tuple-helper compatibility scope changed")
            module = types.ModuleType(package + ".models." + name)
            module.__file__ = str(path); module.__package__ = package + ".models"
            sys.modules[module.__name__] = module
            exec(compile(words.replace(original, "from torch.nn.modules.utils import _pair as to_2tuple"), str(path), "exec"), module.__dict__)
    model = importlib.import_module(package + ".models.modeling_showo2_qwen2_5")
    misc = importlib.import_module(package + ".models.misc")
    masks = importlib.import_module(package + ".models.omni_attention")
    vae = importlib.import_module(package + ".models.wan21_vae")
    transport = importlib.import_module(package + ".transport")
    return types.SimpleNamespace(model=model, misc=misc, masks=masks, vae=vae, transport=transport)


def assets(root, verify_weights=True):
    root = Path(root).resolve()
    report = json.loads((root / "acquisition.json").read_text())
    if not report["complete"]:
        raise ValueError("asset acquisition is incomplete")
    records = {(x["repo"], x["file"]): x for x in report["files"]}
    model = records["showlab/show-o2-1.5B-HQ", "pytorch_model.bin"]
    vae = records["Wan-AI/Wan2.1-T2V-14B", "Wan2.1_VAE.pth"]
    if model["revision"] != MODEL_REVISION or model["sha256"] != MODEL_SHA256 or vae["sha256"] != VAE_SHA256:
        raise ValueError("unrecognized model or codec")
    for record in records.values():
        path = Path(record["path"])
        if path.stat().st_size != record["bytes"] or (verify_weights or record["bytes"] < 10_000_000) and sha(path) != record["sha256"]:
            raise ValueError("asset bytes changed: " + record["file"])
    # Keep snapshot paths so adjacent tokenizer/config files remain available.
    def snapshot(repo, name):
        record = records[repo, name]
        return root / "hub" / ("models--" + repo.replace("/", "--")) / "snapshots" / record["revision"] / name
    return types.SimpleNamespace(
        model=snapshot("showlab/show-o2-1.5B-HQ", "pytorch_model.bin"),
        config=snapshot("showlab/show-o2-1.5B-HQ", "config.json"),
        vae=snapshot("Wan-AI/Wan2.1-T2V-14B", "Wan2.1_VAE.pth"),
        qwen=snapshot("Qwen/Qwen2.5-1.5B-Instruct", "config.json").parent,
        siglip=snapshot("google/siglip-so400m-patch14-384", "config.json").parent,
        receipt=report)


def load(root, dtype=torch.bfloat16, device="cuda:0"):
    """Strictly load all native weights; no network/base-weight downloads."""
    from accelerate import init_empty_weights
    from transformers import SiglipConfig
    native = source_modules(root)
    files = assets(root)
    config = json.loads(files.config.read_text())
    config = {k: v for k, v in config.items() if not k.startswith("_")}
    config.update(llm_model_path=str(files.qwen), clip_pretrained_model_path=str(files.siglip), load_from_showo=True)
    # The original constructor loads an entire SigLIP before overwriting its
    # retained layers from Show-o2. Construct that placeholder from config only.
    def siglip_from_config(*args, **kwargs):
        cfg = SiglipConfig.from_pretrained(files.siglip, local_files_only=True)
        return native.model.SiglipModel._from_config(cfg, attn_implementation="sdpa")
    with init_empty_weights(include_buffers=False), patch.object(native.model.SiglipModel, "from_pretrained", side_effect=siglip_from_config):
        model = native.model.Showo2Qwen2_5(**config)
    state = torch.load(files.model, map_location="cpu", mmap=True, weights_only=True)
    loaded = model.load_state_dict(state, strict=True, assign=True)
    if loaded.missing_keys or loaded.unexpected_keys:
        raise ValueError("native checkpoint does not cover the constructed model")
    if model.showo.config.tie_word_embeddings:
        # init_empty_weights can temporarily split Python Parameter objects.
        # Restore the source model's tying only after checking actual values.
        if not torch.equal(model.showo.model.embed_tokens.weight, model.showo.lm_head.weight):
            raise ValueError("source checkpoint disagrees with its declared embedding tying")
        model.showo.tie_weights()
        if model.showo.model.embed_tokens.weight is not model.showo.lm_head.weight:
            raise ValueError("native embedding tying was not restored")
    del state
    model.eval().requires_grad_(False).to(device=device, dtype=dtype)
    vae = native.vae.WanVAE(vae_pth=str(files.vae), dtype=dtype, device=device)
    tokenizer, ids = native.misc.get_text_tokenizer(str(files.qwen), return_showo_token_ids=True)
    if len(tokenizer) != model.config.llm_vocab_size:
        raise ValueError("native special-token rows differ")
    counts = dict(native_model=sum(p.numel() for p in model.parameters()),
        language_body=sum(p.numel() for p in model.showo.parameters()),
        vae=sum(p.numel() for p in vae.model.parameters()),
        parameters_requiring_grad=sum(p.numel() for p in model.parameters() if p.requires_grad))
    return types.SimpleNamespace(model=model, vae=vae, tokenizer=tokenizer, ids=ids, native=native,
        files=files, counts=counts, dtype=dtype, device=torch.device(device))


@torch.inference_mode()
def image_latent(bundle, path, seed):
    """Preserve the complete supplied RGB geometry; use native VAE sampling."""
    import numpy as np
    from PIL import Image
    with Image.open(path) as image:
        image = image.convert("RGB")
        if image.width % 16 or image.height % 16:
            raise ValueError("input dimensions must be native multiples of 16")
        array = np.asarray(image).copy()
    pixels = torch.from_numpy(array).permute(2, 0, 1)[None, :, None].float().to(bundle.device) / 127.5 - 1
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    return bundle.vae.sample(pixels).squeeze(2), pixels


@torch.inference_mode()
def reading_prefix(bundle, latents, instruction):
    """Native visual fusion and source chat layout; diagnostic text helper only."""
    model, native, tok, ids = bundle.model, bundle.native, bundle.tokenizer, bundle.ids
    z = latents.to(bundle.dtype)
    h, w = z.shape[-2:]; p = model.config.patch_size
    und = model.image_embedder_und(z)
    gen = model.image_embedder_gen(z)
    if h // p == w // p == 27:
        pos = model.position_embedding(torch.arange(729, device=z.device)[None])
    else:
        pos = native.misc.interpolate_pos_encoding(model.config.clip_latent_dim, model.position_embedding, h // p, w // p, 1)
    und = model.und_trans(und + pos)["last_hidden_state"]
    visual = model.fusion_proj(torch.cat([und, gen], dim=-1))
    encode = lambda words: tok(words, add_special_tokens=False).input_ids
    a = [ids["bos_id"]] + encode("system\nYou are a helpful assistant.<|im_end|>") + encode("\n<|im_start|>user\n")
    b = [ids["boi_id"], ids["eoi_id"]] + encode(instruction) + encode("\n<|im_start|>assistant\n")
    ta, tb = [torch.tensor([x], device=z.device) for x in (a, b)]
    ea, eb = [model.showo.model.embed_tokens(x) for x in (ta, tb)]
    t = model.time_embed(torch.ones((1, 1), device=z.device), ea.dtype)
    if hasattr(model, "time_embed_proj"):
        t = model.time_embed_proj(t)
    embeddings = torch.cat([ea, eb[:, :1], t, visual, eb[:, 1:]], dim=1).to(bundle.dtype)
    # This is the exact offset used by the official inference_mmu script.
    positions = torch.tensor([[[len(a) + 2, visual.shape[1] + 1]]], device=z.device)
    mask = native.masks.omni_attn_mask_naive(1, embeddings.shape[1], positions, z.device, inverted=True).to(bundle.dtype)
    return embeddings, mask
