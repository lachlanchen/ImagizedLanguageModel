"""Frozen, training-only OCR supervision through actual output pixels.

This reader is never imported by the autonomous paired-image inference runner.
Its vocabulary and teacher-forced labels belong solely to supervision.
"""
from pathlib import Path

import torch
from torch.nn import functional as F

from .baseline import sha

REPO = "stepfun-ai/GOT-OCR-2.0-hf"
REVISION = "d3017ef2c2c1395888c8d635c5e0508bcb0ac78d"
FILES = {
    "model.safetensors": "6175ac7868a4e75735f5d59f78c465081ad3427eb4f312d072a0f1d16b333ba4",
    "tokenizer.json": "36b382a3c48c9a143c30139dac6c8230ddfb0b46a3dc43082af6052abe99d9de",
    "config.json": "cbe8aacd6cd84a2d58eafcd0045c6ac40e02e3a448f24b8cee51cc81d8bdccf2",
    "generation_config.json": "31915c5a692f43c5765a20cfc5f9403bcd250f5721a0d931bb703169c08993b4",
    "preprocessor_config.json": "ef9a0dc0935cac11f4230ca30d00a52bedfa52b6633e409e9fbd2ea56373aa7e",
    "special_tokens_map.json": "7c2368a3889fdfb37c24cabeb031b53f47934f357b54e56e8e389909a338ea47",
    "tokenizer_config.json": "8b0542937d32a67da8ea2d1288b870e325be383a962c65d201864299560a2b8e",
}


def verify_teacher(snapshot):
    for name, digest in FILES.items():
        if sha(Path(snapshot) / name) != digest:
            raise ValueError("training reader source differs: " + name)
    return dict(repo=REPO, revision=REVISION, files=FILES)


def acquire_teacher(cache):
    from huggingface_hub import hf_hub_download
    snapshot = None
    for name in FILES:
        file = Path(hf_hub_download(REPO, name, revision=REVISION, cache_dir=cache))
        if snapshot is not None and snapshot != file.parent:
            raise ValueError("reader files belong to different snapshots")
        snapshot = file.parent
    verify_teacher(snapshot)
    return snapshot


class RecognitionTeacher:
    def __init__(self, snapshot, device):
        from PIL import Image
        from transformers import GotOcr2ForConditionalGeneration, GotOcr2Processor
        self.source = verify_teacher(snapshot)
        self.processor = GotOcr2Processor.from_pretrained(snapshot, local_files_only=True,
            use_fast=True)
        self.model = GotOcr2ForConditionalGeneration.from_pretrained(snapshot,
            local_files_only=True, torch_dtype=torch.bfloat16).to(device).requires_grad_(False).eval()
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if any(isinstance(m, torch.nn.Dropout) and m.p != 0 for m in self.model.modules()):
            raise ValueError("gradient-checkpointed frozen reader must have no stochastic dropout")
        self.device = device
        self.prompt = self.processor(Image.new("RGB", (512, 256), "white"),
            return_tensors="pt")["input_ids"].to(device)
        self.eos = self.processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
        config = self.processor.image_processor
        self.mean = torch.tensor(config.image_mean, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(config.image_std, device=device).view(1, 3, 1, 1)
        self.size = (config.size["height"], config.size["width"])

    def pixels(self, rgb):
        if rgb.ndim != 4 or rgb.shape[:2] != (1, 3):
            raise ValueError("one RGB page required")
        value = F.interpolate(rgb.float().clamp(0, 1), size=self.size, mode="bicubic",
            align_corners=False, antialias=True).clamp(0, 1)
        return ((value - self.mean) / self.std).to(torch.bfloat16)

    def labels(self, text):
        tokens = self.processor.tokenizer(text, add_special_tokens=False)["input_ids"]
        if not tokens or len(tokens) >= 64:
            raise ValueError("nonempty short writing target required")
        return torch.tensor([tokens + [self.eos]], device=self.device, dtype=torch.long)

    def loss(self, rgb, text):
        targets = self.labels(text)
        inputs = torch.cat((self.prompt, targets[:, :-1]), dim=1)
        # The first scored position is the last prompt position. Its next-token
        # target is the first answer token. Prompt and image-pad logits are not scored.
        out = self.model(input_ids=inputs, attention_mask=torch.ones_like(inputs),
            pixel_values=self.pixels(rgb), use_cache=False, logits_to_keep=targets.shape[1])
        return F.cross_entropy(out.logits.float().reshape(-1, out.logits.shape[-1]), targets.flatten())

    @torch.inference_mode()
    def read(self, rgb):
        self.model.eval()
        generated = self.model.generate(input_ids=self.prompt,
            attention_mask=torch.ones_like(self.prompt), pixel_values=self.pixels(rgb),
            do_sample=False, max_new_tokens=64, eos_token_id=self.eos, use_cache=True)
        return self.processor.tokenizer.decode(generated[0, self.prompt.shape[1]:],
            skip_special_tokens=True)


def decode_features(pipe, features):
    """Native source inverse with a real differentiable frozen VAE decoder."""
    mean = pipe.vae.bn.running_mean.view(1, -1, 1, 1).to(features)
    std = (pipe.vae.bn.running_var.view(1, -1, 1, 1)
        + pipe.vae.config.batch_norm_eps).sqrt().to(features)
    native = pipe._unpatchify_latents(features * std + mean)
    decoded = pipe.vae.decode(native.to(pipe.vae.dtype), return_dict=False)[0]
    return (decoded.float() * .5 + .5).clamp(0, 1)
