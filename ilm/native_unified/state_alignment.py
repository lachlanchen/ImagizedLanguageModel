"""Preserve the native coarse reader and learn a fine-image state correction."""
import torch
from torch import nn
from torch.nn import functional as F

from .windowed import windowed_prefix

LAYERS = (4, 8, 12, 16, 20, 24, 28)


def add_bridge(bundle):
    if bundle.model.showo.config.hidden_size != 1536 or bundle.model.showo.config.num_hidden_layers != 28:
        raise ValueError("pinned native language-body geometry required")
    bridge = nn.Linear(1536, 1536, bias=False, device=bundle.device, dtype=torch.float32)
    nn.init.zeros_(bridge.weight)
    bundle.model.visual_context_bridge = bridge


def visual_start(bundle):
    encode = lambda words: bundle.tokenizer(words, add_special_tokens=False).input_ids
    return len([bundle.ids["bos_id"]] + encode("system\nYou are a helpful assistant.<|im_end|>") + encode("\n<|im_start|>user\n")) + 2


def corrected_prefix(bundle, fine_latent, coarse_prefix, coarse_mask):
    # Only the inference decorator is unwrapped. Actual visual operations are
    # the authenticated window computation, with normal autograd in training.
    fine, _, geometry = windowed_prefix.__wrapped__(bundle, fine_latent, "Follow the request written in the image. Output only the requested answer, with no explanation.")
    if geometry["grid"] != [64, 64]:
        raise ValueError("complete high-resolution source required")
    start = visual_start(bundle)
    features = fine[:, start:start + 4096].reshape(1, 64, 64, 1536).permute(0, 3, 1, 2)
    pooled = F.adaptive_avg_pool2d(features.float(), (27, 27)).flatten(2).transpose(1, 2)
    coarse = coarse_prefix[:, start:start + 729]
    correction = bundle.model.visual_context_bridge((pooled - coarse.float()).float()).to(coarse_prefix.dtype)
    value = torch.cat([coarse_prefix[:, :start], coarse + correction, coarse_prefix[:, start + 729:]], dim=1)
    if value.shape != coarse_prefix.shape or value.shape[1] != 768:
        raise ValueError("native contextual interface changed")
    return value, coarse_mask


def normalized(states):
    states = states.float()
    return states / torch.sqrt(states.square().mean(dim=-1, keepdim=True) + 1e-6)


def reading_states(outputs, position):
    return torch.stack([outputs.hidden_states[layer][0, position].float() for layer in LAYERS])


def teacher_prefix(bundle, text):
    words = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n" + text + "\nFollow the request. Output only the requested answer, with no explanation.\n<|im_start|>assistant\n"
    ids = torch.tensor([bundle.tokenizer(words, add_special_tokens=False).input_ids], device=bundle.device)
    prefix = bundle.model.showo.model.embed_tokens(ids)
    mask = bundle.native.masks.causal_attn_mask_naive(1, prefix.shape[1], bundle.device).to(bundle.dtype)
    return prefix, mask


def answer_input(bundle, prefix, mask, expected):
    answer = torch.tensor(bundle.tokenizer(expected, add_special_tokens=False).input_ids + [bundle.tokenizer.eos_token_id], device=bundle.device)
    history = bundle.model.showo.model.embed_tokens(answer[:-1])[None]
    value = torch.cat([prefix, history], dim=1)
    attention = bundle.native.masks.causal_attn_mask_naive(1, value.shape[1], bundle.device).to(bundle.dtype)
    n = prefix.shape[1]
    attention[:, :, :n, :n] = mask
    if history.shape[1] and not bool((attention[:, :, :n, n:] < -1e4).all()):
        raise ValueError("answer history is visible to pre-answer state")
    return value, attention, answer


def loss(bundle, sample, expected):
    prefix, mask = corrected_prefix(bundle, sample["fine"], sample["coarse_prefix"], sample["coarse_mask"])
    value, attention, answer = answer_input(bundle, prefix, mask, expected)
    output = bundle.model.showo(inputs_embeds=value, attention_mask=attention, use_cache=False,
        output_hidden_states=True, num_logits_to_keep=answer.numel())
    ce = F.cross_entropy(output.logits[0].float(), answer)
    kd = F.mse_loss(normalized(reading_states(output, prefix.shape[1] - 1)), sample["teacher_state"].float())
    return ce + kd, dict(answer_ce=float(ce.detach()), state_mse=float(kd.detach()),
        prefix_positions=prefix.shape[1], target_decisions=answer.numel())
