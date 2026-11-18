"""Decode a native glyph mixture using physical context as a contrast."""
import math

import torch

from ilm.visual_lm.continuous_autoregressive_language import mixture_log_prob


def candidate_densities(candidates, parameters):
    """Score each proposed vector under the entire mixture, in the same coordinates."""
    logits, means, scales = parameters
    count = candidates.shape[1]
    return mixture_log_prob(candidates, logits[:, None].expand(-1, count, -1),
                            means[:, None].expand(-1, count, -1, -1),
                            scales[:, None].expand(-1, count, -1, -1))


def select_group(full, suffix):
    logits, means, _ = full
    eligible = logits >= logits.amax(-1, keepdim=True) - math.log(10)
    density = candidate_densities(means, full)
    full_choice = density.masked_fill(~eligible, -torch.inf).argmax(-1)
    if suffix is None:
        return full_choice, full_choice, eligible.sum(-1)
    gains = density - candidate_densities(means, suffix)
    best = gains.masked_fill(~eligible, -torch.inf).amax(-1, keepdim=True)
    tied = eligible & (gains == best)
    choice = density.masked_fill(~tied, -torch.inf).argmax(-1)
    return choice, full_choice, eligible.sum(-1)


@torch.inference_mode()
def draw_glyph(model, full_state, suffix_state=None):
    """Return generated ink, component decisions and the emitted latent vector."""
    writer = model.writer
    both = suffix_state is not None
    states = torch.cat((full_state, suffix_state)) if both else full_state
    features = writer.condition(states)
    batch = len(full_state)
    previous = caches = None
    codes, choices, controls, counts = [], [], [], []
    for position in range(writer.groups):
        inputs = torch.cat((previous, previous)) if both and previous is not None else previous
        parameters, caches = writer.step(features, inputs, position, caches)
        full = tuple(p[:batch] for p in parameters)
        suffix = tuple(p[batch:] for p in parameters) if both else None
        selected, control, eligible = select_group(full, suffix)
        previous = full[1][torch.arange(batch, device=features.device), selected]
        codes.append(previous)
        choices.append(selected)
        controls.append(control)
        counts.append(eligible)
    code = torch.cat(codes, -1)
    return (model.codec.decode(code) < 0).to(torch.uint8), dict(
        selected=torch.stack(choices, 1), density_choice=torch.stack(controls, 1),
        eligible=torch.stack(counts, 1), code=code)


@torch.inference_mode()
def generate(model, prompt, *, contrast, maximum_new=128):
    """Use the original full-history stop decision and re-read each written image."""
    if model.training or maximum_new < 1 or not len(prompt):
        raise ValueError('Expected an evaluation model, nonempty prompt and positive output cap')
    history = model.encode_ink(prompt[-model.maximum_cells:])
    outputs, traces = [], []
    for _ in range(maximum_new):
        state = model.history_codes(history[None])[:, -1]
        if (model.stop(state) >= 0).item():
            break
        short = model.history_codes(history[-1:][None])[:, -1] if contrast else None
        ink, trace = draw_glyph(model, state, short)
        history = torch.cat((history, model.encode_ink(ink)))[-model.maximum_cells:]
        outputs.append(ink)
        traces.append(trace)
    return torch.cat(outputs) if outputs else prompt[:0], traces, len(outputs) < maximum_new
