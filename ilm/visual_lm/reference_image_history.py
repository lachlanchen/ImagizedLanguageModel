"""Frozen diagnostic with supplied image history; never autonomous inference."""
import torch

from .semantic_glyph_loop import tensor_hash
from .semantic_image_memory import native_budget_image


@torch.inference_mode()
def trace_reference_history(actor, prompt, reference, guard=lambda: None):
    """Predict at every reference boundary, feeding only preceding gold pixels.

    Stopping is measured but bypassed. Includes the terminal boundary, for which
    no glyph target exists. The current target never enters its own prediction.
    Caller must label the result reference-conditioned and retain the raw trace.
    """
    if (reference.ndim != 4 or reference.shape[1:] != (1, 32, 32)
            or not len(reference) or not torch.all((reference == -1) | (reference == 1))):
        raise ValueError('nonempty signed binary reference glyph images required')
    guard()
    arguments, geometry = actor.interface.assemble(native_budget_image(prompt))
    output = actor.interface.actor.context(**arguments, use_cache=True)
    state, past = output.last_hidden_state[:, -1:], output.past_key_values
    del output, arguments
    offset = geometry['rotary_offset']
    states, pixels, stops, feedback = [], [], [], []
    for i in range(len(reference) + 1):
        guard()
        states.append(state[0, -1].float().cpu())
        logits, semantic, stop = actor.bridge.predict(state[:, -1].float())
        pixels.append((logits.sigmoid() >= .5).to(torch.uint8).cpu()[0])
        stops.append(float(stop.sigmoid()[0]))
        del logits, semantic, stop
        if i == len(reference):
            break
        observed = reference[i:i+1].to(actor.device)
        embedded = actor.bridge.feedback_embedding(observed)
        feedback.append(dict(image_sha256=tensor_hash(observed),
                             embedding_sha256=tensor_hash(embedded)))
        positions = torch.tensor([past.get_seq_length(), offset, offset, offset],
                                 device=actor.device).reshape(4, 1, 1)
        output = actor.interface.actor.context(inputs_embeds=embedded[:, None],
            position_ids=positions, past_key_values=past, use_cache=True)
        state, past = output.last_hidden_state[:, -1:], output.past_key_values
        del output, observed, embedded
        offset += 1
    return dict(states=torch.stack(states), predicted_ink=torch.stack(pixels),
        stop_probabilities=stops, reference_feedback=feedback, observer_geometry=geometry,
        reference_conditioned=True, supplied_target_length=True, stopping_bypassed=True,
        autonomous_language_generation=False, new_optimizer_updates=0)
