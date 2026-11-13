"""Initialize the controlled joint-likelihood continuation comparison."""
from pathlib import Path
import json

import torch

from .continuation_training import build_actor, restore_trainable
from .glyph_joint_density import GlyphJointDensity, JointGlyphBridge
from .joint_writer_training import ROOT, RUN as WRITER_RUN, PARENT, require, sha
from .visual_continuation import prefill_visual_prompt


TRAINABLE_PARAMETERS = 5005760


def load(path):
    return json.loads(Path(path).read_text())


def build_joint_context_actor(device):
    """Same point-final state and fresh head draw as the head-only control."""
    r = load(WRITER_RUN / 'registration.json')
    parent = load(PARENT / 'preparation.json')
    actor, provenance = build_actor(device, parent['seed'])
    record = r['parent_checkpoint']
    require(sha(ROOT / record['file']) == record['sha256'], 'Initial continuation checkpoint changed')
    value = torch.load(ROOT / record['file'], weights_only=True, map_location='cpu')
    restore_trainable(actor, value['trainable_weights'])
    del value
    norm = torch.load(WRITER_RUN / 'cache/normalization.pt', weights_only=True, map_location='cpu')
    density = GlyphJointDensity(norm['mean'], norm['scale']).to(device)
    require(sha(WRITER_RUN / 'initial_parameters.pt') == r['initial_parameters_sha256'],
            'Fresh registered image-head draw changed')
    initial = torch.load(WRITER_RUN / 'initial_parameters.pt', weights_only=True, map_location='cpu')
    state = density.state_dict()
    require(set(initial) == set(dict(density.named_parameters())), 'Initial image-head parameter map differs')
    state.update(initial); density.load_state_dict(state)
    require(all(torch.equal(p.detach().cpu(), initial[n]) for n, p in density.named_parameters()),
            'New head must start from the identical fresh control draw')
    actor.bridge = JointGlyphBridge(actor.bridge, density)
    actor.dynamics.train(); density.train()
    parameters = dict(actor.named_parameters())
    trainables = {n: p for n, p in parameters.items() if p.requires_grad}
    require(all(n == 'dynamics.query' or n.startswith(('dynamics.adapters.', 'bridge.density.'))
                for n in trainables), 'Unexpected trained module')
    require(sum(p.numel() for p in trainables.values()) == TRAINABLE_PARAMETERS,
            'Wrong complete trainable parameter count')
    require(all(p.dtype == torch.float32 for p in trainables.values()), 'FP32 trainables required')
    account = dict(total_unique_parameters=sum(p.numel() for p in actor.parameters()),
        trainable_parameters=sum(p.numel() for p in trainables.values()),
        continuation_parameters=sum(p.numel() for n, p in trainables.items() if n.startswith('dynamics.')),
        image_head_parameters=sum(p.numel() for n, p in trainables.items() if n.startswith('bridge.density.')),
        inherited_visual_contextual_parameters=provenance['retained_base_parameters'],
        source_point_step=41298, image_head_updates_at_initialization=0,
        inherited_pretraining_cost_not_measured_here=True)
    return actor, provenance, account


def differentiable_states(actor, prompt, ink, *, memory=None):
    """Recompute actual states; no immutable predictor cache substitutes for theta."""
    if memory is None:
        memory, _, geometry = prefill_visual_prompt(actor.interface, prompt)
    else:
        geometry = None
    with torch.no_grad():
        signed = (1 - 2 * ink.float()).to(actor.dynamics.query.device)
        feedback = actor.bridge.feedback_embedding(signed)[None]
    states = actor.dynamics(actor.dynamics.embeddings(feedback), memory, checkpoint_layers=True)[0].float()
    require(states.shape == (len(ink) + 1, 4096) and states.requires_grad,
            'Complete differentiable target-preceding states required')
    return states, memory, geometry
