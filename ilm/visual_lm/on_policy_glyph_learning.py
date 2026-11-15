"""Training-only whole-answer policy gradients for the physical glyph actor.

No optimizer or job is launched here. Full-actor qualification and a separately
registered experiment are required before use. CPU contract checks are not
evidence of learned language. The reward is a fixed-font image surrogate, not
an OCR system or a semantic judge.
"""
from dataclasses import dataclass
import math

import torch
from torch.nn import functional as F


def _binary_images(value):
    if value.ndim != 4 or value.shape[1:] != (1, 32, 32):
        raise ValueError('Expected complete [N,1,32,32] physical images')
    if not bool(((value == 0) | (value == 1)).all()):
        raise ValueError('Expected binary ink, including actual whitespace')


@dataclass(frozen=True)
class SampledAnswer:
    ink: torch.Tensor
    stopped: bool
    maximum_glyphs: int
    sampled_log_probability: float

    def __post_init__(self):
        _binary_images(self.ink)
        if type(self.maximum_glyphs) is not int or self.maximum_glyphs < 1:
            raise ValueError('A positive reference-independent safety cap is required')
        if type(self.stopped) is not bool or len(self.ink) > self.maximum_glyphs:
            raise ValueError('Invalid stopped/capped answer')
        if self.stopped and len(self.ink) == self.maximum_glyphs:
            raise ValueError('The sampling cap is a censoring event, not a stop decision')
        if not self.stopped and len(self.ink) != self.maximum_glyphs:
            raise ValueError('An unstopped sample must have reached its cap')
        if not math.isfinite(self.sampled_log_probability) or self.sampled_log_probability > 1e-5:
            raise ValueError('Invalid sampled probability')


@torch.no_grad()
def sample_physical_patterns(density, states, *, generator):
    """Ancestral sampling from the exact, untempered normalized pixel law.

    Greedy, top-k and temperature sampling would define a different policy.
    The caller must replay the same distribution before changing any weights.
    """
    if density.training:
        raise ValueError('Use eval mode for reproducible sampling/replay; gradients remain enabled in replay')
    condition = density.condition_features(states)
    caches = previous = None
    pieces, log_probabilities = [], []
    for position in range(density.config.positions):
        logits, caches = density.step(condition, previous, position, caches)
        log_probs = logits.float().log_softmax(-1)
        if not bool(torch.isfinite(log_probs).all()):
            raise ValueError('Nonfinite physical policy')
        previous = torch.multinomial(log_probs.exp(), 1, generator=generator).squeeze(-1)
        pieces.append(previous)
        log_probabilities.append(log_probs.gather(-1, previous[:, None]).squeeze(-1))
    return torch.stack(pieces, 1), torch.stack(log_probabilities, 1).sum(1)


def physical_log_probability(density, states, patterns):
    """SUM all physical decisions, never average by image or answer length."""
    if density.training or patterns.shape != (len(states), density.config.positions):
        raise ValueError('Complete physical actions and an eval-mode density required')
    if not bool(((patterns >= 0) & (patterns < 256) & (patterns == patterns.long())).all()):
        raise ValueError('Invalid eight-bit physical action')
    return density(states, patterns).float().log_softmax(-1).gather(
        -1, patterns.long()[..., None]).squeeze(-1).sum(1)


def stopping_log_probability(logits, emitted_glyphs, *, stopped):
    """Account for each sampled continue and, if observed, the terminal stop.

    A cap censors the process; it must not invent a successful EOS action.
    """
    decisions = emitted_glyphs + int(stopped)
    if type(emitted_glyphs) is not int or emitted_glyphs < 0 or type(stopped) is not bool:
        raise ValueError('Invalid stopping event')
    if logits.ndim != 1 or len(logits) < decisions or not bool(torch.isfinite(logits).all()):
        raise ValueError('Missing finite stopping logits')
    value = -F.softplus(logits[:emitted_glyphs]).sum()
    return value - F.softplus(-logits[emitted_glyphs]) if stopped else value


@torch.no_grad()
def sample_answer(actor, prompt, *, generator, maximum_glyphs=32, memory=None, guard=lambda: None):
    """Generate from a PNG tensor and sampled pixels; accepts no reference.

    Cap is independent of any answer length. A cap event has zero reward in
    the proposed training surrogate. The existing deployed greedy actor is
    unchanged; this is a training policy with explicit sampled stop actions.
    """
    from .glyph_joint_density import unpack_glyphs
    from .visual_continuation import prefill_visual_prompt

    if actor.training or type(maximum_glyphs) is not int or maximum_glyphs < 1:
        raise ValueError('An eval-mode actor and positive fixed cap are required')
    if memory is None:
        memory, _, _ = prefill_visual_prompt(actor.interface, prompt)
    current, history = actor.dynamics.query[None, None], None
    images, score, stopped = [], 0., False
    for _ in range(maximum_glyphs):
        guard()
        h, history = actor.dynamics(current, memory, checkpoint_layers=False,
                                    previous=history, return_history=True)
        state = h[:, -1].float()
        logit = actor.bridge.stop(state).reshape(())
        if not bool(torch.isfinite(logit)):
            raise ValueError('Nonfinite stopping policy')
        stop = bool(torch.rand((), device=logit.device, generator=generator) < logit.sigmoid())
        score += float(-F.softplus(-logit) if stop else -F.softplus(logit))
        if stop:
            stopped = True
            break
        patterns, log_probability = sample_physical_patterns(actor.bridge.density, state, generator=generator)
        ink = unpack_glyphs(patterns)
        images.append(ink.cpu())
        score += float(log_probability[0])
        # The emitted sample itself supplies feedback, with no reference history.
        current = actor.bridge.feedback_embedding(1 - 2 * ink.float())[:, None]
    ink = torch.cat(images) if images else torch.empty(0, 1, 32, 32, dtype=torch.uint8)
    guard()
    return SampledAnswer(ink, stopped, maximum_glyphs, score)


def replay_answer_log_probability(actor, sample, memory, *, guard=lambda: None):
    """Differentiate the same actions on their own preceding pixel history.

    Sampling is discrete. Score-function gradients do not require derivatives
    through chosen image bits. Frozen prompt prefill is intentional; the
    current continuation/query and density parameters receive gradients.
    """
    from .glyph_joint_density import pack_glyphs

    if actor.training:
        raise ValueError('Sampling and replay must both use eval-mode layers')
    guard()
    device = actor.dynamics.query.device
    ink = sample.ink.to(device)
    with torch.no_grad():
        feedback = (actor.bridge.feedback_embedding(1 - 2 * ink.float())[None] if len(ink)
                    else torch.empty(1, 0, actor.dynamics.query.numel(), device=device,
                                     dtype=actor.dynamics.query.dtype))
    states = actor.dynamics(actor.dynamics.embeddings(feedback), memory,
                            checkpoint_layers=True)[0].float()
    stop = actor.bridge.stop(states).flatten()
    value = stopping_log_probability(stop, len(ink), stopped=sample.stopped)
    if len(ink):
        value = value + physical_log_probability(actor.bridge.density, states[:-1], pack_glyphs(ink)).sum()
    guard()
    return value


@torch.no_grad()
def image_answer_reward(ink, reference, *, stopped):
    """Ordered fixed-font image edit reward in [0,1], with a whole-answer bonus.

    This is training supervision, not an autonomous language assessment. It
    rejects capped/all-white answers, preserves real spaces and penalizes
    insertions, deletions and duplication. Dice substitution supplies a dense
    surrogate when exact answers are rare; it can still reward a wrong but
    visually similar glyph. It must be calibrated on real outputs before a fit.
    """
    _binary_images(ink); _binary_images(reference)
    if type(stopped) is not bool or not len(reference) or not bool(reference.any()):
        raise ValueError('A visible complete reference and explicit stop event are required')
    if not stopped or not len(ink) or not bool(ink.any()):
        return 0.
    a = ink.detach().cpu().reshape(len(ink), -1).float()
    b = reference.detach().cpu().reshape(len(reference), -1).float()
    mass = a.sum(1)[:, None] + b.sum(1)[None]
    similarity = torch.where(mass == 0, torch.ones_like(mass), 2 * (a @ b.T) / mass.clamp_min(1))
    previous = list(map(float, range(len(b) + 1)))
    for i in range(len(a)):
        current = [float(i + 1)]
        for j in range(len(b)):
            current.append(min(previous[j + 1] + 1, current[-1] + 1,
                               previous[j] + 1 - float(similarity[i, j])))
        previous = current
    soft = max(0., 1 - previous[-1] / max(len(a), len(b)))
    exact = ink.shape == reference.shape and torch.equal(ink.cpu(), reference.cpu())
    return .5 * (float(exact) + soft)


def rloo_loss(log_probabilities, rewards):
    """Unbiased expected-fixed-reward gradient for iid on-policy answers.

    One prompt per group. No optimizer step or policy changes between sample
    collection and replay. No length normalization, reward standardization,
    PPO clipping or best-of-group deployment is hidden in this estimator.
    The rewards and leave-one-out baseline are detached by construction.
    """
    if log_probabilities.ndim != 1 or len(log_probabilities) < 2:
        raise ValueError('At least two independent answers to one prompt are required')
    rewards = torch.as_tensor(rewards, device=log_probabilities.device,
                              dtype=log_probabilities.dtype).detach()
    if rewards.shape != log_probabilities.shape or not bool(
            torch.isfinite(rewards).all() & torch.isfinite(log_probabilities).all()):
        raise ValueError('One finite reward per complete sampled answer required')
    baseline = (rewards.sum() - rewards) / (len(rewards) - 1)
    return -((rewards - baseline) * log_probabilities).mean()


def backward_answer_group(actor, prompt, reference, *, generator, reward_function=None, group_size=4,
                          maximum_glyphs=32, strength=1., replay_tolerance_nats=.05,
                          guard=lambda: None):
    """Accumulate one RLOO group without retaining multiple backbone graphs.

    There is no qualified default reward. The rejected image-edit reward is
    retained only for reproducing its calibration. A separately registered
    caller must qualify and explicitly supply its training-only reward.

    This function performs no optimizer step/zero_grad, source selection or
    resource admission. Its registered caller must own those operations and
    the single GPU mutex. Calling after an optimizer update is off-policy.
    No answer references enter sampling. The unchanged supervised objective
    can accumulate into the same update as an explicit anchor.
    """
    if reward_function is None:
        raise RuntimeError('No qualified default answer reward; do not train the rejected image-edit or selective-template surrogates')
    if not callable(reward_function):
        raise TypeError('An explicitly qualified training-only reward callable is required')
    from .visual_continuation import prefill_visual_prompt

    if actor.training or type(group_size) is not int or group_size < 2:
        raise ValueError('An eval-mode actor and at least two iid samples are required')
    if not math.isfinite(strength) or strength <= 0 or not math.isfinite(
            replay_tolerance_nats) or replay_tolerance_nats <= 0:
        raise ValueError('Positive finite strength and replay tolerance required')
    _binary_images(reference)
    if not len(reference) or not bool(reference.any()):
        raise ValueError('Reward admission requires a visible reference before model work')
    guard()
    with torch.no_grad():
        memory, _, _ = prefill_visual_prompt(actor.interface, prompt)
        samples = [sample_answer(actor, prompt, generator=generator, maximum_glyphs=maximum_glyphs,
                                 memory=memory, guard=guard) for _ in range(group_size)]
    rewards = [float(reward_function(s.ink, reference, stopped=s.stopped)) for s in samples]
    if not all(math.isfinite(r) and 0 <= r <= 1 for r in rewards):
        raise ValueError('Every fixed training reward must be finite and in [0,1]; discard pending update')
    advantage = [(r - (sum(rewards) - r) / (group_size - 1)) for r in rewards]
    event = dict(rewards=rewards, advantages=advantage,
                 varying_reward=max(rewards) > min(rewards),
                 stopped=sum(s.stopped for s in samples), capped=sum(not s.stopped for s in samples),
                 emitted_glyphs=[len(s.ink) for s in samples], language_quality_proved=False)
    if not event['varying_reward']:
        # Exactly zero estimator: do not spend multiple backbone backwards on
        # it. The caller must count these as groups with no learning signal.
        return dict(event, maximum_replay_error_nats=None, replay_performed=False), samples
    largest_error = 0.
    # Authenticate every replay before allowing this group's gradients to be
    # used. On an exception the caller MUST discard the whole pending update.
    for sample, a in zip(samples, advantage):
        value = replay_answer_log_probability(actor, sample, memory, guard=guard)
        error = abs(float(value.detach()) - sample.sampled_log_probability)
        largest_error = max(largest_error, error)
        if error > replay_tolerance_nats:
            raise ValueError('Sample/replay probability mismatch; discard pending gradients')
        (-strength * a * value / group_size).backward()
        del value
        guard()
    return dict(event, maximum_replay_error_nats=largest_error, replay_performed=True), samples
