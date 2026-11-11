# Visual language interfaces: primary-source check

The completed compact diagnostic identifies representation and conditional
writing failures. Published results help frame candidate interventions; they
do not establish the cause or success of this implementation.

[PIXAR, ACL Findings 2024](https://aclanthology.org/2024.findings-acl.874/)
provides an established pixel-input, pixel-output autoregressive language
model. Its authors study noisy rendered-text generation and an adversarial
training stage. The project cannot claim the first visual autoregressive
language model. Its existing PIXAR/MIXAR baselines and failed adaptations must
remain part of the decision history; re-reading the paper does not justify
repeating an unchanged failed recipe.

[TextLDM, May 2026](https://arxiv.org/abs/2605.07748) reports that reconstruction
fidelity alone is insufficient for useful continuous language latents and
that alignment to frozen language-model features improves generation. Its VAE
encodes discrete tokens, so its deployed representation does not meet this
project's image-only interface objective. The relevant inference here is to
measure prediction-relevant representations separately from codec fidelity,
not to import its reported performance as evidence for ILM.

[On the Design Fundamentals of Pixel Text Representation Learning, September
2026](https://arxiv.org/abs/2609.01147) studies visual text encoders, including
resolution/font variation, layout, grounding and multilingual training. Its
abstract reports Pixel Linguist II training over 280M examples and evaluation
on representation/retrieval tasks. It is a potential source of visual-reading
methods, not proof of pixel-output language generation or a one-GPU training
budget. Broader font and layout training remains downstream of establishing
usable conditional writing in the current model.

These primary abstracts/metadata were checked on 11 November 2026. The next
training intervention is not selected by an analogy to physics or a claim of
novelty. The [measured diagnostic](compact_visual_span_diagnosis_result.md) and
a same-cohort comparison with the earlier readable model determine it.
