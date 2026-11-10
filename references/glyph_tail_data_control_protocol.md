# Separate broader supervision from contextual adaptation

This control is fixed while the 3,176-question cache is still being completed,
before the main fit or any new adapted-model answer. The main candidate and its
prospective pilot criterion stay unchanged. The control is not a checkpoint
search and must not be chosen post-hoc as the only reported model.

The contextual candidate changes both training coverage and the sequence model.
Its improvement, if any, would not alone establish which change mattered.
The native-head counterfactual also does not prove that the frozen state lacks
information for an independently trained glyph readout. A matched readout-only
condition is therefore necessary before attributing a gain to the four adapters.

Use every original frozen `original_final_predictors` tensor already stored in
the complete 3,176-case cache. These are genuine source states under the same
prompt and reference-image history as the contextual candidate's layer-32 inputs.
Keep the whole source body fixed. Train only G and stop, starting from the same
qualified pixel-only checkpoint, with the same sequence pixel/stop objective,
18,000-step schedule, three question updates per copy-rehearsal update, 1e-4
readout learning rate, cosine decay, weight decay and clipping. Use the same
prescribed seed and sampler. The main run does not retain every sampled index,
so do not claim observed bitwise sample equality without further evidence.

Initialize the tail and bridge in the same order before discarding the control's
unused tail, so parameter initialization does not silently advance a different
random stream. Retain the control's complete sampled-index trace and compare
available final RNG-state fingerprints with the main checkpoint. Those checks
are execution diagnostics, not substitute language scores. Record all setup
memory, including unused-tail initialization; do not reset the peak counter to
advertise a smaller control footprint. Export zero contextual adapter-up weights
with the control checkpoint, explicitly mark zero trained body parameters, and
use the same image-only inference path.

Reuse the already computed feature files by reference rather than duplicating
them or running another reader cache. Fitting must be serial with the main job
under the same GPU mutex. Keep every physical qualification outcome. Generate
the same 72 complete image-answer streams for both final models, preserving
all raw pixels, stops and caps. Inspect both models' outputs before opening the
new 32-question reference strings. No loss, schedule or checkpoint may be
changed using that assessment. The fixed physical gate and primary pilot target
remain reported separately for each condition.

If both conditions improve, the experiment supports broader visual supervision
without isolating a unique need for body adaptation. If only the adapted body
improves, it supports that intervention under this one fixed training recipe.
If neither improves, the proposed repair has failed. None of these outcomes
alone proves general language learning from scratch, historical-script reading,
LLM parity or a new physics-based principle. The experiment must report both
final conditions and all uncertainty, not select a favourable explanation.
