# Balanced first-decision writer refinement

This is a new training experiment after the completed contextual/readout-only
comparison. Preserve that comparison, its rubric and all failures. Its reserved
questions are now development evidence, not an untouched test for this repair.

Initialize from the completed contextual checkpoint
`5c434a9a9b0c6ddc0c3fe4005265e4f8ca5b7d0add9a74a80ed371806fa71085`.
Freeze the entire image reader, all four learned contextual adapters, image
encoder, image decoder and normalization buffers. Train only the existing
17,848,833 readout/stopping parameters. No architecture or runtime output
substitution is introduced. Inherited contextual training costs remain part
of the method; this is not a new small foundation model.

Cache the actual final hidden states of the fixed learned tail for every
training sequence, using the original frozen-prefix cache. Cache equality
requires recomputing every pre-refinement training pixel score and stop count
against the completed qualification records. The state predicting position t
sees only preceding reference images. Store first, later and terminal states
in separate pools. No held-out images, answers or generated answers enter
these pools. Cache conversion uses no vocabulary head.

Run exactly 18,000 new AdamW updates with seed 202611121, learning rate 1e-4,
cosine decay to 10%, weight decay 0.01, gradient norm cap 1.0. Three updates
out of every four sample 64 first states, 64 later states and 64 terminal
states, independently with replacement. Use

\[
 L_{image}=\tfrac12 L_{BCE}(D(G(h_0)),I_0)
             +\tfrac12 L_{BCE}(D(G(h_{>0})),I_{>0}),\quad
 L_{stop}=\tfrac12 E\,softplus(s(h_{continue}))
             +\tfrac12 E\,softplus(-s(h_{terminal})).
\]

Each BCE balances foreground and background within that sampled group. The
other update rehearses 128 existing genuine-copy features with the previous
pixel-only objective. These original copy features come from the frozen source
body, as in the parent training recipe; they are not falsely relabelled as
features from the adapted body. Copy performance must subsequently be checked
through the actual adapted full reader. No source-code or semantic-anchor loss
is added. With a frozen reader these cached states give the exact restricted
gradient for G and stopping; they do not train the reader or expose it to its
own generated errors.

The new run adds 864,000 sampled first-decision targets, 864,000 later targets
and 576,000 copy targets. This changes both exposure and position weighting;
an improvement would not isolate which change caused it. Preserve final
weights even if the training gate fails. Never select among checkpoints using
held-out answers. Parent optimizer state is not reused; report the two stages
and counters separately.

Reserve 16 new development contexts with two questions each, using the fixed
SHA256 ordering namespace `balanced-glyph-writer-fresh32-v1`. Exclude every
training context, the prior sixteen reserved contexts, and contexts in the
earlier development manifest. Do not filter on answer text, form coverage,
teacher success or model output. Prepare prompts from the existing physical
dataset and inspect all32 complete inputs before fitting. Pretraining exposure
remains unknown; this is not exhaustive historical decontamination.

Evaluate the unchanged parent and refined writer on the same new32 cohort,
plus the existing eight training controls and sixteen normal/question-erased
development pairs. Reuse the parent's unchanged physical qualification only
with authenticated weight and report identities; the refined writer needs a
new physical qualification including all256 held-copy inputs through the full
adapted reader. Retain failed gates. Keep the original72-stream assessor,
output-first reviewer and scorer unchanged. Inspect both complete output sets
before either new reference score. The pilot remains24/32 clear complete
literal matches and10/16 contexts with both questions correct. A pilot pass
alone is not the full ILM goal.

Use the common GPU0 mutex, one neural job at a time, existing resource guards,
and no simultaneous publication build. Cache and fit have bounded deadlines;
interruption must preserve status and previous checkpoints, not masquerade as
a completed run. This protocol makes no claim that balancing will solve
multi-modal pixel averaging or autonomous feedback drift. Those are distinct
remaining hypotheses.
