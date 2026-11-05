# Offline inherited-context check before image-language training

5 November 2026. The complete image actor fits and executes, but parameter
names/shapes alone do not prove preservation of the source's language behavior.
Run a bounded cross-engine diagnostic before relying on semantic transfer.

This helper alone may reconstruct the source tokenizer and temporarily read
its vocabulary input/output weights. The actor keeps its vocabulary module
absent; no helper or reference-service call is added to image inference.
Preserve the completed actor preflight and both failed precision attempts.

Use eight fixed short English/Chinese prompts in the implementation, covering
arithmetic, translation and elementary facts. Their role is initialization
sanity, not a language benchmark or training set. The exact raw ChatML prompt
must be shared by the owned reference engine and native contextual forward.
Use the cached authenticated archive, no model download, physical GPU 0 only,
and close the owned reference server before loading the native model. Never
call the shared service on GPU 1.

Reference settings: raw prompt, no thinking, temperature zero, top-k one,
top-p one, repetition penalty one, context 2,048, output limit eight, seed
20266411. Request ten raw top log probabilities at every returned position.
Map token bytes back to unique source vocabulary rows, with no lossy decoding.
Reconstruct the source Qwen2 BPE from GGUF metadata; check exact prompt round
trip and source/native prompt token counts. Unknown or ambiguous rows are
errors, not dropped observations.

For each prompt, native teacher forcing uses that prompt and the preceding
reference continuation. Compare next-token predictions at exactly the source
positions. The reference top ten IDs plus an outside-mass bin define a
coarsened distribution; compute KL(reference || native) on these same bins,
not a renormalized top-ten or purported full-vocabulary KL.

Sanity acceptance: all eight cases valid; prompt counts match; first-token
agreement at least 0.75; reference-position greedy agreement at least 0.85;
mean coarsened KL at most 0.05 nat; finite logits; actor vocabulary remains
absent and adapters remain at zero output. Preserve the exact numerical
differences and failures. This does not prove full engine equivalence, visual
understanding, ILM writing quality or broad semantic transfer. No development
or sealed data is used. Commit the code, evidence and interpretation; then
register actual writing/instruction learning from the measured state.
