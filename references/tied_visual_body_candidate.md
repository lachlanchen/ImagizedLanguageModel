# Later small-body control: tied visual-language coordinates

This is an unimplemented candidate, not a change to the current running
experiment. Finish the genuine-context readout control first. No weights were
downloaded and no second model job was started for this note.

The official pinned configurations specify tied input/output embeddings for
[Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct/blob/78448d793a7eb2f7a987a1da76d464384aa1becd/config.json)
and[Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct/blob/ebb281ec70b05090aa6165b016eac8ec08e71b17/config.json).
Their contextual widths are2048and2560respectively. The currently
examined8B-derived body has separate input/output coordinates. These are
architectural facts; they do not establish equal reading accuracy.

Inference: a tied body may remove part of the arbitrary coordinate translation
between an input-vector-anchored glyph codec and the source reader's output
function. In the tied case, the native source logit has form e_j^T h, using the
same e_jthat anchors the offline codec. This still does not make h a clean
glyph code or guarantee that a neural decoder extracts the correct mode.
Subword/first-glyph mismatch, image-code errors, output uncertainty and language
quality remain separate problems. Runtime lexical heads/banks would still be
removed; tying is not permission to select a Unicode ID and render its font.

A width2048target space would also let a2048-wide affine readout avoid the
particular4096-coordinate rank ceiling measured in the current implementation.
That is an architectural inference, not evidence that rank caused all observed
pixel errors. A smaller body requires its own native image-reader check,
numeric-control compilation and actual image-output tests; existing interfaces
hard-code4096and cannot safely load it unchanged.

The local shared Ollama manifest inventory currently contains visual8B/30B
variants and smaller text-only models, but not these visual2B/4Bvariants.
Reuse any verified shared installation if one becomes available; do not
silently duplicate weights. Future comparison must report inherited parameter
counts/pretraining and matched reading/writing quality. No small independent
foundation model or Qwen8Bequivalence is established by this configuration note.
