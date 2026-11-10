# Missing first-target coverage and damaged known forms

An offline join of the original training labels and saved first-image diagnostics
separates two limitations. Five of the sixteen reused Chinese cases have no
allowed reference first form seen initially in language training. All five have
first-image F1 below 0.95. Among the other eleven, nine have an allowed reference
form nearest in code cosine, but only one image is pixel-exact. A familiar
target form is therefore insufficient to ensure a correct generated image.

| Original language-target coverage of an allowed first form | Cases | Nearest allowed reference in code | Exact first images | First-image F1 at least 0.95 |
| --- | ---: | ---: | ---: | ---: |
| Seen at an initial answer position |11|9|1|5|
| Seen only later in an answer |2|0|0|0|
| Never seen as a language target |3|0|0|0|

The original 508 questions provide 3,855 positive glyph events, 1,074 target forms
and 266 initial target forms. The primitive encoder/decoder trained on 4,087 forms:
"never seen as a language target" does not mean an unseen primitive glyph.
The two later-only reference initials are 玛 and 米; the absent language-target
initials are 批, N and 氨. The nearest-code diagnostic does not supply or repair
runtime images. Upper/lower case and all source aliases remain as recorded.

All nine nearest-reference successes fall in the initially seen group. Four
such cases still have first-image F1 near 0.60–0.64 for 上, 亚, 8 and 因. This is
compatible with imperfect motor coordinates despite a useful nearest form;
it does not establish a causal mechanism. One case allows two alternative
initial forms and is grouped by any allowed reference. The entire per-form
occurrence count, rank and image score is retained in the
[complete sixteen-case report](evidence/glyph_target_coverage_v1/report.json).

The audit validates the training inventory and geometry checkpoint hashes and
reproduces the prior aggregate nine-nearest/one-exact counts. It performs no neural
forward, optimizer update, GPU operation or new image generation. It is a
small descriptive analysis of reused cases, not a sealed accuracy estimate
or a controlled causal effect of coverage. No full answer is scored here.

The running genuine-copy curriculum already covers all 4,086 visible primitive
forms as initial targets and mixes them with original language histories.
Its actual result will test whether this broader contextual coverage helps;
these observations do not guarantee it. The
[conditional direct-pixel ablation](direct_pixel_readout_protocol.md) is a
separate, unexecuted test of the motor objective if that physical gate fails.
It uses the same data, retains the architecture and actual-image feedback
encoder, and does not infer that new image-generation machinery is required.

Reproduce from the existing metadata and saved first-state report:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. python scripts/audit_glyph_target_coverage.py \
  --output artifacts/glyph_target_coverage_v2
```

No model, source vocabulary head, font or target image is loaded by this audit.
