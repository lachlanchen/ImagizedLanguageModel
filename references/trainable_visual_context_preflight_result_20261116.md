# Complete visual-context updates fit on one RTX 4090

The [registered execution preflight](trainable_visual_context_preflight_protocol_20261116.md)
passes all six real complete-record update/reset trials. In each treatment
trial all 309 contextual tensors receive finite nonzero gradients and change;
in each matched control trial none receives a gradient or changes. Up to
seventeen actual-gradient entries per active tensor match independent GPU
PyTorch AdamW exactly. Every full model and objective state is restored byte
exactly after each update. These are execution measurements, not language scores.

The records are a complete 513-glyph book, an 846-glyph QA record and the longest
native-unit training record (1,042 glyphs, 961 native units). The control and
treatment begin with identical six losses in every case. Supplied-history losses
decrease after each update, but this does not establish autonomous answers,
generalization or semantic understanding.

The [exported evidence](evidence/trainable_visual_context_preflight_result_20261116.json)
reports 149.509 seconds, no pressure waiting, peak allocated VRAM **11.252 GiB**
and minimum available host RAM **42.713 GiB**. Host optimizer moments occupy
11,730,771,112 bytes in treatment and 455,490,728 bytes in control. The deployed
model has 1,465,381,142 parameters, including the 1,409,410,048 inherited context
parameters. Image trainables total 48,547,733; the additional 8,388,608-parameter
teacher projection is training-only. The raw loader's pre-widen image count is
retained separately in the export rather than misreported as the wider model.
Raw runtime identity and process inventory remain in the private original
receipt, whose byte hash is exported. No foreign service was stopped.

All trials are reset and their owner has exited. No trained checkpoint is
retained. The next step is a separately registered matched full-corpus comparison
with a checkpoint format that retains every learned context tensor, complete
emitted-pixel generation, stopping, paired questions, erasures and books. The
old frozen body and all earlier failed actors remain controls. Useful visual
language, independent from-scratch learning, large-model parity and efficiency
improvements remain unproved. None of the rejected rewards was used.
