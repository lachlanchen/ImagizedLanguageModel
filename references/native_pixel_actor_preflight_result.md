# Native image-to-pixel integration fails its numerical preflight

The complete native-context candidate has not qualified for training. The
registered preflight exits after 209.257 seconds, before any optimizer update,
when cached and whole-prefix teacher-conditioned pixel logits exceed the fixed
maximum absolute difference of 1e-4. No autonomous answer image is generated.
The [failure archive](evidence/native_pixel_actor_preflight_failure/report.json)
retains the original traceback, terminal status, preparation, case identities
and all five prompt/reference PNG pairs. Code and source-data hashes remain
unchanged. The failure is preserved without a retry or relaxed tolerance.

The implemented path retains full native prompt KV, replaces the output-code
flow with a direct physical-pixel density, and encodes emitted images as
history. The [protocol](native_pixel_actor_preflight_protocol.md) separately
requires prefix causality, real gradient updates, exact optimizer replay,
learned stopping and autonomous complete outputs. Execution reaches the pixel
cache check after native prompt prefill, but the later requirements remain
unverified. Neither a completed implementation nor a loaded model establishes
language learning.

The failed comparison scores all 256 physical positions under the same true
preceding pixel patterns. It compares logits, not independently generated
answer pixels. The exact error magnitude and failing tensors were not saved;
cached/uncached greedy agreement was not reached. Consequently this record
does not identify floating-point accumulation, an implementation defect or
another unique cause. A follow-up needs retained numerical diagnostics before
another full-reader run. Increasing the tolerance would not diagnose it.

The process exits and GPU0 returns to 16 MiB; GPU1 remains at 7,178 MiB.
The elapsed time includes model loading and checks, with zero training
updates. Peak allocated VRAM was not recorded on this failure path, so the
registered 22 GiB allocation bound is not a measured result.

The strongest completed writing result remains the
[512/512 exact training first-block fit](pixel_density_long_fit_result.md),
with 0/192 exact development first blocks. That head used cached image-derived
states and does not establish complete prompt-image to answer-image language
understanding. The earlier [full native actor](native_chinese_v80_result.md)
also fails its held-out quality criteria despite question-dependent output.
The current candidate inherits a large externally pretrained reader; it is
not evidence of a lightweight foundation model trained from scratch.

The unresolved goal is useful language generalization through image input and
image output. Exact fitting, question sensitivity and numerical execution are
separate requirements; none may substitute for correct new answers.
