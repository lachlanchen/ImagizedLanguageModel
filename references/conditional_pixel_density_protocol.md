# Conditional physical-pixel likelihood: first-block learning protocol

This prospective probe trains a writing head on frozen input-only contextual
states. It tests whether a normalized image density can learn precise first
blocks from this representation. It does not qualify complete answers, native
PNG-input inference, stopping, broad semantics or a language model trained from
scratch. The general ILM objective remains independent image-to-image language
generation. The [decision record](conditional_writing_decision_20261109.md)
connects this test to the previous failures and related work.

## Fixed sources and representation

Use the complete source-bound V69 cache, whose capture and corrected audit
hashes are checked by the implementation. Its two 4096-dimensional states
come from the V68 custom Qwen3-VL image actor, after image-only prompt forward
calls. This interface differs from the source's native QA interface. The
native source's 13/16 Chinese text-answer result must not be attributed to
these states or to this probe. No backbone is loaded or updated here.

The 512 training records contain 256 Chinese CMRC and 256 English Dolly
examples. The reused development set has 128 Chinese questions in 64 article
pairs and 64 English questions, each with normal and registered erased-input
states. Development data are not a sealed test. Former V69 fit/calibration
records are all training records for this fixed-schedule experiment; no head
selection uses development images. Retain the owning 32-by-64 first reference
block, including any clipped character at its physical boundary.

Pack each binary block into 256 adjacent 2-by-4 pixel pieces, row by row.
Within a piece use little-endian physical bit order. Packing and unpacking
must be exact inverses on every target. All 2^2048 binary images are
representable. This is a discrete pixel-pattern alphabet, not a lexical or
Unicode vocabulary; it does not eliminate all discrete variables. No font,
glyph table, answer retrieval or teacher is used for head generation.

## Architecture, likelihood and schedule

Use four causal attention layers, width 192, six heads and feedforward width
768, with LayerNorm, GELU and learned physical positions. Flatten the two
reader states and standardize each coordinate using the 512 training records
only; population standard deviation has floor 1e-4. A linear projection adds
the condition to each physical position. Input patterns enter as their eight
signed bits through a linear projection, shifted right after a learned start
vector. Predict 256 possible patterns with a normalized softmax. Train the
mean negative log likelihood of all 256 pieces (all 2048 physical pixels),
not squared error in arbitrary image-code coordinates.

Seed 202611091 fixes initialization and 75 complete random training epochs:
1,200 logical updates, batch 32, 38,400 retained example draws. Use FP32,
AdamW (betas 0.9/0.95, weight decay 0.01), peak learning rate 3e-4, 32-update
linear warmup, cosine decay to 3e-5 and gradient-norm clipping at 1.0. No
dropout, sampling-temperature sweep, early selection or checkpoint choice.
The final update is the sole assessment checkpoint. Record initial/final
supplied-prefix NLL separately from autonomous generation.

Before training, change the target suffix starting at index 128 and require
exactly unchanged logits through index 128. Require altered later logits,
an active condition path, and cached/full-prefix logits within 1e-4 over
all 256 positions. Cached and uncached greedy output must be bit identical.
After two actual updates save model and optimizer; execute update three,
restore update two and replay update three. Loss, gradient norm, parameters
and optimizer state must match exactly. The extra replay adds one optimizer
call and 32 executed draws; it adds no retained learning exposure. Any failure
is preserved and stops this execution, without automatic retry or a changed
tolerance.

## Assessment and interpretation

Generate all 512 training and 192 development normal blocks, and all 192
development erased-input blocks. The head receives only contextual states and
its own emitted pixel prefix. Greedy generation has 256 physical steps and
no supplied answer pixels. This fixed block size supplies no full-answer
length or stop evaluation. Archive every unmodified output mask and PNG.

Independently score all 896 outputs against their owning first reference
blocks: exact physical-block agreement and foreground F1 by language/split/
route. Report the normal-minus-erased Chinese F1 effect, its 2,000-resample
95% article-bootstrap interval (seed 202611092), and pairs with both normal
first blocks exact. The current erased route is inherited unchanged; its
original prompt geometry and erasure definition must remain explicit.
Display all outputs with their references at native pixel scale; display
enlargement never enters scoring. No OCR repair or post-hoc target selection.

A useful first-block result would require at least 90% exact blocks on both
training languages and at least 50% exact normal blocks on both development
languages, with a positive Chinese question-effect lower interval bound.
These are prospective component criteria, not a definition of general
language understanding. Falling NLL, training memorization, sharp isolated
glyphs or first-block accuracy cannot qualify full-answer language generation.
Even if the criteria pass, the next required stage is actual PNG-to-context
inference, past-image feedback, learned stopping and complete held-out answers.

The offline audit independently checks packing, source/image bindings and
every physical score. It does not replay all inherited neural forwards or
claim an independent semantic benchmark. Earlier code-density and adaptive
heads already exist; this is a specific remaining combination, not the first
autoregressive pixel likelihood.

## Resource admission and reproduction

This experiment loads only a head with fewer than four million parameters,
not the multi-billion-parameter reader. Acquire the existing project GPU0
lock; require at least 24 GiB available RAM, swap at most 75%, and at least
4 GiB GPU0 free. Preserve all other projects and GPU1. The 4 GiB head budget
is separate from the earlier 20 GiB cold-load threshold for full backbones.
Stop if allocated head VRAM exceeds 4 GiB or the execution exceeds 1,800
seconds. Record actual memory and wall time. No project GUI is required.

Use the existing LocalLLM image-generation Python runtime and PYTHONPATH=.
Preparation and audit require CUDA_VISIBLE_DEVICES=''. Execution requires
CUDA_DEVICE_ORDER=PCI_BUS_ID and CUDA_VISIBLE_DEVICES=0. With a fresh RUN:

```bash
python scripts/train_conditional_pixel_density.py --stage prepare --output RUN
python scripts/train_conditional_pixel_density.py --stage run --output RUN
python scripts/audit_conditional_pixel_density.py --output RUN
```

Commit implementation and protocol before preparation. Bind all source
hashes, prepared data and the schedule before GPU execution. Preserve failed
attempts; changes require a new registered execution directory.
