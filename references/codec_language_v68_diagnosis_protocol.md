# V68: diagnose prediction before and after generated history

Registered after the independently verified autonomous failure and before
diagnostic model calls. Keep the original training, audit, raw outputs and
quality gates immutable. This experiment updates no weights and cannot
qualify language.

## Fixed model and samples

Use the same completed final adaptation and native actor. Require the original
384-generation audit report `0b63ee86c8ee3c4053a5d8e4b1afdf0644ba9b5c9e82c4d8d8c6e5a955c44191`
and independent receipt `ca09f2fb88374406cc2ddfc8e44637776ddbea2fc5c18240ebce83b8df9c026c`,
with all verification checks passing and the original language gates failing.
Use GPU 0 only, after the prior model process exits and its memory is released.
No source-model call, OCR, font lookup, vocabulary head or optimizer is used.

Fix 32 examples. For training, use Python `random.Random(20266840)` to sample
eight sorted admitted CMRC filenames, then eight sorted admitted Dolly
filenames without replacement. Use complete source images and answers.
For development, take the first eight CMRC cases in the existing fixed audit
(four shared-article pairs), then the lexicographically first admitted case
from each of its eight sorted Dolly categories. Do not choose by model score.
The sealed final split remains unopened.

Prepare and hash every owning prompt/reference tensor on CPU before model
execution. Fold training prompts using the accepted lossless fold. Use the
canonical, matched-layout normal pages for development. Remove only the
training cache's two explicit white termination blocks. Preserve ordinary
answer padding and every reference block.

## Three routes

1. **Supplied codes:** encode each target block through the fixed output codec,
   then decode its resulting code. This is image reconstruction with a supplied
   answer, and checks the writing component on these specific images.
2. **Exact history:** use the actual cached inference path to predict the next
   code, pixels and stop probability. Append the true previous block only after
   its prediction has been recorded. Never expose the current or future block
   to its contextual query. Continue through the entire reference for diagnosis,
   even if the recorded stop probability would terminate generation. Record a
   separate terminal stop decision. This route has supplied history and length.
3. **Autonomous:** generate the 16 training cases with the original uniform
   128-block cap and learned stopping. Reuse the existing normal autonomous
   outputs for the 16 development cases; do not re-generate or select alternatives.

For nonempty autonomous outputs, require the first block and first stop
probability to agree with the exact-history route, which starts from the same
prompt. Preserve predicted codes, supplied codes, continuous pixel scores,
reference blocks, stop decisions and unmodified autonomous streams. The codec
decodes in its native FP32 path, as in the original actor.

## Measurements and decision

Verify physical scores offline from the retained arrays. Report by split and
source: first-block fidelity, all-block exact-history fidelity, supplied-code
fidelity, autonomous full-width F1/exactness and stop behavior. Record normalized
code error and the decoded-image margin comparison described in the
[mathematical review](physics_math_after_codec_20261105.md). The latter is a
sufficient-condition diagnostic, not a global robustness certificate.

Poor first-block writing is present before any generated-history drift.
Poor exact-history prediction on training cases, alongside faithful supplied-
code reconstruction, would motivate repairing conditional prediction or fitting
before adding rollout training. Good exact-history writing with poor autonomous
writing would instead motivate a generated-history intervention. Neither
distinction identifies a unique optimizer or proves a semantic representation.
Report failures without changing the original language thresholds.
