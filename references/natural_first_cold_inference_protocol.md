# Standalone replay after natural first-image training

Use the first fixed third-cohort input, `fresh/0000/normal`, irrespective of
whether its eventual answer succeeds. The bound natural-writer condition must
have a completed final 36,000-update fit and all 72 actual assessment streams.
Keep the learned stopping rule and 64-glyph cap. This check establishes one
execution replay, not general language accuracy.

The separate verifier launches `infer_natural_first_language.py` from an empty
working directory and empty library cache, with offline settings. Only its child
sees GPU0. It retains the normal project mutex, host/device admission and stage
limits. It reuses the existing authenticated base weights rather than copying
or downloading a second model. The verifier never opens reference answers.

Capture actual file opens and connection attempts. Apply the preserved older
corpus/font/tokenizer/IP check, then reject undeclared files anywhere under the
project artifact tree. The artifact exceptions are the exact prompt, bound
checkpoint, fixed numeric task controls, GPU mutex, and two original-owner
status/event files needed by resource admission. The replay's own output and
directory traversal are allowed. No other cache, training state, assessment
output, glyph inventory or answer reference may enter the child. Retain raw
traces privately and publish only source-bound verification evidence. An
archive may contain unused lexical metadata; the architecture checks establish
which tensors are materialized. A trace does not prove unused bytes absent.

Compare every actual output tensor and event field, including decision states,
binary glyph images, stop/cap decisions and emitted-image feedback hashes, with
the previously retained assessment stream. Require exact PNG bytes as well as
matching input/checkpoint identities and a hashed saved observed prompt. The
parent verifier remains unchanged. No new successful replay is claimed until
this registered child actually runs and passes.
