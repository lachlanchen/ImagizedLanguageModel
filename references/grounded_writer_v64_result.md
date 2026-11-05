# Grounded writer: completed reconstruction test

5 November 2026. The fixed 10,000-update writer-only experiment is complete
and fails its registered component gate. It provides no language-quality
qualification. The [protocol](grounded_writer_learning_protocol.md) and all
model/training source bytes remain unchanged.

| Measurement, 128 development blocks | Correct image condition | Cyclically shuffled condition |
| --- | ---: | ---: |
| Foreground F1 | 34.000% | 21.420% |
| Exact blocks | 0/128 | 0/128 |
| Blank blocks | 1/128 | 0/128 |

The paired mean conditioning gain is 12.581 percentage points. Both fixed
requirements fail: aligned F1 at least 90%, gain at least 20 points. These
are reconstruction tests using source-answer-image features, not answers to
new prompts. No confidence interval or language-generalization claim is added.
The first eight manifest cases were inspected without selecting for quality;
their generated strokes remain fragmented.

The 19,374,096-parameter flow writer trains on balanced Chinese/English blocks
from 12,976 complete-line examples, with all inherited vision/context weights
frozen. Training reaches 15.801 GiB peak allocated memory. The resumed audit's
CUDA peak resets to 15.514 GiB and must not be reported as the whole run's peak.
The durable update-10,000 training elapsed counter is 1,499.890 s; the final
resumed report records 1,553.787 s including evaluation. The timer resets after
model loading/checkpoint restoration, so both exclude those setup operations
and external foundation pretraining. They are not complete wall-clock costs.

The final EMA was saved before the initial evaluator failed to create its
destination. No development output existed at that failure. The recovery
preserves the initial failure/status, creates only `development/`, and resumes
unchanged code from update 10,000. Final EMA, durable checkpoint, metrics and
sample-order journals remain byte-identical. There are no extra learning
updates. The one earlier optimizer replay belongs to the registered exact
resume check. Both journals contain exactly updates 1–10,000.

The [CPU verifier](../scripts/verify_grounded_writer_result_v64.py) independently
recomputes all 256 generated-image scores, validates the fixed manifest and
quality decision, and binds reports, raw rasters and source hashes in the
[receipt](evidence/grounded_writer_v64_result.json). Final EMA SHA-256:
`c96186e2089b3f7f24f8461b0db532fa810638e83dc695f1b3db18b961365336`.

Local artifacts remain in `artifacts/grounded_writer_learning_v64_20261105/`:
`writer_final.pt`, current/previous checkpoints, all raw development rasters,
fixed manifests, initial failure, recovery receipt, complete training journals
and `first_eight_review.png`. GPU 0 is released; no project GUI or model service
is left running. Shared GPU 1 services were untouched.

The result localizes an immediate failure of faithful image writing. It does
not identify whether the two-feature low-resolution encoder interface,
decoder conditioning, optimization or their interaction causes the failure.
The next experiment should distinguish those possibilities with matched
visual-information controls before adding semantic training or reducing
sampling steps. The [physics/mathematics update](physics_math_action_update_20261105.md)
explains that decision. A useful independent ILM remains the goal.
