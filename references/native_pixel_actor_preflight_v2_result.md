# Actual PNG-to-pixel learning path passes execution checks, not language quality

The repaired native actor completes its registered preflight on one RTX 4090 D.
Actual prompt PNGs enter the full native reader; the direct physical head emits
binary images; the history encoder consumes those actual emitted images; and
the model makes its own stop decisions. Two retained optimizer updates exercise
the density, image-history embedding, start query, contextual suffix adapters
and stop head. Every required component has finite nonzero gradients. Saving
after update one, executing update two, restoring and replaying update two gives
bit-identical model, optimizer, loss and gradient measurements. There are three
actual optimizer calls and two retained updates, not three learning exposures.

The [complete output audit](evidence/native_pixel_actor_preflight_v2_result/audit.json)
finds **0/2 exact training answers and 0/3 exact development answers**. All
five stop without hitting the 16-block safety cap, but every length is wrong:
12/2/2/4/4 generated blocks versus 5/4/3/2/8 reference blocks. Foreground F1
ranges from 0.1132 to 0.1899. Actual inspection of all five unique raw outputs
finds fragmented writing rather than correct answers. These two updates are an
execution and gradient check, not a substantive language-training experiment.

The head's FP64 arithmetic repairs the earlier numerical fault under the
unchanged 1e-4 gate. Cached/full teacher-logit error is 1.74971e-13, with exact
greedy agreement and zero target-suffix leakage in the tested block control.
Native full-history/incremental start states have relative normwise discrepancy
2.11771e-6 and identical emitted first-block pixels. Changing a future image
leaves preceding query states unchanged before and after learning. All five
native PNG prefills, including complete KV hashes, remain bit-identical after
suffix learning. Frozen source parameter identities and versions are unchanged.

The first autonomous answer and its replay are bit-identical. The independent
CPU audit checks every output PNG against emitted tensors and verifies 37
actual image-feedback records, including the repeated stream and one separately
forced feedback control. CPU re-encoding differs from saved GPU codes by at
most 3.99351e-6, within the registered 1e-4 bound. The forced control receives
no autonomous-answer credit. This audit does not independently replay the full
inherited reader or optimizer; the actual optimizer replay is the producer's
separate execution check. No semantic accuracy is assigned.

The complete actor contains **7,543,984,691 parameters**, of which **13,855,681
are trained**. The direct physical head contains 2,666,944 FP32 stored
parameters with FP64 arithmetic; suffix adapters contain 6,193,152 parameters.
The reader retains its existing arithmetic and substantial external
pretraining. The old output-code prediction flow is removed; its codec is
retained only for encoding observed history, and decoder calls are guarded
against use in output generation. This remains a large pretrained-body
adaptation, with no tokenizer, OCR or teacher service in deployed generation.
It is not evidence of a lightweight foundation model learned from scratch.

The execution, including model loading, checks, updates, six complete streams
and saving, takes 163.349 seconds and peaks at **17.953 GiB allocated VRAM**.
This is not a training-throughput estimate or a matched-quality efficiency
comparison. The process exits; GPU0 returns to 16 MiB and GPU1 remains at
7,178 MiB. No GUI runtime is launched.

The [source-bound report](evidence/native_pixel_actor_preflight_v2_result/execution/report.json),
[gradient measurements](evidence/native_pixel_actor_preflight_v2_result/execution/updates.json)
and [raw-image review/manifest](evidence/native_pixel_actor_preflight_v2_result/review_and_manifest.json)
retain every output and relevant evidence. Original checkpoints, numerical
tensors and feedback records remain at the hashed paths under
`artifacts/native_pixel_actor_preflight_v2/`. The earlier
[failed preflight](native_pixel_actor_preflight_result.md) and rejected
normalization-only control remain unchanged as historical results.

The next learning experiment must train on complete natural question-answer
image streams and assess held-out questions, including distinct questions over
the same passage. This working execution path removes an integration obstacle;
it does not supply the missing evidence of language generalization, historical
script understanding or a usable independent ILM.
