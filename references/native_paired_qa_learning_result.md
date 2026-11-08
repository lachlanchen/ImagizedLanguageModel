# Complete paired-question learning fails its language requirements

The registered 128-update QA adaptation produces **0/32 seen and 0/32 held-out
source-accepted OCR-exact answers**, unchanged from its initial checkpoint.
Neither model answers both questions correctly in any of the sixteen held-out
passage pairs. All six prospective exploratory requirements fail. This run
does not qualify a useful image-language model.

| Complete fixed cohort | Initial | Final |
| --- | ---: | ---: |
| Seen Chinese QA, source-accepted OCR exact | 0/32 | 0/32 |
| Held-out Chinese QA, source-accepted OCR exact | 0/32 | 0/32 |
| Held-out passage pairs, both answers correct | 0/16 | 0/16 |
| Held-out questions, question ink erased | 0/32 | 0/32 |
| Held-out questions, passage ink erased | 0/32 | 0/32 |
| Seen Chinese/English copies, canonical OCR exact | 18/32 | 19/32 |
| Held-out Chinese/English copies, canonical OCR exact | 19/32 | 16/32 |

The copying-retention requirement permits losing at most two held-out exact
copies; the observed loss is three. A one-case seen-copy gain is not evidence
of semantic learning. Identical question-erased inputs and seeds produce
bit-identical actual tensors in all 32 initial/final erased-page pairs.
That execution check establishes deterministic input handling, not answering.

## Complete evidence and raw review

The [complete export](evidence/native_paired_qa_result/README.md) retains all
384 original initial/final answer PNGs and 128 actual pre-update training
images, together with inputs, references, original evaluation observations,
source bindings and galleries. This includes 320 new native assessment calls
and 64 reused initial-copy outputs. Independent export recomputation
authenticates every retained model PNG against its actual decoder tensor and
reproduces all counts and prospective conditions.

All 384 original answer pages were visually inspected. Natural QA and erased
controls contain malformed passage/question fragments, answer labels and
extra writing. Copies are often legible but still have omissions, substitutions,
repetition, extra quotes or instruction fragments. Some final pages are very
faint. No contrast enhancement, rerendering, inferred transcription or manual
accuracy replacement is used. [Raw-review coverage](evidence/native_paired_qa_result/raw_visual_review.json)
binds all eight cohorts without claiming an independent blinded evaluation.

The OCR evaluator recognizes only 23/32 seen and 26/32 held-out canonical
reference answers exactly. These limitations remain material: scores are
OCR observations, while raw visual inspection separately rejects reliable
answer generation. No uncertain reading is silently counted as correct.

## What ran

The final checkpoint follows four complete permutations of 32 natural Chinese
questions from sixteen passages, for 128 optimizer updates. All 206 adapter
tensors change, actual optimizer states reach step 128 and the complete
schedule and RNG bindings authenticate. The successful training invocation
takes 12,782.358 seconds (3.55 hours), with 12.204 GiB peak allocated GPU memory.
The earlier save-path failure occurs before an optimizer update and remains
separately recorded. The [training verification](evidence/native_paired_qa_training_completed.json)
preserves its scope, first native/differentiable tensor equality and all
actual training-page observations.

The runtime retains a 3,899,903,488-parameter image transformer, including
24,358,912 trainable adapter parameters, and an 84,046,115-parameter codec.
The 560,528,640-parameter recognition reader supervises training only.
External generator/codec pretraining remains part of the system. Standalone
inference receives input images and the same generic fixed condition, with
no expected answer, varying question text, teacher service or output repair.

The [prospective protocol](native_paired_qa_learning_protocol.md) fixes the
data, final checkpoint, seeds, sampler and requirements. Public CMRC cases
have been reused in development, so this is not a pristine benchmark.
Source reports have local filesystem paths made relative in the public
package; original and public report hashes are retained separately. All
original image bytes remain unchanged.

## Research decision

Completed optimization and a lower training objective have not become
correct answers, even on the seen questions. This result does not support
book-scale continuation of the unchanged recipe. It does not identify a
universal impossibility or a unique causal bottleneck.

A separate pretrained visual-language-body-to-image-generator route has
completed actual native replay, codec and one-update checks. Its longer
[image-pair learning experiment](native_visual_flow_learning_protocol.md)
is running. Its execution feasibility, inherited language pretraining and
future answer quality must be reported separately from this failed QA run.
Free language generation, historical-script understanding and efficiency
at matched useful language quality remain unachieved.
