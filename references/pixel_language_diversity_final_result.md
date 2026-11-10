# Broader pixel-language training: final result

The registered 512-question run fits most training answers but does not
establish useful unseen-answer generation. At 16,000 updates, 495 of 512
training answers are pixel-exact. No normal development response is
pixel-exact, and inspection of all 24 normal outputs identifies no usable
correct complete answer. The current recipe has a substantial generalization
gap despite its ability to reproduce familiar answer images.

All [560 final outputs and 140 display sheets](evidence/pixel_language_diversity_final16000/)
are retained, including failures. No output pixels are repaired or selected
by quality.

| Final assessment group | Cases | Pixel-exact complete answers | Foreground F1 | Learned stops | Cap failures |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original training controls | 32 | 29 | 0.978044 | 32 | 0 |
| Added training examples | 480 | 466 | 0.990639 | 480 | 0 |
| Chinese development, normal | 16 | 0 | 0.336570 | 16 | 0 |
| Chinese development, question erased | 16 | 0 | 0.225725 | 16 | 0 |
| English development, normal | 8 | 0 | 0.059018 | 8 | 0 |
| English development, prompt erased | 8 | 0 | 0.040126 | 8 | 0 |

Both complete answers are pixel-exact for 239 of 256 training passage pairs
and none of the eight Chinese development pairs. On the same 64 training
examples assessed at every interim stage, exact matches progress from zero
initially to 3, 21 and 60 at updates 4,000, 8,000 and 16,000. The final
assessment additionally covers every other training example.

Whole-training correct-history image NLL falls from 9.749791 initially to
0.251025, 0.020290 and 0.000259. It averages physical-piece losses within
blocks and questions; it is not semantic accuracy or an unscaled full-answer
joint likelihood. Chinese normal-minus-erased foreground F1 is 0.110845;
English is 0.018892. Input-dependent overlap does not establish correct meaning.

The [first-image analysis](evidence/pixel_language_diversity_final16000/prefix_diagnosis.json)
finds all 512 training first images exact, while none of the 24 normal
development first images matches an allowed reference. A physical mismatch
is already present before any earlier complete generated image can be fed
back. Autoregression within that first image remains a possible contributor.
The 17 training streams with an exact first image and an incorrect complete
answer also retain a later continuation or stopping failure. These checks
do not isolate a single mechanism or assign semantic grades.

All normal development outputs were viewed on output-only sheets before the
current source-reference comparison. All 24 erased outputs and the first
four examples from each training group were also viewed. The other 504
training outputs were checked numerically and retained. The
[source-first notes](evidence/pixel_language_diversity_final16000/source_first_notes.json)
remain separate from the
[later comparison with all 24 source questions and full references](evidence/pixel_language_diversity_final16000/manual_review.json).
These cases were reviewed in earlier experiments: this is not a sealed or
fully blinded evaluation. There is no independent human grading or formal
semantic-accuracy estimate.

Most development strings contain incomplete or invalid forms. The two
date-like outputs are approximately `183?年` and `200?年`, with damaged final
digits; the source references require `1874年` and `2001年`. The first visible
prefix disagrees with its reference, while the second has no confident
complete year. Missing strokes are not reconstructed toward gold. Source
annotations and prepared pixels remain frozen throughout the trial.

The [independent full-run audit](evidence/pixel_language_diversity_final16000/independent_full_run_audit.json)
checks all 896 streams across the four assessments. Every PNG matches its
emitted tensor, complete-image scores and stopping/cap events match, and
2,731 actual feedback records are re-encoded on CPU. Maximum CPU/GPU feedback
error is 7.57e-6, below the registered 1e-4 tolerance. All 32 original control
prompt prefixes are bit-exact after learning. These checks verify the tested
recording and numerical paths; they do not prove semantic correctness of the
imported reader or learned readout. The generic display-review report's
pending-audit field predates this separate completed full-run audit.

The run retains 64,000 example draws, with 124 or 128 exposures per question.
Its optimizer ledger contains 16,128 actual calls, including 128 calls
discarded by the [resource recovery](pixel_language_diversity_resource_recovery.md).
Total primary active-process time is 42,549.17 seconds, or 11.82 hours,
including both processes and assessments. The cumulative 12-hour limit was
preserved. The completed process reports a peak allocation of 18.138 GiB;
this is a process-local measurement, not a measured maximum across both
processes. Both used the 22 GiB guard. External pretraining and independent
audit compute are outside this timing.

The [owning-state receipt](evidence/pixel_language_diversity_final16000/owning_state_receipt.json)
binds the assessment to its complete checkpoint, preparation and 286 learned
tensors. The actor includes a 7.54B pretrained body with 13.856M locally
trained parameters. It is not a small foundation trained from images from
scratch, and no matched-quality efficiency advantage over an LLM is
established. English is untrained transfer in this Chinese training stage;
historical forms and long-form writing are not evaluated here.

More unchanged fitting is not the next intervention. The
[first-image likelihood diagnostic](pixel_language_first_block_diagnosis.md)
has now [completed at all four fixed states](pixel_language_first_block_likelihood_result.md).
Every provided correct development first image is less likely than the saved
emission, despite some transferred paired-question preference. This does not
support a decoding-only repair for the fixed references. The next controlled
check measures the inherited reader itself before changing the learned
readout. The criterion remains readable, correct, complete answers to unseen
image prompts.
