# First-image likelihood: a learned distribution failure

The completed fixed-state diagnostic finds that every provided correct
development first image has lower likelihood than the image actually emitted.
An exact likelihood maximizer therefore cannot select any of these reference
images under the current model. A decoding-only change is not a supported
repair for this fixed-font reference task. This does not exclude other
semantically correct renderings that were not scored.

There is nevertheless transferred question conditioning. At update 16,000,
14 of 16 Chinese reference-image events have greater likelihood under their
own question than under the other question about the same passage. All eight
paired assignment contrasts are positive; six pairs favor both owning
questions individually. These relative preferences are not semantic accuracy:
the model still assigns much greater absolute likelihood to its unusable
emissions.

| Update | Chinese references favor owning question / 16 | Positive paired assignments / 8 | Both owning preferences positive / 8 | Mean best-reference minus emitted-image log probability |
| --- | ---: | ---: | ---: | ---: |
| 0 | 7 | 2 | 0 | −2580.523 |
| 4,000 | 12 | 8 | 4 | −407.504 |
| 8,000 | 13 | 8 | 5 | −585.417 |
| 16,000 | 14 | 8 | 6 | −751.244 |

For the final eight English normal cases, the mean log-probability difference
is −941.818. Across all 24 normal development cases, the largest individual
reference-minus-emission difference is −6.051 and the smallest is −1651.678;
all are negative. At every fixed checkpoint there are zero development
reference witnesses above the corresponding saved emission. The improved
training fit from 4,000 to 16,000 updates does not improve this absolute
reference preference.

For a complete 32×64 first image, the measured score is

\[
\ell(I\mid q)=\sum_{j=1}^{256}\log p(u_j\mid h(q),u_{<j}).
\]

The 256 eight-bit patterns cover every pixel. This is a normalized image
likelihood, measured in nats, with no lexical vocabulary. For each question,
the comparison uses the maximum score of an individual allowed reference,
not the log-sum probability of several references. The pre-image continuation
factor cancels between two image candidates under the same question. It is
recorded separately for comparisons across questions.

The paired contrast sums the owning-question margins for the two different
reference events about one passage. Both first-image events are disjoint for
all eight Chinese development pairs. Any additive question-independent event
score and the continuation terms cancel in this contrast. A positive
contrast still does not imply that either reference has useful absolute mass.

The [execution report](evidence/pixel_language_first_block_likelihood/report.json)
covers 112 common cases at four owning states: 64 training cases and 24 normal
plus 24 erased development cases. It makes 112 fresh native PNG prefills,
448 writing-query calls and 896 physical-head forward calls. Every saved
reader fingerprint and initial stop probability matches exactly. All 448
nonempty first images reproduce all 256 original greedy decisions under
full-forward scoring. Gold images enter only as likelihood targets. No new
answer image, optimizer update, lookup answer or font rendering is used.

The [independent scalar audit](evidence/pixel_language_first_block_likelihood/scalar_audit.json)
recomputes reference-event scores and reference-minus-emission differences.
All [four score files and preparation bindings](evidence/pixel_language_first_block_likelihood/)
are retained. The raw prompt-derived query tensors remain local and are
identified by hash. The 299.82-second diagnostic peaks at 17.999 GiB GPU
allocation. Its separate read-only admission records at least 73.35 GiB
available host RAM and zero sampled 10-second memory-pressure averages.
The completed training protocol and its admission thresholds are unchanged.

The evidence localizes the immediate failure to the first conditional image
distribution. It does not establish that the inherited reader answers every
question correctly. The next controlled check should measure that reader on
the same normal and erased image inputs, alongside a source-text baseline,
before choosing an input-reading or visual-readout intervention. Helper text
decoding in that check is a reader baseline, not an ILM answer. Useful,
autonomous image-language generation remains the required outcome.
