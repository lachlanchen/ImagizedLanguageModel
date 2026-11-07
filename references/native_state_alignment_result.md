# State alignment improves English but fails Chinese reading

The completed 2,560-update learner gives **16/32 exact stopped English QA
answers**, compared with 11/32 for the frozen native source. Chinese remains
**0/32**. Copying improves from 7/32 to 16/32, entirely through English:
English copying is 16/16, Chinese copying 0/16. Four of six declared reader
requirements fail. This is limited English transfer through an auxiliary
Unicode head, not a usable independent image-answer ILM.

| Fixed primary assessment | Frozen native source | Final learner |
| --- | ---: | ---: |
| English copies, exact stopped /16 | 7 | 16 |
| Chinese copies, exact stopped /16 | 0 | 0 |
| English QA, exact stopped /32 | 11 | 16 |
| Both English questions correct /16 passage pairs | 3 | 4 |
| Chinese QA, exact stopped /32 | 0 | 0 |
| Both Chinese questions correct /16 passage pairs | 0 | 0 |
| Chinese pairs with different normal answers /16 | 8 | 5 |

English gains six correct answers and loses one on the same 32 questions.
Fifteen of the learner's sixteen correct English answers become incorrect
when the passage is erased; one remains correct without it. This supports
passage dependence for those fifteen cases, without establishing a general
benchmark improvement. The cohort is small, public and repeatedly inspected;
the local English training set excludes its sixteen articles.

Chinese failures are substantive. The first two references are
`印度西太平洋区` and `印度Pondicherry`; the learner answers `印度` to both.
Two different water-power questions both receive `1878年`. Eleven of sixteen
Chinese passage pairs receive identical answers to their different questions.
All 32 Chinese normal answers stop, so preventing endless repetition alone
does not repair correctness. Some responses contain related fragments; the
zero strict score must not be read as proof that no information is retained.

## Training and actual execution

The [prospective protocol](native_state_alignment_protocol.md) fixes 640
training records and four complete passes. All 2,560 optimizer observations
are finite, every training ID appears four times, and all six bound source
files remain unchanged. The complete adapted model has 2,841,626,336
parameters, of which 10,849,280 learn, plus a frozen 126,892,531-parameter
codec and external pretraining. The actual optimizer invocation takes
3,112.084 seconds and peaks at 6.370 GiB allocated on one RTX 4090 D;
offline cache preparation and model loading are separate costs. This does
not establish efficiency at matched quality against an LLM.

The zero-initialized bridge preserves the native source at initialization.
The additional fresh-source check reproduces four prefix/mask pairs exactly
and all eight complete decision readouts. Its execution occurs after main
optimization, as recorded in the protocol's preflight-scope deviation.
The original controller finishes training and all 160 after-readouts, then
stops at the missing-control guard. The updated controller reuses those
completed stages, executes the missing source check and scores. No optimizer
or assessment is restarted.

The [complete export](evidence/native_state_alignment_result/result.json)
binds 50 files, all 160 autonomous after-responses, the source comparison,
preflight and additional source controls, a compact per-update optimizer
journal and all copying inputs. The full per-parameter gradient journal and
numeric checkpoint remain local with SHA256 bindings. The final checkpoint
SHA256 is `916d172c5e18a0f201c1e94e5ba0e42ad74f5c615114eb60781dcdd38f90b98f`.
Original QA pixels and references remain bound to the preceding public
Chinese/English exports. No teacher states, transcripts, correct history or
OCR enter autonomous image assessment.

## Why the Chinese outputs remain unreliable

The separate, prospectively fixed [seen-prompt diagnosis](native_state_fit_result.md)
finds 0/8 exact Chinese copies and 0/8 strict Chinese QA answers even on
training records. Every failed Chinese copying response visibly changes
content; formatting alone cannot explain those errors. English seen copying
is 8/8 and seen QA is 6/8 against the original annotation alternatives.

Chinese answer CE falls only from 4.237 to 3.580 across the first and fourth
passes; Chinese copying CE falls from 4.903 to 3.202. English QA CE falls
from 0.775 to 0.401 and copying from 0.071 to 0.009. These assisted training
losses are diagnostic observations, not autonomous quality measures.
The results identify a Chinese fit obstruction in this recipe; they do not
uniquely identify its cause among the source representation, fixed pooling,
adaptation capacity, objective or optimization budget.

The source's text control also makes Chinese content and formatting errors.
Matching its reading states is therefore supervision from an imperfect helper,
not a guarantee of a correct language state. Codec readability, finite
gradients and lower loss cannot substitute for prompt-conditioned answers.

This recipe does not admit image-writing adaptation or scale-up under its
declared requirements. The immediate prerequisite is reliable Chinese
new-string identity and question-dependent answers. Training whole books
with this unchanged recipe is not justified as a demonstrated repair.
The prepared swapped-fact cohort remains unexecuted; it does not replace
these failed primary results. Direct continuous image answers, historical
forms and general lightweight language learning remain unachieved.

Reproduction uses [the execution guide](../docs/native-state-alignment.md).
CMRC 2018 and SQuAD 1.1 passages/annotations retain CC-BY-SA-4.0 attribution
to Cui et al. (2019) and Rajpurkar et al. (2016). Show-o2 model/source are
NUS Show Lab, Apache-2.0.
