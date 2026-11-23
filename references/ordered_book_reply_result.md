# Writing two facts from a pictured passage

A visual writer can compose a correct answer about a book in Chinese, but the result depends strongly on the question and the page. On one new passage the frame writer answered “出版时采用了榆树的树名，初稿写成了柳树。” Asked in the opposite order, it answered “初稿写成了柳树，出版时采用了榆树的树名。” The source described correcting an interview transcript. The underlying reader selected the names; the writer generated the wording around each selection. Across all 48 requests on 12 new pages, the direct writer and the frame writer each earned four complete-statement credits. Local text Qwen3-8B earned 34. One model-based reader reviewed all four arms with their identities hidden and had prior familiarity with some direct outputs. [Original reply](evidence/anchored_book_clauses_20261124/examples/elm_ab.png) · [Reverse order](evidence/anchored_book_clauses_20261124/examples/elm_ba.png) · [Complete-statement scores](evidence/anchored_book_clauses_20261124/reading_result.json)

| Replies with both facts once, in the requested order | Direct writer | Frame writer | Bare selected excerpts |
| --- | ---: | ---: | ---: |
| 96 novel passages, 384 requests | 6 | 34 | 77 |
| 12 authored passages, 48 requests | 9 | 7 | 18 |
| 64 additional novel passages, 256 requests | — | 30 | 48 |

Each passage was requested in Chinese and English, in two orders. On the first 96 novel passages no direct or frame writer passed all four requests on the same page. The direct writer with correct excerpt placement tied the shifted-excerpt control on the novel requests. Both exact facts appeared in the ranked banks on 168 requests; only four of those responses passed. Training on a prefix and suffix around a marked fact improved success in both orders from zero to 15 of 192 novel passage–language pairs. The bare excerpt list earned 37 pairs. Ordering the two independently written clauses is a runtime operation. [Direct and shifted comparisons](evidence/ordered_book_replies_20261124/result.json) · [Paired audit](evidence/ordered_book_replies_20261124/audit.json) · [Predicted, bare and oracle comparisons](evidence/anchored_book_clauses_20261124/result.json)

Providing the correct two excerpts at inference raised the frame writer's novel score to 212 of 384. The remaining losses include malformed insertion frames, budget overruns and incorrect occurrences of the facts in the surrounding wording. A separate automated grader credited 22 direct responses on the 48 new requests, including a wrong assignment of the flute and drum after the two performers exchanged instruments. The closer reading credited four. The original grades and pixels are preserved. [Automated assessment](evidence/ordered_book_replies_20261124/statement_assessment.json) · [All 48 requests, replies and judgments](evidence/anchored_book_clauses_20261124/reviewed_answers.json)

## What the proof establishes

Lean checks that a writer returning identical pixels in both orders cannot earn both strict order credits. If two component checks pass on $A$ and $B$ of $N$ requests, at least $A+B-N$ pass both checks. This applies even to correlated errors. Finally, dropping the generated prefix from a correctly spliced frame recovers the selected source pixels followed by the suffix. The proofs describe an evaluation limit and an implementation invariant. Neither determines whether the reader picked a true fact or the writer stated the correct relationship. [Lean source](../formal/span_list/OrderedVisualReplies.lean)

## Training and inference

The direct fits used the same 671 accepted BiPaR training passages, drawn from 1,024 human-question pairs sent to a local Qwen3-4B for clause preparation. Each fit saw 2,684 pictured presentations, took 135 updates and used 5.23 GiB of allocated GPU memory. The clause writer took 405 updates over three passes, 31.4 seconds of optimization and 4.11 GiB. Each writer contained 87,087,170 parameters, of which 28,599,938 were trainable. Inference also uses the separate pretrained visual reader and its task-specific heads. The runtime receives segmented 32 × 32 cells and uses a fixed physical spelling bank and a source-image pointer. The timings exclude the teacher, previous model training and inherited pretraining. [Teacher preparation](evidence/ordered_book_replies_20261124/teacher.json) · [Ranked fit](evidence/ordered_book_replies_20261124/matched_fit.json) · [Clause fit](evidence/anchored_book_clauses_20261124/fit.json)

From the repository root, run the trained clause writer on CPU with the saved page and question images. Output is a PNG. The program reports whether the structural and length checks passed.

```bash
examples=references/evidence/anchored_book_clauses_20261124/examples
PYTHONPATH=. python -m scripts.write_ordered_pictured_reply --mode frame \
  --page "$examples/elm_page.png" \
  --questions "$examples/elm_first.png" "$examples/elm_second.png" \
  --output /tmp/elm-reply.png
```

Both selected interview replies reproduced the original GPU output pixels on CPU after restoring the 59 changed tensors from the standalone update. [Standalone restoration](../models/anchored_book_writer/README.md) · [CPU replay](evidence/anchored_book_clauses_20261124/cpu_release.json)

The frame checkpoint is in `artifacts/anchored_book_clauses_v1/page/`. The direct writer is in `artifacts/ordered_book_replies_v1/matched/page/`. Both run offline with the shared visual reader and bitmap controls. Raw book-derived teacher material, training rows and evaluation streams are retained locally. The checked-in data contain aggregate counts, question identifiers, authored examples and review receipts. [Preparation and training sequence](ordered_book_reply_protocol.md) · [Clause-writing experiment](anchored_clause_protocol.md)

On a fresh run with the existing BiPaR corpus, the fitted bilingual reader and parent writer installed, the following commands prepare supervision, fit both direct controls, evaluate them and fit the frame writer. The local Qwen teacher is required only during dataset preparation and text-model assessment. Each training command owns GPU 0 exclusively.

```bash
export PYTHONPATH=. CUDA_VISIBLE_DEVICES=0
python -m scripts.prepare_ordered_book_replies
for action in prepare shifted matched evaluate
do
  python -m scripts.train_ordered_book_replies "$action"
done
python -m scripts.assess_ordered_book_responses
python -m scripts.train_anchored_book_clauses
```

The remaining question is whether the writer can distinguish two valid-looking statements that assign the same visible names to different roles. Train on verified contrasts where a single role, time or negation changes the truthful answer. Keep the incoming facts and words balanced across alternatives and measure full freely written propositions on unseen passages. Preserve the two current writers as controls. These experiments establish no general language parity or whole-system compute advantage.
