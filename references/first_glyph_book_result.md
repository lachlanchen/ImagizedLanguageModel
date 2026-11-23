# The first glyph and the complete literary answer

A pictured Chinese question about Lu Xun’s sick father elicited **父亲** (father). On the same pictured passage the English question “Whom was the narrator buying medicine for?” elicited **芦根** (reed root), an ingredient mentioned later. The writer produced both strings as legible glyph images and stopped. Question language changed which fact it returned. [Original pixels and source excerpt](evidence/first_glyph_book_20261124/figure.json) · [Editable figure](../publication/ilm-nature/figures/First_glyph_book.pptx)

## New literary passages

The test comprised 34 passages from four works, two different questions per passage, asked once in each language. The target was a Chinese answer image. All 34 passed physical font coverage, answer separation and the screen for a shared run of 47 or more consecutive pictured glyphs against 16,529 distinct pages in the supervised fit. The questions were composed for this evaluation. Source files, source hashes, passage offsets, questions and accepted answers were reserved before inference. The test has four work clusters and one training seed. [Pages and questions](evidence/first_glyph_book_20261124/reserved_books.json) · [Reservation](evidence/first_glyph_book_20261124/reservation.json)

| Outcome | Ordinary bilingual | Later-action weighting | First-action weighting |
|---|---:|---:|---:|
| Correct first glyph, both languages | 47/136 | 42/136 | 48/136 |
| Correct first glyph followed by a failed reply | 17 | 13 | 17 |
| Exact stopped Chinese-request reply | 23/68 | 24/68 | 24/68 |
| Exact stopped English-request reply | 7/68 | 5/68 | 7/68 |
| All four replies on a passage | 0/34 | 0/34 | 0/34 |
| Exact replies with erased question | 2/68 | 2/68 | 3/68 |
| Complete BiPaR development passage | 3/374 | 3/374 | 3/374 |
| Complete regular-print XQuAD passage | 1/227 | 1/227 | 1/227 |
| Complete regular-print CMRC passage | 16/197 | 21/197 | 16/197 |
| Complete familiar historical headword | 99/100 | 99/100 | 99/100 |
| Exact source-absent open writing | 0/300 | 0/300 | 0/300 |

The first-action writer returned four exact answers across 24 requests for Lu Xun, five of 32 for Faxian, eight of 32 for Xu Xiake and 14 of 48 for *Dream of the Red Chamber*. Compared request by request with ordinary training, it gained five and lost four exact Chinese replies, and gained five and lost five in English. The net gain of one therefore hides 19 changed correctness judgments. The earlier bilingual release remains the selected writer. [Complete scores and selection rule](evidence/first_glyph_book_20261124/result.json) · [Per-question results](evidence/first_glyph_book_20261124/case_scores.json)

## What the derivation tests

Each training example carries a pictured passage, pictured question and a sequence of target glyph images ending in a stop. At each position the model sums probability from the fixed glyph inventory and every matching copy address. The experiment adds one negative log probability per eligible example, averaged over eligible examples in the batch. The candidate emphasizes the first position. A matched control emphasizes one later position, including the stop. If the initial glyph is correct on $F$ requests and the remaining sequence fails on $D$ of those, exact replies total $F-D$. Across the two new fits the gain of six correct starts produced two additional exact answers because four more replies subsequently failed. Lean checks the identity for arbitrary finite samples, a necessary mass condition for a wrong greedy first choice and the implication of a strictly dominating wrong complete reply. The statements concern decoding and scoring. Whether a reader learns the question’s meaning is an empirical question. [Lean source](../formal/span_list/FirstActionRisk.lean) · [Training protocol](first_glyph_book_protocol.md)

Both new fits saw the same 49,283 examples, 538,340 supervised output positions and 2,496 optimizer updates. They updated 28,599,938 of 87,087,170 network parameters. The 176.8-second first-action and 175.4-second later-action fits used a peak 4.50 GiB of allocated GPU memory on one RTX 4090 D. The models inherit text pretraining and an earlier visual interface. [First fit](evidence/first_glyph_book_20261124/first_fit.json) · [Later fit](evidence/first_glyph_book_20261124/later_fit.json)

Text Qwen3-8B Q8 answered 119/136 requests exactly and completed 21/34 passages. The first-action image model gave 31/136 exact answers and completed none. Warm serial serving took 26.38 seconds for text and 5.30 for images, excluding initialization and host preparation. All 136 serial image outputs agreed with the archived batch evaluation. On the separate household illustration, first-action training produced six of eight correct replies and later-action training produced four. The experiments leave general visual reading, open composition and efficiency at matched quality unresolved. [Serving comparison](evidence/first_glyph_book_20261124/serving.json)

## Reproduction

The local checkout contains the verified literary sources under the neighboring ZhJpBook project, the earlier mixed corpus, and the trained inherited parent. From the project root, the following entry points verify the passages, train the matched models in fresh output directories, and score the stopped images. The released bilingual model remains available through the existing PNG interface.

```bash
artifacts/role_binding_v1/tools/lean-4.12.0-linux/bin/lean formal/span_list/FirstActionRisk.lean
PYTHONPATH=. python -m scripts.reserve_first_glyph_books
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python -m scripts.train_first_glyph_book first later
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python -m scripts.evaluate_first_glyph_books
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python -m scripts.benchmark_first_glyph_books
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python -m scripts.audit_book_sequence_probability
```

The ordinary image interface also reproduces both Lu Xun results on CPU using the candidate checkpoint. Input strips and observed replies are included with their hashes. [Interface check](evidence/first_glyph_book_20261124/png_interface.json)

```bash
sample=references/evidence/first_glyph_book_20261124/examples
PYTHONPATH=. python -m scripts.respond_pointing_glyphs \
  --model artifacts/first_glyph_book_v1/first/page \
  --passage "$sample/lu-3_page.png" --question "$sample/answer_hit_question.png" \
  --output /tmp/ilm-father-reply.png --device cpu
```

## Why a broader search alone has limited scope

Replaying the released writer reproduced all 833 archived greedy actions on the new literary questions. Among 104 wrong stopped replies, the emitted string had higher joint probability than every accepted reference on **94**. An accepted string had higher probability on ten. Two other misses reached the length cap. On 82 of the 94, the first glyph was already wrong. The writer placed its best accepted string above the other question’s best reference on 97 of all 136 requests. A reader limited to choosing between the two human answers would therefore conceal errors that emerge in free generation. The next training comparison should teach against the writer’s own high-scoring wrong replies on training pages. [Case-level sequence probabilities](evidence/first_glyph_book_20261124/sequence_probability.json) · [Diagnostic rule](first_glyph_sequence_diagnostic.md)
