# Real image-copy contexts for broader readout coverage

The verified date-capable model is unchanged. Its target coverage is sparse:
1,074positive glyph forms across508questions,266initial forms. Synthetic
class-forced hidden-state distillation failed, as did a contrastive loss-only
variant. Decoder-only and dense/spatial flow motors fail hard-mixture gates.
The spatial motor does generate sharp known forms under confident conditions
(meanF10.998260), but that does not recover identity from an uncertain G vector.

This experiment returns to genuine visual contexts. Keep the architecture,
source body,E,D,affine buffers and stop predictor fixed. Broaden only G using
real reader states from copy instructions covering the original glyph inventory,
and retain the508-question gold-image histories. No synthetic hidden-state
projection, native output head, source-unit labels or external teacher supplies
these new hidden states. The source body itself retains its external pretraining.

## Image preparation

Use all observable canonical forms in the existing4,087-form training-only
primitive; exclude and record empty images because a standalone empty cell
cannot identify a space character. Paste each exact32x32binary glyph, enlarged
threefold without retyping, inside a fixed black frame on512x256white paper.
Instructions use the same authenticated Noto Sans CJK SC font at26pixels.
Two fixed instructions are training contexts: Chinese copy-only wording and
English copy-only wording. A third Chinese wording supplies256evenly spaced
form indices, fixed before any neural call, held out from this new fit. Those
identities are seen in training; this tests wording transfer, not unseen glyphs.
Neither existing development questions nor their references choose classes.

Inspect the16-form input preview and all three full-size instruction previews
before caching. The reader receives only each PNG through the same declared
native-budget observer and numeric task frame. The target class index is an
offline training label and is never supplied separately to the reader.

## Actual reader cache

Reuse the authenticated original image reader and its unchanged initial
assistant state. Prefill equal-size images in batches of four, combining
embeddings, four-axis positions, visual masks and deepstack features in batch
order. Use no KV cache or target feedback: each new example trains only its
first glyph. On the first batch, independently prefill each image alone and
require max state error<0.002before continuing. Retain all states by hash,
all original PNG hashes, observer geometry and measured batch error. No
batching equivalence or source-head answer quality is assumed in advance.
Cache all training and held-template inputs, but keep held states out of fit.

## Readout fitting and qualification

Initialize from the preserved complete-glyph language checkpoint. Train G alone
for18,000batch128 AdamW updates, lr2e-4, weight decay0.01, clip1 and the prior
cosine decay. Seed202611111. Half each batch comes from new training-copy
states, half from the original supported gold-image next-glyph states. Terminal
states are excluded from this G-only fit; stop weights remain exact. Use the
original MSE +0.2cosine discrepancy +balanced pixel BCE through frozen D.
The added examples and updates are extra computation, not a matched-budget
comparison or acquisition of a foundation model's knowledge.

Compare the original and candidate on every training-copy state, every held
wording state, and every original nonterminal history. Retain all raw pixels
and per-example physical F1/exactness. Check all three fixed64-triplet boards.
Require training-copy meanF1>=0.99; held meanF1>=0.95 with80%of cases>=0.95;
held first-state non-stop fraction>=0.95; and original-history meanF1 loss
no worse than0.01relative to the preserved baseline. The inherited checkpoint
already saw the original histories, so that route measures retention.

A pass qualifies only this head/wording transfer. It does not establish
complete copying, natural language understanding or correct stopping after a
glyph. Only after passing and reviewing actual outputs run the unchanged
forty-stream autonomous assessment: eight training controls, sixteen normal
Chinese questions and sixteen erased-question controls;64-glyphcap and learned
stop, actual emitted-pixel feedback, no runtime glyph/answer bank or reference.
Review complete raw images before reference scoring. These development cases
are reused, not a sealed semantic benchmark. Preserve all failures and the
original changed-date model regardless of the result.

## Resources and provenance

Each NN stage uses only physical GPU0 and the existing global project mutex.
Cache uses the unchanged original diagnostic admission,22GiBallocation cap,
24GiBruntime reserve and at most300spressure waiting; its declared wall cap
is14,400s for the larger corpus. Fitting/qualification uses the separate small
head48GiBentry/32GiBruntime,8GiBallocation and1,800scaps. Assessment keeps the
original1,200scap/admission. Do not change any completed source binding or kill
foreign services. Record resources, PID/start identity, statuses and timings;
preserve a failed batch/cache instead of silently retrying with another recipe.
No general success, Unicode independence on unseen forms, novelty priority or
LLM-efficiency advantage is claimed from preparing or fitting this curriculum.
