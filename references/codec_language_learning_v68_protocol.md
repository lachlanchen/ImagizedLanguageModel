# V68: complete causal instruction learning in continuous writing space

Registered 5 November 2026 before training. This stage trains the actual
image-language actor to predict answers from prompt pixels and completed
answer images. Reconstruction from supplied answer features cannot qualify
this stage. The completed V64–V67 failures and component results stay frozen.

## Prerequisites and model

Require all 24 checks of the complete codec-actor preflight, pinned report
`960a79cd02eab2b9f084a926dc137bef275356cec3ae9df26a567acc33d7f0ba`,
and its source-bound receipt. Verify every accepted actor source against that
preflight. Preserve the actual 27-layer visual reader and 36-layer contextual
body, FP32 computation with inherited contextual BF16 storage, absent language
vocabulary, rank-eight query/value/output adapters, two writing queries,
shared 4096-to-768 code projection and stop classifier. Only the existing
9,351,937 trainable parameters update. Keep the qualified V34 codec frozen.

Initialize freshly from the same authenticated inherited archive and original
V34 EMA, with model/global seed 20266820. Do not use failed V66 writer weights
or the two-update preflight checkpoint. External foundation pretraining is
inherited and must not be counted as local training. No live teacher is used.

## Data and loss

Use all 12,976 existing Chinese CMRC / English Dolly training examples. Pin
training report `0ffcfd101ea26d79c4084bfd166bd146ff5560f5fa83b7aa57c6b48558377d7e`.
Use a fresh seeded random permutation of the complete admitted corpus each
epoch, without replacement; CPU data seed 20266821. Accumulate two complete
examples per optimizer update, each weighted one half. This consumes 20,000
examples across 10,000 updates, including every admitted source at least once.
Journal exact source indices and names, epoch/cursor and selected loss blocks.
Do not open development or sealed targets in the training process.

Retain every prompt pixel and every actual answer block; discard only the
cache's two explicit white termination blocks. Independently encode each
image so future pixels cannot leak through the reader. Teacher-forced causal
queries predict normalized continuous output-codec labels. Uniformly sample
up to 16 actual block losses per example using CUDA generator seed 20266822,
while retaining complete history. Use the already implemented unit-weight sum
of code MSE, decoded-image MSE and balanced structural stop loss. Labels are
detached, and gradients pass through the frozen decoder into the contextual
adapters. Do not add glyph IDs, OCR, font lookup or teacher calls to the actor.

## Optimization, replay and resource limits

Fix 10,000 optimizer updates, fused AdamW (1e-4 peak learning rate, betas
0.9/0.95, decay 0.01), linear warmup over 500 updates and cosine decay to
one tenth of peak at the final update. Clip global gradient norm at one.
Use the accepted deterministic cuBLAS configuration, no TF32, FP32 attention
and nonreentrant contextual checkpointing. Use the final raw trainable state;
there is no EMA, checkpoint selection or development-based early stopping.

Persist the second update, replay the third from that durable state and demand
exact parameters, optimizer, data permutation/cursor, both generators, global
RNG and update/sample metrics. Retain one immediately previous checkpoint.
Checkpoint every 250 updates and on handled interruption. Resume only with
exact source, data, prerequisite and configuration identity; preserve any
uncommitted journal tail rather than silently replacing evidence. Final
adapter-state storage does not by itself constitute a standalone deployment.

Use only physical GPU 0, with no second owned model job. Require available
host RAM at least 24 GiB, swap below 75% and allocated GPU memory below 22 GiB.
On a resource or 24-hour per-process execution ceiling, checkpoint the actual
completed update and mark interruption, without claiming completion. Do not
stop GPU 1 or another project's service. Preserve errors and the same running
process when observations time out.

## Fixed subsequent autonomous development audit

After the final training state is fixed, prepare a source-bound audit before
opening any generated development output. It uses the existing matched
complete-line image interface and excludes all training prompts. CMRC
development has been used before, so this is reused development, not a sealed
final test. Do not use the codec reconstruction cases as language questions.

Choose 64 different CMRC articles with at least one admitted two-question,
distinct-reference pair, from the original 393-pair file. Sort articles, use
NumPy seed 20266830 to choose 64 without replacement, and choose the first
lexicographically sorted eligible pair within each selected article. Retain
both questions: 128 primary Chinese prompts. Use the original full rendered
references, admitting alternative references only if the same fixed renderer
can preserve them completely. Log every exclusion before generation.

For English, choose eight admitted Dolly development instructions per each
of its eight categories, from sorted IDs using the same generator after CMRC
selection: 64 prompts. Report exact recorded-reference agreement and raw
images descriptively; multiple valid open responses prevent treating this
number as a complete English semantic score.

Generate using prompt pixels alone, with the fixed maximum capacity of 128
blocks and learned stopping. Never supply target codes, target length, source
text or reference images to generation. Compare Chinese normal prompts with
the identical image geometry after question pixels are blanked, preserving
context; prepare the masks offline against the original renderer. Both normal
questions of each pair share context but require different answers. English
uses a same-shape all-blank prompt as a descriptive dependence control.
No actor-side OCR, inventory or post-generation repair is allowed.

Predeclare the Chinese development requirements: normal full-image exact
reference agreement at least 50%; foreground F1 at least 80%; both-correct
question pairs at least 35%; normal-minus-question-blank F1 at least 10
percentage points, with a strictly positive 95% article-cluster bootstrap
lower bound (2,000 draws, seed 20266831); stopped rather than capped output
on at least 90% of normal prompts. Report every gate separately and preserve
failures. These are new prospective requirements, not changes to an old test.
An exact reference match also requires correct learned stopping and full
physical width, including ordinary rectangular white padding.

Even a pass is bounded Chinese development evidence. Useful general language,
English semantic evaluation, unseen writing forms and independent standalone
inference still require appropriate subsequent tests and export. The final
unscored split remains unopened until a candidate qualifies development.
