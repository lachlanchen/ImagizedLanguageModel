# Full-corpus clean image-language candidate

The paired clean-writing control fits 32 seen training questions, with 29
exact original-image answers out of 32. It establishes learning capacity,
not unseen language competence. This successor tests whether the same clean
architecture can learn from the full admitted Chinese/English instruction
corpus. The candidate has no language-success claim before the original
autonomous development requirements pass. Historical scripts, general
assistant quality and parity with existing LLMs remain unproved even after
a bounded Chinese development pass.

## Fixed data and training

Reuse the authenticated V72 cache of all 12,976 training examples: 3,193 CMRC
and 9,783 Dolly. Its report SHA-256 is
1da155a5476f59f0c6cff1d0e897534f371dba7e8bf7e07188157176204b4aeb;
its recipe SHA-256 is
8156258a7a708556c37db4320d54966adadd5d7192dcb3bd615cd6995e14482e.
Link this cache into the run directory; do not duplicate it or recompute the
expensive inherited image features. Every loaded record is authenticated.
There is no development or sealed-example optimization.

Initialize all 64,520,193 writer parameters afresh. Neither the failed V72
writer nor the fitted 32-question writer initializes this run. Reuse the
frozen qualified writing codec and explicitly attributed externally
pretrained visual reader. The training cache contains only prompt-derived
image memory and continuous codes derived from complete answer images.

Train for 40,000 updates at batch 16, giving 640,000 draws with exact shuffled
corpus passes. Use AdamW, peak learning rate 0.0001, 500-update warmup, cosine
decay to one tenth of peak, betas (0.9,0.95), weight decay 0.01, gradient
clipping 1 and no EMA. Model, sampler and noise seeds are 20267500,
20267501 and 20267502. At most 16 target blocks per example are sampled,
weighted equally per example. Use logit-normal time
t=sigmoid(-0.8+0.8*u), u standard Gaussian, and mean squared clean-code error
divided by max(1-t,0.05)^2, plus balanced learned continuing/terminal loss.
The body uses BF16 and the writing output projection executes in FP32.

A separate real-example execution preflight runs 33 updates and checks
future-target exclusion, exact optimizer/RNG/data-cursor resume and forced
two-block re-encoding of actual emitted pixels. It receives no quality
credit. Reset the model, optimizer and random streams for production. Save
resumable checkpoints every 500 updates; evaluate only update 40,000.

## Fixed inference and decision

Run the original complete 192-question image audit, including all 128 CMRC
Chinese questions and 64 Dolly English questions, with matched question-erased
controls: 384 actual autonomous model calls. Use seed 20267503 plus case
index, shared by normal/control. The actor sees actual prompt pixels,
recomputes the inherited reader memory and receives no answer length, gold
code, text, character IDs, OCR, vocabulary head, glyph inventory or teacher.
Emit physical image blocks with 32 analytic clean-endpoint updates and
re-encode the actual emitted binary pixels into causal history. Stop at
learned probability 0.5, with the unchanged 128-block safety cap.

Keep the original Chinese requirements unchanged: at least 50% exact full
answers, foreground F1 at least 80%, both-correct article pairs at least 35%,
normal-minus-question-erased F1 at least 10 points, a strictly positive lower
95% source-article bootstrap bound, and learned stopping at least 90%.
The bootstrap uses the original 64 articles, 2,000 draws and seed 20266831.
English literal reference agreement is reported separately and is not an
exhaustive semantic score. Preserve every raw answer without pixel repair.

The original independent CPU verifier authenticates all source/cache/output
payloads, reconstructs prompt layouts and independently recomputes scores,
paired outcomes, bootstrap intervals and all six decisions. It does not
replay the entire neural forward pass or every feedback encoding. Integrity
verification cannot turn failed quality into success. Sealed examples remain
unopened in this pipeline; a development pass would require a separate
declared sealed evaluation.

This changes clean prediction, head capacity, numerical projection, time
sampling and initialization relative to V72; the experiment cannot attribute
a difference to only one factor. The architecture/objective direction was
chosen after earlier development failures, so this is a reused development
benchmark, not a fresh confirmatory generalization claim.

## Execution

Commit the source/protocol before execution. Run:

```bash
PYTHONPATH=. artifacts/external/mixar_env_py312/bin/python \
  scripts/run_clean_corpus_v75.py --run artifacts/clean_corpus_v75_20261106
```

The coordinator serializes execution preflight, full training, autonomous
audit and CPU verification on physical GPU 0, holds a run-specific coordinator
lock, records owned PIDs and uses the shared project GPU lock for GPU stages.
It creates no GUI stack and leaves GPU 1 untouched. All phases and checkpoints
remain under the run directory; the cache is a link to the original source.
