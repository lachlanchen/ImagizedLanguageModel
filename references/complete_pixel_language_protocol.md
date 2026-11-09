# Complete natural question-answer image learning on one GPU

The native pixel actor passes actual PNG input, physical output, generated-image
feedback, gradient and optimizer-replay qualification. Its two-update answers
remain incorrect. This experiment now trains the complete causal image-language
path on natural questions, rather than fitting cached first blocks.

Reuse the fixed V80/V81 cohort of 32 Chinese reading questions: 16 passages
with two different questions and distinct answers per passage. Reconstruct
and verify every original owning prompt and complete answer image. Remove only
the two structural blank termination blocks supplied by the original renderer;
retain every actual answer block. The same 107 answer blocks remain the full
training targets. Prepare the first eight existing development article pairs
(16 questions), plus the first question in each of eight English categories,
before any new model call. Retain all complete reference alternatives. English
is an untrained transfer assessment in this stage. These are reused development
cases, not sealed evaluation data. Erased controls keep the same layout while
removing Chinese question ink or the English prompt ink. Neither development
references nor case identifiers enter generation or optimizer updates.

Use the qualified native reader with full prompt KV, scoped suffix adapters,
start query, image-history projection and FP64 physical density with FP32
stored parameters. Restore the exact untrained initial actor from the captured
numerical control, SHA-256
e3949fc68937affb19e0ec22386b577f702dfb9e45b07f10c6d6cca735a160d6.
Discard the two qualification updates. External source pretraining and the
prior physical-head fit remain part of initialization and must be disclosed.
No new model weights, online teachers, tokenizer, OCR pipeline or font-based
answer renderer are used in deployed generation.

Batch four actual questions, padding native KV storage while masking each
owning source's padded keys and preserving its original rotary coordinates.
Teacher history after each answer may be padded, but only valid causal states
and the owning terminal state are supervised. Score all 256 physical pieces of
every complete answer block. Optimize the mean per-question mean-block image
NLL plus mean per-question balanced continue/end softplus loss, each at weight
one. This disclosed weighting is not the unscaled finite-stream joint NLL.
Before optimization, compare the actual four-question batch against independent
complete-answer calls with normwise state tolerance 1e-4 and relative loss
tolerance 1e-4. Require exact tested future-image and future-code-padding
exclusion. Do not assume numerical bit equality between different native FP32
batch shapes.

Use AdamW betas 0.9/0.95, weight decay 0.01, foreach false and global clipping at
one. The physical head peaks at 3e-4; other trainable components peak at 1e-4.
Warm up for 100 updates, then apply the declared cosine schedule over at most
8,000 updates, ending at one tenth of peak. Draw fresh full-cohort permutations
in batches of four. The maximum is 32,000 retained examples, or 1,000 exposures
per question. Save after update one, execute update two, restore and replay it;
require exact model, optimizer, sampler, losses and gradient measurements.
That adds one actual optimizer call and four executed draws without adding
retained learning exposure. Check finite gradients throughout and nonzero
required component gradients on the initial two updates. Zero gradients after
convergence are not by themselves failure.

Generate all 32 complete initial training answers, then all 32 at fixed updates
1,000/2,000/4,000/8,000. The only early-stop rule is 32/32 exact complete training
images with learned stopping at one of those milestones; otherwise use8,000.
Development quality never selects an update. Record matched whole-training NLL
at these milestones. Initial and intermediate training generations may reuse
authenticated native prompt KV; final training and all normal/erased development
generations invoke the real actor on original owning PNG files. Every output
uses its own emitted pixels as history and a learned stop threshold 0.5 with a
uniform 128-block cap. A cap event is failure. No desired answer length is
supplied. Recompute all 32 native prefills after learning and require their
complete KV and initial-state hashes unchanged before claiming the replay pool
remained valid.

Keep source-bound rolling checkpoints, sampler and RNG state. Resume is an
explicit command after verifying the prior model process is absent; never
restart a live job on an observation timeout. Preserve uncommitted history tails
and partial output folders before explicit recovery. Completed output receipts
bind the checkpoint step, prepared inputs, result tensors, PNG and actual
feedback trace. Preserve numerical or data failures without an automatic retry.
A durable optimizer-call ledger retains executed calls and timing even when
explicit recovery discards uncommitted learning history after a checkpoint.

One project-owned GPU0 job has a six-hour deadline and a 22 GiB allocated-memory
cap. Require 24 GiB host RAM available, swap at most 75% and 20 GiB GPU0 free before
loading. Retain only the current and immediately previous rolling checkpoint.
Preserve GPU1 and foreign services, and run no Office/LaTeX model job alongside
training. Keep runtime ownership in the ignored handoff. After process exit,
an independent CPU auditor checks complete output tensors, PNGs, learned stop
events, every source split, permutation/exposure accounting and actual image
feedback through CPU re-encoding at 1e-4 tolerance. It recomputes exact pixels,
foreground F1, paired-question correctness and erasure effects separately from
producer scores. It does not independently replay the full inherited neural
body or optimizer and does not assign semantic correctness from pixel overlap.

At least 24/32 exact training answers and foreground F1 at least 0.98 are bounded
writing-fit requirements, not completion of the ILM goal. All development
outputs and question pairs must be reported regardless of quality. Any useful
language claim additionally requires independently assessed correct content
and readable answers to unseen prompts. Historical forms, arbitrary scripts,
a lightweight foundation trained from scratch and LLM-equivalent quality are
not established by this fitting experiment. Maintain the manuscript with the
actual learning results and unedited example images after completion.
