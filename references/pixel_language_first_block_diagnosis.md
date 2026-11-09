# First-image mismatch precedes complete-image feedback

Exploratory analysis of the saved initial and 4,000-update assessments finds
that all 24 normal development responses already differ from every allowed
reference in their first 32-by-64 image block. None is an empty response.
This mismatch occurs before a previous complete generated image is encoded and
fed back. Accumulated inter-block feedback drift therefore cannot explain
every observed physical mismatch. Autoregression over the 256 physical pieces inside
that first image remains a possible source of error.

| Group | Cases | First block exact, initial | First block exact, update 4,000 | Complete image exact, update 4,000 |
| --- | ---: | ---: | ---: | ---: |
| Assessed training questions | 64 | 0 | 7 | 3 |
| Chinese normal development | 16 | 0 | 0 | 0 |
| English normal development | 8 | 0 | 0 | 0 |

Of the seven training responses whose first block is exact, four fail later
or stop at a different length. Both initial and later-stage errors matter.
Pixel equality is a physical criterion, not a semantic grade. The
[complete 4,000-update review](pixel_language_diversity_interim04000_result.md)
also records broken writing and two readable but factually wrong dates.

Erasing the question changes the first block and complete stream in every
one of the 24 development cases, both initially and after learning. All 32
assessed training pairs and eight Chinese development pairs also produce
different first blocks for their two questions. Their reference first-block
sets do not overlap. Output variation alone cannot establish learned meaning:
it is already present in the untrained writing interface.

The earlier [imported-reader check](native_visual_transfer_v71_result.md)
contains the first eight Chinese development questions. The current original
prompt pixels match all eight archived inputs exactly. Applying the current
observer resize reproduces all eight archived resized pixel arrays exactly
as well. Their fixed control-vector digest matches, and all four shared
loader/interface source files bound by that earlier preparation remain
unchanged. Different PNG file encodings do not change these pixel arrays.
The earlier diagnostic text readout answers four of those questions exactly,
including `288条`; its literal answers are retained in the
[new analysis record](evidence/pixel_language_prefix_diagnosis/report.json).

This is bounded evidence that removing the pretrained text head does not
automatically transfer its answer-writing capability to a new image head.
It does not independently replay the complete current reader or establish
that every current failure belongs to the writer. The two date questions
are outside that eight-case imported-reader diagnostic. A separate
[native source check](native_visual_baseline_v70_result.md) answers both
dates correctly, but uses a different runtime and output interface.

## Distinguish conditional learning from local greedy decoding

The next model-based diagnostic should start with the first image, where
there is no previous complete generated image to confound the comparison.
For a prompt image $q$, let $h_\theta(q)$ be the actual writing condition and
$u(r)=(u_1,\ldots,u_{256})$ the lossless eight-bit physical packing of a first
answer image $r$. Its normalized conditional log likelihood is

$$
\ell_\theta(r\mid q)=\sum_{j=1}^{256}
\log p_\theta\!\left(u_j\mid h_\theta(q),u_{<j}\right).
$$

For two questions $q_i,q_k$ about the same passage, compare the likelihood
of the same reference image under the owning and other question. Deduplicate
identical first-block alternatives before aggregating their probabilities.
This tests conditional preference under identical physical targets; it is
not autonomous answer accuracy.

Also score the actual saved greedy first block $g_i$ and every allowed
reference first block. Define

$$
\Delta_{\mathrm{decode},i}=
\max_{r\in\mathcal R_i}\ell_\theta(r\mid q_i)
-\ell_\theta(g_i\mid q_i).
$$

A positive value supplies a concrete witness that the greedy image is not
the most probable complete image under this head: an evaluated reference
has higher joint probability. Greedy choice at each physical piece does not
guarantee a globally most probable image. A probability sum over multiple
references cannot replace the maximum for this particular witness. Even a
positive result would not prove that the most probable image is correct or
that a small beam search can find it.

The comparison separates useful next actions. Weak owning-question preference
would motivate improving the conditional learning interface. A higher-scoring
reference than the emitted greedy image would justify testing decoding over
physical patterns before replacing the architecture. Neither outcome is
known yet. If search is tested, it must generate without consulting reference
images; evaluation references cannot become output candidates.

Model-based scoring must wait until the sole registered training process has
exited. Reconstruct an authenticated assessed state, use the original prompt
PNGs, and first check its saved initial stop probability and all 256 greedy
physical decisions against the recorded output. Report cold-replay failures
before interpreting likelihood differences. The pre-image continuation
probability should be reported separately: it cancels between two image
candidates under the same condition, but not between different questions.
Stopping after an emitted image is a separate, image-dependent decision and
is not included in this first-block head likelihood.

The current CPU analysis makes zero neural forward calls, optimizer updates
or new output images. It consumes the earlier complete tensor/PNG audits,
checks the original PNG receipts and reference hashes, and compares every
registered saved response. The
[reproducible implementation](../scripts/analyze_pixel_language_prefixes.py)
does not modify the running experiment:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. python scripts/analyze_pixel_language_prefixes.py \
  --run artifacts/pixel_language_diversity_training_v2 \
  --review artifacts/pixel_language_diversity_initial_review_v1/audit.json \
  --review artifacts/pixel_language_diversity_review04000_v1/audit.json \
  --earlier-reader artifacts/native_visual_transfer_v71_20261105/attempt01 \
  --output artifacts/pixel_language_prefix_diagnosis_recheck
```

Use a new output directory. This diagnosis does not qualify the ILM, establish
LLM-equivalent efficiency or replace the final registered assessment.

## First-block likelihood runner

[`scripts/score_pixel_language_first_blocks.py`](../scripts/score_pixel_language_first_blocks.py)
implements the subsequent scoring test. CPU preparation has authenticated the
initial and 4,000-update states, 112 source-fixed common streams, 88 distinct
reference first images and 40 same-passage question pairs. The saved reader
fingerprints agree between those two states for every prepared input. An
actual invocation of its neural command against the live trainer refuses
before importing PyTorch and creates no scoring execution directory. The
[preparation receipt](evidence/pixel_first_block_likelihood_preparation.json)
records these checks. **The neural path has not yet been executed or validated.**

Preparation binds source files, owning learned states, original prompt PNGs,
saved output receipts and all selected references. No frozen model weights
are copied. Neural execution also requires the bound source files to match
committed code. The implementation uses the existing GPU0 lock and RAM,
swap, free-VRAM and 22 GiB allocation limits. It checks both current status
and the last recorded process identity, including the interval after the
trainer clears its status PID but before that process exits.

The model stage first captures prompt-only queries and reproduces each
saved reader fingerprint and initial stop probability. It retains the
queries, releases the large contextual body, then scores first images through
the small normalized density. Every nonempty saved first image must pass a
full-forward, saved-prefix replay of all 256 greedy argmax decisions. No
reference image enters prompt reading or query capture. Reference pixels
enter only teacher-forced likelihood scoring, and no new answer image is
generated or selected from references.

The output records individual reference probabilities, their deduplicated
event probability, the maximum individual-reference-minus-greedy gap, owning
versus other-question margins, paired assignments and question-erasure effects.
Raw scores remain available alongside summaries. A 1e-8 comparison tolerance
labels numerical ties; it is not a semantic-accuracy threshold or a proved
uniform error bound. The common 64 training examples support comparisons
between states without replacing the full final 512-question evaluation.

Prepare a new diagnostic directory on CPU:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. python scripts/score_pixel_language_first_blocks.py \
  --stage prepare --updates 0 4000 \
  --output artifacts/pixel_first_block_likelihood_new
```

Run neural scoring only after the original trainer process exits:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONPATH=. \
python scripts/score_pixel_language_first_blocks.py --stage run \
  --output artifacts/pixel_first_block_likelihood_new
```

The command refuses to overlap a live training owner. It does not queue or
restart that owner. To compare all four registered states after completion,
create a fresh preparation with `--updates 0 4000 8000 16000`; `--updates`
applies only to preparation. Preserve any earlier or interrupted execution.
Successful numerical replay or a positive likelihood margin would still
leave useful autonomous language generation to be demonstrated.
