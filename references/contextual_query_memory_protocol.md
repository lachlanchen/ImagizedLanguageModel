# Does the completed span head use its task memory?

Prospective auxiliary diagnostic, 7 November 2026. The completed contextual
span component copies accurately but generalizes poorly on questions. The
running contextual adaptation comparison remains unchanged. This diagnostic
uses only its predecessor's completed head and saved features on CPU, with
no source-body invocation or optimization.

## Fixed intervention

Use all 64 normal development questions and the 32 previously assessed
training questions in `artifacts/contextual_pixel_spans_v1/result.json`.
These form 32 and 16 distinct-answer paragraph pairs. Authenticate the
preparation, feature manifest, checkpoint and original assessment files.
Every partner reference must be an admissible span in the owning image;
record any overlap between the two sets of reference images.

For each question i with paired question j, evaluate two conditions:

\[
p_i^{\mathrm{own}}=H(V_i,M_i,G_i),\qquad
p_i^{\mathrm{swap}}=H(V_i,M_j,G_i).
\]

Here V is the saved contextual visual field sampled at original ink units,
M is the complete saved numeric task-frame memory, and G is physical
geometry. The source image, units, visual features, geometry, learned length
bias and all head weights remain identical. The sole change is M. Both
answers are generated before reading reference images for scoring.

Reproduce all 96 original selected answer images from the owning CPU path.
A mismatch is a failed replay, not a new language outcome. Save all swapped
outputs, both complete span-log-probability arrays and input identities.
Reference images enter scoring only. No reference, source annotation or
candidate whitelist enters the head or its selection method.

## Measurements and independent check

Report original-reference and partner-reference exact image matches,
selected-image changes, total variation over the same physical span-index
distribution, both-correct paragraph pairs, and marginal log odds of owning
versus partner reference images before and after the swap. Sum all source
occurrences of each admissible complete reference image. Distinguish
overlapping reference sets rather than treating them as exclusive answers.
Report distance between the actual projected/pooled memory vectors too;
different memories alone do not establish useful conditioning.

A NumPy/PIL verifier authenticates every retained file, reconstructs every
emitted image from the declared original source columns, rescans reference
occurrences, recomputes marginal probabilities and total variation, and
checks all group and pair counts. It checks stored neural outputs rather
than independently reimplementing the neural forward pass. Save all 192
head outcomes, with public raw swapped PNGs and links to the existing full
owning-input/reference gallery. Keep large log-probability arrays local.

Use at most four CPU threads, an empty `CUDA_VISIBLE_DEVICES`, and the
existing Python environment. Load only the 2,378,370-parameter head. No
7.531B source body, teacher, tokenizer, OCR or optimizer runs. The retained
full model and its external pretraining do not become a 2.38M language model
because this diagnostic omits its already cached computation.

## Interpretation

The hybrid features are not a native encoding of a new input. V_i may itself
contain question information, and can disagree with M_j. A changed answer
shows that this particular memory branch influences the fixed head. An
unchanged answer does not prove that the full model ignores the question.
Movement toward a partner reference is a module-level intervention result,
not an instruction-following or free-generation benchmark pass. This
diagnostic neither changes registered gates nor selects a new checkpoint.
The complete independent image-language goal remains unachieved.
