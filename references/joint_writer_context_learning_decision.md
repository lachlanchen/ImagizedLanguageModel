# Train conditional image likelihood through the continuation state

The [completed joint writer](joint_writer_physical_result.md) improves selected
conditional readability from 599/1,040 to 970/1,040 forms but fails its fixed
physical prerequisite. Its 124-case [probability diagnosis](evidence/joint_writer_blank_density_result_20261113.json)
is also complete. Across all 12,597 supplied-history positions there are 349
empty outputs whose source images contain ink. Only sixteen prefer their source
image to the empty image by the fixed 0.001-nat margin. For fifteen of the
42 development-book false blanks, the first-nonzero prefix bound supports the
empty image as a global mode of the fitted density. This is numerical evidence,
not a formal floating-point certificate.

Only one of thirteen selected development-book false blanks and none of the
selected training-fixed-QA false blanks prefer the source. The registered
search-only criterion requires at least seven and one, respectively, and fails.
There is no disagreement between the saved empty outputs and the parallel
zero-prefix argmax checks. No source image repaired an output or frozen reading.
This rejects the proposed search-only decision rule; it does not prove that
every nonempty alternative is bad or that search could never change a glyph.

The comparison changes the interpretation of the remaining failure. The new
head can draw many clean forms, but it was trained on contextual states kept
fixed after the earlier marginal-pixel objective. That objective never asked
these states to support the new complete-image likelihood. Higher readability
alone has not qualified their conditional prediction, and an empty mode cannot
always be removed by a more thorough search over the same density.

## One bounded model intervention

Allow the existing continuation residual adapters and initial query to receive
gradients from the complete-image likelihood. Train them with the physical
writer; keep pretrained prompt memory, native body weights, image feedback,
normalization buffers and stop-head parameters frozen. Retain the existing
balanced stop loss and weight 0.1; it now supplies gradients to the continuation
state through the same frozen stop predictor. No lexical output head, live
teacher, glyph bank, OCR or font enters the training forward or answer path.

Let `h_theta(X,Y_<t)` be the continuation state, `phi` the joint image writer,
and `g` the frozen stop predictor. Use

    L = -sum_(t,j) log p_phi(r_tj | h_theta, r_t,<j)/(128 T)
        + 0.1 * [mean_(t<=T) softplus(g(h_t))
                 + softplus(-g(h_(T+1)))]/2.

Every image predictor precedes its own target. The state gradient contains
`dL/dtheta = sum_t (dh_t/dtheta)^T (dL/dh_t)`, rather than fitting only `phi`
on immutable states. This is ordinary conditional maximum likelihood and the
chain rule, not a new physics mechanism or proof of language understanding.
The intended change is learning which complete visible form follows a prompt
and its history, while preserving the coherent image distribution.

Initialize the continuation state from the same final point-model checkpoint
and the head from the **same registered fresh draw** as the completed head-only
trial, with its training-only normalization. Use the same complete corpus,
schedule, exposure, prescribed checkpoints and initial frozen interfaces.
The completed head-only trial is the unchanged control. Do not select an
intermediate or successful output from that failed trial as the new final
model. The new comparison changes gradient reach into the continuation
adapters/query; it does not obtain an extra unreported training prefix.

## Executable proof before a complete fit

Implement exact memory-bounded backpropagation: differentiate image-head
microbatches through a detached state leaf, accumulate its gradient, then apply
that gradient once to the actual continuation graph. This vector-Jacobian
product must match ordinary unsplit backpropagation on the real short complete
QA control. A complete long book must yield finite nonzero continuation/head
gradients within the 22-GiB ceiling. Check frozen-parameter versions and retain
all preflight costs. Preflight optimizer changes must not enter the fresh fit.
These are computational checks, not language demonstrations. The
[state-gradient implementation](../ilm/visual_lm/joint_continuation_learning.py)
has an independent unsplit-autograd comparison using the actual joint head;
its computational test does not replace the required real-source GPU preflight.

Before neural execution, bind source, initial weights, full source manifests,
schedule, gradient checks, resource caps and unchanged controls. Do not launch
an unregistered fit. Train all 13,766 complete sources for 41,298 retained updates
and 6,108,741 target-image exposures, retaining all five snapshots. Audit the
actual optimizer and source ledger. Use the unchanged 124-case physical gate,
including its original whitespace positions and every 90% group threshold.
No unexecuted outcome is assigned a score.

If physical qualification passes, evaluate full autonomous responses with
actual emitted-pixel feedback, paired questions, all erasures, coherent long
books, natural stopping and all earlier required successes. Retain the full
12/32 exact, 4/16 pairs, 24/32 readable and 4/12 coherent-book pilot, followed by
untouched confirmation. A physical failure rejects this intervention before
autonomous evaluation. A clear-forms/language failure redirects work toward
prompt use and sequence generalization rather than more rendering repair.

The complete model still includes the large inherited body. Its actual
parameters, data preparation, wall time, VRAM and later matched-quality cost
must be recounted. Useful modern-Chinese language, a compact independent
foundation, historical/unencoded writing and the complete ILM goal remain
unproved. Preserve the failed trials and update the Nature-style manuscript
and native figures from their measured results.
