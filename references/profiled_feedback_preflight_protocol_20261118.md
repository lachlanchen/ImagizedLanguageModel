# Complete-record support for fixed-ridge feature learning

The preceding direct readout fit reaches .271039 actual native-coordinate error
and fails the unchanged .10 gate. Its fixed feature map remains the source for
this new optimizer comparison. Exact-query ambiguity imposes a .016775 floor,
which does not exclude the gate. Preserve every prior result and source.

Implement the [variable-projection direction](profiled_visual_feedback_direction_20261118.md)
with the same 33,941 source/slot rows, native targets, original corpus masses,
raw query cache, LayerNorm, source mean/scale and nonlinear head family. Only
the 4,096-wide expansion's weight and bias are outer variables. The affine and
residual output coefficients are solved; neither the query reader nor context,
codec, writer, grouping or stopping changes. No slot residual is added.

## Numerical law and fixed choices

Feedback features and deployed readout use explicit FP32 arithmetic in both the
fixed-feature control and learned-feature arm. Cached queries remain the original
BF16 values. This is a new common precision baseline, not an attributed feature-
learning gain over the old BF16 run. FP64 streamed moments and Cholesky provide
the native-coordinate output coefficients K. At each trial's initial feature
map, set lambda=1e-4 trace(G_0)/4,481 once and keep that scalar fixed thereafter.
Record native data error, lambda||K||², their sum and actual stored-weight error.

For fixed positive lambda, the exact-arithmetic profiled gradient is the partial
feature gradient at solved K. Accumulate that gradient over *every* native row
of the complete record before changing any parameter; no clipped unique-row
minibatch updates replace it. Use the same native relative-error coefficients
m_i/max(||e_i||²,1e-8). Source mean and scale remain unchanged. CPU qualification
compares the partial gradient with differentiation through the solve, checks
streaming, native affine reconstruction, FP32 execution and exact reset.

The descent proposal changes the joint expansion weight/bias by at most two
percent of their current Euclidean norm. Use -g, initial step
eta=.02 max(||A||,1)/||g||, Armijo constant 1e-4, and at most six trials, halving
eta after each rejection. Each trial resolves K with the **same fixed lambda**.
Accept only F_new <= F_old -1e-4 eta||g||². A failed finite search stops the
execution; it does not authorize a new rate, extra trials or retry.

## Real support before a full-support head comparison

Register this preflight separately. Reproduce the old BF16 source-head error
.271038527 within 1e-6 before changing arithmetic. On each of the prescribed
complete 513/846/1,042-glyph training records, include all 509/740/961 native
positions. Fit the initial FP32 output layer, measure its actual execution,
accumulate the full profiled gradient and require finite nonzero gradients.
On the book record, compare its directional gradient with a centered finite
difference of the re-solved profiled objective, using a parameter displacement
.001 max(||A||,1) in the unit negative-gradient direction. Require relative
disagreement <=.05 with an absolute allowance 1e-8. This finite check concerns
the book only; the independent CPU gradient check covers the algebra.

Require one accepted descent step per complete record and actual FP32 native
error no worse than its initial value plus 1e-4. Preserve both actual and ideal
values. Restore every trial feature and output coefficient exactly. No changed
head is retained by the preflight. Any failure is terminal and preserved; do
not silently crop, replace a source, tune a tolerance or resume.

Only after a passing real preflight may a separate full-support head comparison
be bound: fixed-feature FP32 control versus twelve outer feature steps, all
33,941 rows and original masses, the same finite search and full gradients.
The fixed reader rule stays actual error <=.10, at least .10 improvement over
the control, and actual error <=.10 on all three complete records. Passing
coordinate fitting permits only a separately registered live-image conditional-
writing evaluation. It does not admit language fitting or establish language.

## Scope, ownership and resources

No development/DRCD source, reference answer, OCR, font rendering, new glyph,
teacher forward or language generation enters this execution. Source embeddings
are offline training supervision. Deployed weights contain no lexical table,
catalogue or output vocabulary. Inherited pretraining remains part of provenance;
head-only costs are not whole-model or matched-quality efficiency claims.

Commit and push code, tests, protocol and input bindings before execution. Use
one GPU0 owner under the existing mutex, two CPU threads, 48-GiB host/20-GiB
free GPU/48-GiB disk entry, and the existing low-PSI check. Retain 24 GiB host;
cap allocation at 8 GiB, monitored time at 1,800 seconds and pressure waiting
at 300 seconds. External deadline is 1,860 seconds with thirty-second cleanup
grace. Preserve failed attempts, stop the exact owner and protect foreign
services. The general image-language, historical-form, independence, manuscript
and matched-quality efficiency objective remains active.
