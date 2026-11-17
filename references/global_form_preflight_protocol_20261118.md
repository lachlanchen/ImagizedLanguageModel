# Global form supervision: fixed-budget physical preflight

## Question and comparison

The completed occupancy tree trial draws dense damaged forms: pooled F1
.455979811, zero exact images and ink at all four true spaces. Its normalized
physical likelihood and cached probability replay pass numerical checks. Test
one learning change: add temporary whole-form classification on the exact shared
condition used by the physical writer, before any current-target prefix.

Use the completed tree as unchanged control, without re-executing it. Initialize
the candidate from the same fresh seed (202611190) and require exact initial
tensor hash and all 552 initial output masks to match the immutable control.
Do not initialize from either failed trained physical head or select a snapshot.
Keep all earlier successful complete-language cases and failed evaluations.

## Objective and sources

Source-only preparation maps the 68,948 existing training-image cells to 9,581
exact binary-mask classes, including empty. All 512 book, five short-QA and
35 long-QA target positions map exactly. Use their saved actual pre-target
context states, physical within-span indices and supplied span lengths. Preserve
all source bytes and labels. No development answer, reserved DRCD passage,
teacher winner, synthetic condition, generated-image recognizer or font enters.

The physical tree remains the 2,261,074-parameter writer. Add a separate
192-to-192 GELU-to-9,581 classifier on its actual shared condition. Construct
the helper with CPU seed 202611191 without changing the writer's random stream.
For each complete record, minimize the mean of

\[
 -\log q_\theta(Y\mid h,i,L)/1024
 +\mathrm{CE}(r_\eta(C_\theta(h,i,L)),c(Y))/\log(9581).
\]

The coefficient is one and is not tuned. This auxiliary objective is not the
image law's exact likelihood. Image likelihood alone was already a proper
objective; this tests a finite-budget representation-learning intervention.
Classes are visible masks, not meanings or a proposed inference alphabet.

Cycle the three whole records in the original order for 96 AdamW updates
(32 passes each), learning rate .0002, weight decay .01, norm clip one. Accumulate
every complete-record mean in microbatches of eight images before an update.
Use FP32, TF32 off, unchanged greedy generation batches of sixteen and exact
physical topology/probability replay with maximum error .0002 nat. The helper
and class mapping never enter generation; every output pixel comes from q.
Never turn a classifier prediction into a saved image or font lookup.

Before updating, probe all three complete records. Require finite positive
auxiliary gradients in the actual shared condition, physical position and span
length embeddings, with no gradient to prefix/event blocks from the helper
alone. Require complete finite joint gradients and a positive context gradient.
After fitting, reset every physical/helper trial tensor exactly and retain no
trained successor or optimizer. Source weights and caches remain unchanged.

## Prospective decisions

Evaluate all 552 positions initially and finally. Preserve all 1,104 images,
actual generated topology, cached and parallel probabilities, complete source
order, helper predictions and original control bytes. Require all of:

- Pooled physical F1 at least .75 and every case at least .60.
- At least .05 F1 gain over the saved final tree; retain the original .05 gain
  over the saved flat control as well.
- Zero false blanks and zero false-ink images at true empty targets.
- Pooled within-record rotated-minus-correct context NLL at least .5 nat/image.
- Source-form classification accuracy at least .90 overall and .80 in each
  complete case, measured on pre-target conditions only.

These are training-only support thresholds. Correct supplied histories and
lengths, independent glyphs within a span and the inherited actor are material
limitations. Even a complete support pass only justifies separately registering
a larger comparison; it grants no full-language fit or acceptance. High helper
accuracy with bad images rejects this as a sufficient writing repair. If neither
passes, preserve the failure and do not extend unchanged epochs, alter thresholds
or select an intermediate checkpoint. A failure at this budget is not a proof
that all auxiliary supervision is impossible.

Use one serial GPU0 owner under the project mutex. Entry: 48 GiB available host,
20 GiB free GPU, 48 GiB free disk and low memory pressure. Runtime: 24 GiB host,
8 GiB allocated GPU maximum, two CPU threads, 1,800 seconds including at most
300 seconds pressure waiting; external timeout 1,860 seconds plus 30 seconds
termination grace. Preserve failed executions, with no automatic retry. Do not
stop foreign GPU1 or other projects. Whole-model efficiency still includes the
1,465,381,142-parameter actor, inherited pretraining and cache preparation.
No semantic or autonomous-language score is assigned to this physical preflight.
