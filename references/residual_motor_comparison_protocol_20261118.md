# Frozen-plan joint motor: complete physical learning comparison

Bind the completed residual target preparation, all original 68,948 source masks,
saved spatial-256 plans/bases and the exact frozen final binary codec. Keep the
64,041-cell / 8,629-form training partition and 4,907-cell / 952-form validation
partition. All physical aliases and source occurrences remain unchanged; the
reused validation split is development. No reserved language confirmation data.

The sole new trained component is the normalized XOR correction motor from
`residual_plan_motor.py`: two 128-wide layers, four heads, 615,314 trainables.
Its 1,770,449-parameter frozen codec makes the complete physical motor 2,385,763
parameters. This excludes a plan prior and contextual language model. Initialize
the new motor with seed 202611214, zero root weights and root bias [0,-2].
Initial greedy correction is empty for every plan; the original base is the
unchanged control. Do not retrain that control or rerun its old fit.

Retain equal-form sampling with order seed 202611196, batch 128, 6,000 updates
and 768,000 retained image draws. Optimize exact conditional image NLL given the
supplied plan, divided by 1,024 and averaged equally over images; do not divide
by varying event counts. Use AdamW (.9,.95), weight decay .01, clipping 1,
250-update warmup to .0002 and cosine decay to .00002. FP32, deterministic
kernels, no TF32, no augmentation, auxiliary loss or checkpoint selection.

Before fitting, differentiate every training image with the original equal-form
source weights, require finite gradients and nonzero condition/root/mask weight
norms, perform one update, measure and reset every trainable tensor exactly.
Keep the frozen codec byte-identical. Replay the actual model/optimizer update
at update three; its extra 128 trial images are separate cost. Initialize the
retained-update counter before the preflight so a failure cannot inherit a
previous arm's counter.

Generate initial and final corrections for every original image in row order.
Only its fixed plan and authenticated saved D(B) enter correction generation;
the target is used afterwards for scoring. Initial generation must equal every
saved base. The public standalone motor accepts only plan bits and computes
D(B) itself. Supplied-image plans are physical motor supervision, not autonomous
language. True XOR correction replay does not count as a generated correction.

Persist every actual image, correction and causal action/depth/row/column prefix,
actual event count and its cached executed-prefix log probability. Check each
trace against the lossless sequence of the emitted correction. Compare every
valid cached probability with parallel teacher-prefix probability, tolerance
5e-5 nat per event; verify each actual greedy action is the cached argmax.
Also retain true remaining image-error tree events separately from generated
correction work. The independent CPU audit recounts all source pixels, actual
actions, events, scores, ledger rows and checkpoint hashes, without refitting.

Keep every original absolute physical rule: final training and validation F1
.98, validation macro-form F1 .98, macro exact rate .80, no source false blanks
or false ink, and validation F1 within .005 of the saved continuous .996423850.
Require .02 gain over the unchanged base .943690071. Thirty-two plan events
plus actual generated correction events must average at most .60 times
97.30248883216336 over the complete source. Include all extra training and
generation cost. Event counts are not bytes, throughput or language efficiency.

Use one GPU0 owner and project mutex. Entry requires 48 GiB available host,
20 GiB free GPU0, 48 GiB disk and low pressure. Runtime reserves 24 GiB host
with 8 GiB GPU allocation / 8 GiB RSS ceilings, two CPU threads, 2,400 seconds
including at most 300 seconds pressure wait. External timeout 2,460 seconds
plus 30 seconds grace. Preserve foreign services, failures and original outputs.
No automatic retry, snapshot selection or unchanged epoch extension.

A pass permits separately binding conditional plan selection on complete
authentic records. It does not admit a full language fit or establish image-
only language learning. A failure ends this correction recipe; preserve its
limited gains and actual residual errors before another proposal. All original
complete-language, stopping, independence, historical-form and cost conditions
remain unchanged.
