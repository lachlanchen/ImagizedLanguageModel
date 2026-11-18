# Imagized Language Model: Engineering Goal

Updated: 2026-11-18. **Active; usable visual language remains unproved.**

## Objective and deployment boundary

Build a useful, independently running language model that reads visible writing,
understands the supplied prompt and generates readable answer images on one
RTX 4090. Its causal history must include its own emitted pixels, and it must
learn when to stop. Begin with modern-Chinese questions, passages and book
continuation; this development stage does not complete the full objective.

The full target includes simplified and traditional Chinese, historical forms,
handwriting and writing absent from Unicode. The eventual word-origin application
accepts Chinese or English questions and produces explanatory writing with
attested historical forms and provenance. Generalize the visual representation
to other writing systems, including cuneiform, after useful modern-Chinese
language is established. Raw pages, three-dimensional glyph streams, movies and
audio remain later extensions of this language objective.

Typed prompts may be rendered into images at the input boundary. Offline
teachers, OCR, text corpora and inherited foundation weights are allowed for
preparation and training; disclose their contribution and cost. Deployed answers
must come from the trained model: no live teacher, tokenizer, lexical output
head, answer lookup, OCR or font renderer in answer generation. Optional
transcription may read a finished image without changing its pixels. Transferring
inherited language capability and learning language from images alone are
separate claims.

## Independent generation of absent Han characters

The [candidate-free idiom trial](../references/pictured_idiom_result.md) provides a [standalone 2.12-million-parameter model](../models/pictured_idiom_zh/README.md) that reads pictured prose and writes the missing expression. It exactly completed 134 of 22,907 official validation gaps, including 100 in which all four target glyphs were absent from the input window. Whole-answer alignment raised the original 35 exact answers to 134. Exchanging the contexts gave 23. The local count baseline gave 1,012. Of the 930 passages solved when the reader could choose among pictured training expressions, the independent writer solved 79. It solved none of 38 whole-expression novelties and both members in zero of 8,900 matched-context pairs. Figure 40 shows a correct answer and a failure. Extra sampled proposals failed the registered improvement criterion. Repeating that search is unwarranted.

The [matched reader-transfer comparison](../references/visual_writer_transfer_result.md) is complete. The independent writer produced 378 exact development expressions, up from 134 originally and 292 with equal extra training on the source targets. It recovered 190 of the 851 previously lost reader successes. On 9,135 newly held-out passages the corresponding counts were 163, 60 and 125. Eight nearest input glyphs alone gave 126, outer glyphs alone seven and the local count baseline 406. None of 38 development or 18 fresh cases requiring an unseen whole expression was solved. Figure 41 shows a new success and a remaining error. The runnable physical writer uses a 1.1 MiB update and the original base weights, with no teacher or expression inventory at inference.

The [bilingual source-plate comparison](../references/historical_plate_result.md) now qualifies an integrated reader. It selected and stopped on the exact requested historical images in 9,565 of 9,600 questions under each of three frames, gave the same score after relocation, and retained every original modern stream. The 249 development families are disjoint from training. All 35 errors involved 文 as both a competing headword and part of a stage name. The public executable accepts typed or pictured questions and a supplied four-row plate. Figure 42 reports the physical outputs, including an error.

The [local teaching and whole-answer comparison](../references/expression_preference_result.md) is complete. Local hints did not reliably improve on the guided writer. A preference loss improved supplied pair choice but reduced independent completion relative to jointly training the reader and writer on the observed answer. That joint reference fit is independently confirmed and released. Across 17,555 previously unused gaps it produces 433 exact stopped answers, versus 276 for the guided writer and 789 for local counts. There are 309 all-four-absent successes, 4/6,646 both-correct pairs and 0/21 exact novel expressions. All 70,220 emitted cells match known glyph bitmaps. The historical path preserves every intact and relocated stream. The fitted runtime remains independent of a teacher, answer inventory, tokenizer and output font.

The released writer's ten most frequent outputs accounted for 54.6% of 17,555 answers. A [matched comparison of passage diversity](../references/passage_diversity_result.md) is now complete. The two fits each saw 571,512 examples. On 17,862 fresh gaps the starting writer, repetition and distinct passages recovered 410, 516 and 550. The 95% interval for diverse versus repeated data touches zero and its top-ten share is higher. On 17,962 further gaps inverse-square-root frequency weighting recovered two rare cases but reduced complete answers from 521 to 427 and increased the top-ten share to 52.5%. Both comparisons failed release. Keep all final weights and original streams. The independent joint-reference checkpoint remains the deployed model. A local baseline returned 974 exact answers on the last holdout, including 739 the uniform writer missed. The writer alone solved 286.

The next model comparison should give the reader an explicit learned transformation of the eight ordered pictured cells nearest the gap, initialized as a zero residual into the existing contextual state. On the 194 fresh cases whose eight adjacent glyphs had appeared together in training, the uniform writer solved 16 and local counts solved 144. Preserve the whole-passage reader, fixed physical renderer, actual pixel feedback, stopping and independent deployment. Match the added compute and parameters with a control that breaks passage-to-local-feature alignment. Measure all new source groups, cases with and without a repeated local window, rare and novel expressions, the 286 complementary kinds of success, output concentration, pairwise correctness, historical regression if qualified, and total costs. No training index or answer list belongs in deployment. Improving repeated phrases alone cannot establish general Chinese language ability or a matched-cost advantage.

## Learning speed on visible Chinese passages

The [512-article successor comparison](../references/visual_successor_result.md) measures how often a reader assigns different glyphs to contexts sharing their final cell. A compact visual model consumed 1,047,284 training targets in 19.46 seconds on two CPU threads. The matched character-indexed model took 18.73 seconds. They scored 55.1% and 56.2% on supplied-candidate assignment. The visual advantage over rotated supervision remained uncertain. The native writer preferred the same candidate for both contexts in 509/512 articles. Preserve these results when evaluating a change to the writing objective. Both individual choices and independently generated answers matter to the deployment goal. General-language efficiency at matched quality remains open.

The [unrestricted drawing comparison](../references/context_contrast_drawing_result.md) tests the same writer without supplied successor images. Subtracting last-glyph density increased the average change between two contexts from 7.08 to 60.18 pixels, with zero exact successors across 1,024 drawings in every arm. Both new complete answers about 言 remained unreadable. Retire this decoding intervention.

The [physical patch writer](../references/patch_count_writing_result.md) now provides a trained generative reference. Across 334 further articles, ordered supervision draws 18/668 exact successors spanning 14 characters, versus zero after rotation of training successors. The targets are absent from both input histories. Ordered writing still produces 348 blanks and no article pair with both independent successors correct. All four ordered free completions reach 128 cells and yield no coherent continuation or etymology. Preserve the successful images alongside the blanks and repeated outputs. This supports learning short-range succession from ink and leaves useful independent language and the matched-cost goal open. A [complete-bitmap probability recount](../references/evidence/patch_count_writing_20261118/blank_probability.json) finds that the learned distribution favors the blank over the recorded successor on 329 of its 348 blank outputs. A change to the learned distribution is needed to make those specific successors win.

The [shared-layout writer](../references/shared_layout_writing_result.md) drew 649/668 known nonempty glyphs and one blank on the same development contexts. It still drew only 18/668 exact successors. The full 言 answer fell into a repeated oval. Enumerating every layout shows that any planner using this fixed greedy writer could reproduce at most 242/668 recorded successors. A language model using only the last two coarse images also loses distinctions between questions with the same ending. Preserve this measured physical improvement; evaluate longer pictured context and distinguishable fine detail for the next model. The [question-state bound](../references/visual_question_information_bound.md) proves that on the 1,788 formatted passage questions any writer retaining only the last two visual signatures can match at most six complete reference answers. Stripping the common answer cue raises the ceiling to 184. Preserve ordered prompt memory and a fine-detail channel in subsequent comparisons. The evidence does not qualify autonomous language or a 1,000-fold saving.

The [grounded two-typeface trial](../references/grounded_detail_writing_result.md) couples the trained full-question reader with a local ink writer. On 3,170 official development questions it selected 565 exact source spans and wrote 124 complete correct Serif answers from Sans inputs. It scored nine exact on 3,158 exchanged questions, and seven both-correct pairs out of 3,100. Answer content and length still depend on selecting a visible span. A 10,833-parameter neural fallback raised withheld shape F1 from .660 to .751. Its advantage over copying (.0271) fell below the required .03 and yielded no additional exact answers. Keep this as a grounded writing reference and preserve the full free-writing, stopping, historical transfer and matched-quality cost criteria.

The [pictured title experiment](../references/visual_title_transducer_protocol.md) tests independent selection, composition and stopping. Of 834 eligible development titles, 46 require joining visible glyphs and 64 require drawing a glyph absent from the full prompt. The [completed title comparison](../references/visual_title_result.md) gives 629/834 correct, all extractive, versus 630 with a contiguous restriction and 580 with a learned opening rule. With genuine answer history, the ink head draws 85/143 absent targets exactly, all punctuation or spaces. Treat the title fit as terminal. Preserve the earlier model and test independent Han drawing and semantic composition as distinct requirements. Both native free-writing arms have now failed [complete blind review](../references/native_continuous_blind_result.md).

## Current model decision: full-context recipe rejected

The [complete full-context comparison](../references/trainable_visual_context_result_20261117.md)
is terminal and fails. Both arms complete all 13,785 records and 8,172 updates;
all 792 prescribed streams, 180 native boards and 144 normal-book visible
prefixes have actual review receipts. All readings were frozen before reference
scoring. Final learned and frozen context each give **0/32 strict complete QA,
0/16 both-correct pairs, 0/32 readable complete QA and 0/12 credited coherent
complete long-book continuations**. Every prescribed snapshot has zero strict,
paired and readable normal QA. Do not choose an intermediate checkpoint.

Treatment's twelve normal long books stop naturally after 209–415 emitted glyph
positions; the control stops on two blank books and caps on ten. This establishes
a stopping difference without usable complete writing. Overall review retains
741 uncertain streams and 51 blank streams. Unreadability prevents credit and
assessment of latent semantic correctness; it is not evidence of a particular
wrong semantic answer. One model reviewer had prior development familiarity;
there was no independent human adjudication. Never repair frozen readings from
references, source weights, OCR, fonts or answer images.

Both original CPU fit audits and both separate strict cold replays pass. Each
cold replay checks the same one fixed final book input, outside the 792 language
streams, with exact pixels, stopping and feedback and no undeclared artifact,
font/tokenizer/corpus or network access. Neither execution nor natural stopping
qualifies language. Preserve the earlier failed admission and cold executions.
The original immutable controller status predates the separate completed review;
current authority belongs in the private handoff.

The whole deployed model has 1,465,381,142 parameters, including 1,409,410,048
inherited contextual parameters. Treatment fit takes 60,127.268 monitored seconds
with 11.647 GiB peak allocation; control takes 4,309.029 seconds with 7.200 GiB.
These fit scopes exclude inherited pretraining, earlier teacher preparation,
all generation and failed executions. Both arms fail useful quality; no
matched-quality efficiency or language-from-images-alone claim follows.

Use the [public hash-bound evidence](../references/evidence/trainable_visual_context_result_20261117.json)
and [failure decision](../references/trainable_visual_context_failure_decision_20261117.md).
Do not restart completed training, generation, audits, cold checks or visual
review; do not relax thresholds or repeat unchanged earlier reader, codec,
writer, native-history or sampling diagnoses. All old failures and successful
cases remain evidence. The preceding goal revision is preserved at Git commit
`6390e80e7ee3e113419279702c2f872448d8c88d`.

## Next work: useful conditional image selection

The [gradient diagnostic](../references/visual_context_gradient_result_20261117.md)
is now complete on all three whole training records in both final actors. The
original host-gradient attempt stops at its pressure-wait ceiling with no complete
case. A separately bound device-copy recovery preserves all original arithmetic,
records and guards, completes in 183.602 seconds at 16.687 GiB peak allocation,
and verifies both full models and training-only projections unchanged. It makes
no optimizer updates or answer generations. Every combined retained-moment
AdamW direction has a negative inner product with the flow gradient; the
registered opposing-direction criterion is not met. Do not select new loss
weights from this result. First-order directions are not actual updated loss or
language measurements, and three records do not establish a corpus-wide cause.

The [saved-output analysis](../references/visual_context_saved_dependence_result_20261117.md)
checks all 792 streams and 488 comparisons without new neural calls or reading
repair. Every final normal QA stream is distinct. Literal prompt independence
is not established. Yet the learned-context actor has only fourteen distinct
first tiles across 32 normal questions, with one repeated ten times; the frozen
control has 32. Same-passage question pairs with matched first-span noise shapes
have median first-tile distance one pixel in treatment (16 pairs), versus sixteen
pixels in control (14 pairs). These are descriptive pixel effects, not semantic
understanding, and the matched subsets differ. Preserve the complete results.

The next algorithm work must address useful conditional image selection and
writing rather than assume opposing auxiliary gradients or repeat unchanged
training. Before selecting a conditional discrimination objective or output-law
change, read the existing failed contrastive, joint-density and writer results
and explain precisely how the proposed intervention differs. Choose one bounded
change with an unchanged control and training-only update/reset support on
complete authentic records. Require actual conditional writing improvement and
fixed complete-language evaluation before a full-corpus claim. No successor
full fit is currently admitted.

The [predictive-state comparison](../references/predictive_state_distillation_result_20261118.md)
is now terminal and fails all support checks. Both arms complete six real
updates on the same three complete training records. Across all 552 conditional
target glyph positions, F1 is .350346 initially, .336480 for cosine control and
.307754 for the KL replacement; false blanks are zero, two and five. Candidate
QA KL falls but book KL increases from 6.805294 to 14.433991. All trial model and
projection updates reset exactly; the frozen training-only readout is unchanged.
No successor is retained, autonomous language is unmeasured and no full fit is
admitted. Preserve all 1,656 actual output tiles and the failed preflight.

The [complete native-feedback calibration](../references/nonlinear_feedback_alignment_result_20261118.md)
fails its fixed reader gate after twenty epochs and 2,660 updates per arm on all
33,941 derived rows, weighted from every original training source. Error is
.578608694 affine and .542496895 nonlinear, from .639791851 initially. No
language fit is admitted. Do not repeat the fit or select an intermediate head.

The [separate weighted ridge refit](../references/feedback_ridge_refit_result_20261118.md)
keeps both final feature maps fixed and improves actual nonlinear error to
.271038527, versus .536556525 affine. Its ideal matrix error is .269287063;
actual stored-weight/BF16 execution remains separate. All six complete-record
trial solves improve and reset; original source head files/features are
unchanged. The 27.085-second, 1.135-GiB head-only run does not qualify a small
foundation or language efficiency. Final error and every complete-record limit
still fail. No full language fit, writer change or autonomous score follows.

The [saved-query identity analysis](../references/frozen_query_identifiability_result_20261118.md)
finds 2,682 conflicting duplicate groups on the same complete support, imposing
an exact-arithmetic .016774741 free-function floor. Most are same-image cross-
slot states: one-key cross-attention loses the query's slot identity when no
query residual is carried forward. This explains a real restriction, not the
whole .271 error. It does not rule out the unchanged .10 gate. Preserve all
conflicts, original image aliases and source labels; do not repeat their audits.

The [next bounded design](../references/profiled_visual_feedback_direction_20261118.md)
is feature learning with a solved output layer (variable projection), using
these same raw queries and source masses. Qualify its fixed-ridge envelope
gradient, source support, fitting/reset and numerical execution before a new
registered head comparison. Keep a common precision-matched fixed-feature
control; distinguish ideal arithmetic from actual weights. Do not silently add
slot residuals, change the pixel/query source, tune lambda or add old epochs.
The [implementation and separate preflight protocol](../references/profiled_feedback_preflight_protocol_20261118.md)
have four passing CPU numerical tests and a passing
[real complete-record preflight](../references/profiled_feedback_preflight_result_20261118.md).
All three trial steps improve and reset exactly. The book's re-solved finite
source derivative differs from the partial gradient by 3.20e-5 relative.
The 25.903-second, 1.355-GiB preflight qualified the separately registered
[twelve-step full-support comparison](../references/profiled_feedback_fit_result_20261118.md),
which is now terminal. Actual error falls from the common FP32 control's
.269500554 to .228074131; the .10 absolute error, .10 gain and all three
complete-record limits fail. All twelve first proposals reduce the fixed-ridge
objective, but numerical optimization is not useful visual language. Preserve
both final heads and every trial; do not extend this fit or choose an
intermediate. The 75.782-second, 1.246-GiB head-only result does not establish
whole-model efficiency or a trained successor.

The [read-only image consequence](../references/reader_writing_consequence_result_20261118.md)
is terminal. With identical noise, correct source-image histories and supplied
lengths, pooled F1 on all 552 targets per reader is .353468785 for control and
.353352368 for the learned head. Both have nine exact masks, one false blank
and four false-ink outputs. All 1,104 generated tiles are independently
recounted and exported losslessly. The unchanged actor and both heads pass
tensor-identity checks; no autonomous language is measured. Coordinate
improvement does not establish an actor repair. Do not continue this head fit.

The [writer-layer conditioning comparison](../references/writer_layer_conditioning_result_20261118.md)
is also terminal and fails. Both arms complete twelve whole-record optimizer
updates, with matched noise and exactly identical initial outputs. Pooled
physical F1 changes from .348600259 to .342911305 for unchanged conditioning
and .339850462 for modulation. Short-QA F1 improves to .876016260, but its
source masks remain nonexact and both book/long-QA case comparisons deteriorate.
Retain that limited positive effect alongside the failed pooled support gate.
Every trial resets exactly, all frozen tensors and stopping remain unchanged,
and all 2,208 generated tiles are independently recounted and exported losslessly.
The run takes 129.921 monitored seconds at 6.957 GiB peak allocation. It proves
execution of a physical learning comparison, not autonomous language.

Do not extend this conditioning trial, select an intermediate, change its
support thresholds or install the failed reader. No new algorithm is qualified
or selected for a full fit. The next substantive proposal must address complete
conditional language-to-writing learning, distinguish its mechanism from the
completed flow, mixture, joint-density, native-state, predictive-KL and modulation
trials, and retain the original language/confirmation criteria. A lower auxiliary
loss, an easy fitted short answer or another representational name is insufficient.
Use the existing positive complete-answer/counterfactual cases as retained
constraints without repeating their completed tests or repairing frozen readings.
Before any new execution, bind one failure-specific intervention, unchanged
control, complete training-only support, real gradients/reset and resources.
No deployment, language score or full-fit permission follows from the new code.

A new [lossless occupancy-density candidate](../references/occupancy_visual_density_direction_20261118.md)
is implemented, with eight passing CPU structure/probability/gradient tests.
It predicts a learned empty/nonempty root and generates physical quadrant masks
in a prefix-determined tree. The equal-parameter flat control has the same root
and a normalized nonempty branch. No glyph identity, font, reference topology
or lexical inventory enters generation. This is a different physical output
factorization, not a qualified language algorithm or a repetition of old
blank margins. Root errors and unreadable nonempty images remain possible.
The [registered CPU structure audit](../references/occupancy_density_structure_result_20261118.md)
now verifies all 68,948 saved-support cells exactly in 6.521 seconds. Both modes
preserve all packed bytes and 3,206 empty cells. On this saved support, tree
refinement averages 97.302 events versus 245.096 for the equal-parameter flat
control; both components have 2,261,074 parameters. No neural call or optimizer
update occurs in the audit. The event ratio .396997 is not a throughput,
compression-byte or language-efficiency measurement; the dense-image support
still permits 342 events. These cells are not corpus exposures or held-out language.

The [registered real-image preflight](../references/occupancy_density_preflight_result_20261118.md)
is now terminal. Both modes complete 96 whole-record updates and exact reset;
all 2,208 actual generated images/traces are independently recounted. Tree
F1 improves from .056570593 to .455979811, versus final flat .004023445, but
three of six fixed support conditions fail. Both have zero exact source masks
and ink at all four true spaces. All-event cached probability replay passes
within 9.54e-7 nat. Keep the finite contextual/root/mask gradients and limited
conditional NLL effect separate from readable or correct language. The run
uses .221 GiB peak allocation and 162.945 monitored seconds, excluding its
inherited 1,465,381,142-parameter actor and earlier cache preparation.

A [saved-array ink diagnosis](../references/evidence/occupancy_density_ink_mass_20261118.json)
finds opposite density errors: source mean 164.824 pixels, flat 1.772 and tree
418.797. Of 548 nonempty source positions, 547 flat outputs contain at most
four ink pixels; 366 tree outputs have more than twice their source ink.
These are descriptive post-hoc bins, not new rules or language scores. Do not
extend this failed fit, choose another snapshot or treat root nonemptiness
as successful writing. The lossless event savings remain a component result.

The [whole-form supervision candidate](../references/global_form_preflight_result_20261118.md)
is now terminal and fails seven of nine fixed conditions after all 96 updates.
Exact training-image equivalence yields 9,581 forms and covers all 552 targets.
Source-form helper accuracy rises from zero to 79/552, but physical F1 is
.455463432 versus unchanged tree .455979811, with zero exact masks and ink at
all four true spaces. Even short-QA helper accuracy 4/5 does not yield an exact
emitted image. Pooled rotated-minus-correct NLL .395939 fails its .5 condition.
Preserve both the classification improvement and the failed drawing.

All initial images match the saved control; no old control is re-executed.
All three complete-record gradient probes pass, every actual physical event's
probability replay is within 1.073e-6 nat, and all physical/helper updates reset
exactly. All 1,104 images, causal traces and helper decisions are independently
recounted. The helper is absent from generation. The 59.973-second, .233-GiB
head-only execution excludes the reused 1,465,381,142-parameter source actor
and earlier preparation, so it proves no whole-model efficiency. No successor,
autonomous score or full fit follows. Do not extend this failed trial.

The next algorithm decision must make a complete visual writing choice usable
by its actual generator, rather than repeat an auxiliary-label fit whose decision
is discarded. Keep global form selection, physical rendering and semantic
learning distinct. Read the completed continuous-codec, spatial-plan, prior-
guidance and exact-pixel generalization results before selecting another route.
A more compact physical representation is a hypothesis about modeling cost,
not proof of language or consumer-GPU foundation parity. Any new route needs a
bounded architectural change, real image support and unchanged controls before
its own prospective fit. The full language criteria below remain unchanged.

The next [implemented factorization](../references/binary_visual_plan_direction_20261118.md)
generates a 64-bit global visual plan, decodes it with a learned neural image
function, then generates a normalized physical XOR correction. The plan is
part of inference, unlike the discarded auxiliary classifier. For each plan,
XOR is bijective, retaining full binary-image support; training a fixed
posterior branch upper-bounds marginal image NLL rather than equalling it.
Quantization is explicit and is not a Unicode/glyph inventory. The 6,536,211-
parameter writer is untrained, excludes the contextual actor and establishes
no whole-model efficiency. Six CPU tests pass; the earlier signature-error
check is preserved. Existing model/data sources remain unchanged.

Next prospectively bind source preparation and a matched continuous/binary
64-dimensional codec comparison. Group exact image aliases in one physical
partition, retain all original bytes and leave reserved language data untouched.
Require faithful actual decoded images and lower actual refinement work before
registering a conditional prior/correction fit. Supplied-image compression,
mathematical normalization and numerical tests are not autonomous language.
Four additional CPU checks now qualify the matched continuous/binary training
operator, per-image objective and alias-preserving split. The
[source-only preparation](../references/binary_codec_preparation_protocol_20261118.md)
is complete: all 68,948 cells remain, with 64,041 training cells (8,629
forms), 4,907 physical-validation cells (952 forms), all aliases grouped and
3,206 empty training images. Two independent CPU metric checks also pass.
The [matched codec comparison](../references/binary_codec_comparison_result_20261118.md)
is terminal and fails six of nine conditions after all 6,000 updates per arm.
Validation F1 is .932736083 continuous and .810400756 binary; binary has zero
exact validation cells and loses .122335327 F1. The continuous branch also fails
its .98 ceiling. Both have zero source false blanks/ink. Binary source plan/
correction mean 30.015026 events passes the source event gate against 97.302489,
but poor per-form and held-mask fidelity rejects the recipe. Event counts are
not bits, throughput or language efficiency.

All complete-training-support preflight updates reset exactly, both actual
optimizer replays pass, and all 275,792 source reconstructions, correction
events and 12,000 retained updates are independently recounted. Both final
checkpoints and all source bytes remain. The 256.499-second, .593-GiB neural
execution loads no contextual actor and proves no whole-model advantage. Do
not extend this global 64-bit fit or select a snapshot. No conditional language
fit is admitted. A new bounded architectural decision must address the measured
common generalization loss as well as quantization; neither is a unique-cause
diagnosis or an impossibility proof. All original language criteria remain.

A [saved-plan information analysis](../references/evidence/binary_plan_information_20261118.json)
finds all 9,581 physical forms have distinct final 64-bit codes, with no code
conflicts or unstable aliases. All bits vary and empirical H(form|code)=0.
This rules out an exact-code collision floor on this finite support, not a
learning or universal-capacity limit. A [saved-frequency analysis](../references/evidence/binary_codec_frequency_20261118.json)
finds 110 common forms take 54.6% of training cells and reconstruct exactly,
while 5,821 once-occurring forms have binary macro F1 .823330. These descriptive
bins do not prove causality or change any rule.

The [equal-form codec comparison](../references/balanced_codec_comparison_result_20261118.md)
is now terminal after 6,000 updates per arm and fails seven of ten conditions.
It preserves initial weights, architecture, 64 bits and optimizer budget.
Binary validation F1 rises from .810400756 to .820592454, below the fixed .02
gain and every absolute fidelity requirement, with zero exact validation cells.
Continuous training macro F1 improves to .996662318 but validation macro F1
remains .903684546. Preserve the limited allocation effect and the remaining
generalization failure together; neither collisions nor word-frequency
imbalance alone explains the result. No unchanged extension is admitted.

All weighted complete-source preflight updates reset exactly; optimizer replay
passes and all 275,792 reconstructions/events and 12,000 updates are independently
recounted. Natural-frequency controls remain unchanged and are not rerun. The
257.158-second, .593-GiB physical run establishes no contextual language or
whole-model efficiency. The mask partition is reused development. A new
architectural proposal may test spatial representation/decoder geometry with
all original absolute criteria, rather than add discarded auxiliary labels,
unmeasured code width or more unchanged exposure. No conditional fit is admitted.

The [spatial 64-bit comparison](../references/spatial_codec_comparison_result_20261118.md)
is terminal after 6,000 updates per arm. Binary validation F1 improves from
.820592454 to .847344074, passing the fixed .02 gain, but six of ten conditions
fail with zero exact binary validation images. Continuous validation F1 is
.929688379. Spatial structure is helpful in this bounded comparison and remains
insufficient for faithful drawing. Both 1,767,365-parameter codecs preserve
all 1,766,209 common CNN initial values and the original balanced source order;
old global controls are not rerun. Every original threshold remains.

All real preflight updates reset, optimizer replay passes and all 275,792
reconstructions/events and 12,000 updates are independently recounted. The
261.088-second, .524-GiB run loads no contextual actor and proves no language
or whole-model efficiency. Keep the limited architectural gain and the failed
absolute quality together. Do not extend this recipe or select a snapshot.
A later spatial capacity intervention must keep geometry, allocation, loss
and budget fixed, declare increased plan events and preserve every original
quality/resource condition. No conditional language fit is admitted.

The [spatial capacity comparison](../references/spatial_capacity_comparison_result_20261118.md)
is terminal after 6,000 updates per arm. At 256 values/bits, continuous physical
validation F1 reaches .996423850 and macro F1 .993422244, with 316/952 exact
forms; binary F1 reaches .943690071 and macro F1 .928693592, with only three
exact forms. The common CNN initialization, data/order, loss and resources stay
fixed. Each codec has 1,770,449 parameters. Binary training F1 .981661645 passes,
but four of ten conditions fail: held pooled/macro F1, macro exactness and
continuous retention. Thirty-two plan events plus actual XOR refinement average
49.475474 on the complete source, passing the original total event condition.

All complete-source preflight updates reset, optimizer replay passes and all
275,792 saved reconstructions/events and 12,000 updates are independently
recounted. Saved spatial-64 controls are unchanged and not rerun. The
256.635-second, .524-GiB physical run loads no contextual actor. The continuous
motor can reconstruct new forms accurately on this reused development split;
hard quantization still loses detail. Neither observation proves language or
whole-model efficiency. No conditional language fit is admitted. Preserve the
failed absolute rules and test a bounded representation change before any
successor fit; do not extend unchanged epochs or select a snapshot.

The [equal-bit scalar-plan comparison](../references/scalar_codec_interrupted_result_20261118.md)
is interrupted and unqualified. Its 128-value continuous control completes all
6,000 updates but reaches only .978102816 validation F1, below the unchanged
.98 ceiling. The hard arm stops before any update because all initial features
round to zero and the preflight requires a nonzero expansion-weight gradient,
which is mathematically zero for a zero input. This does not establish zero
encoder/total decoder gradient or inability to learn. Hard-code quality remains
unmeasured; no unchanged recovery is selected. All 137,896 saved control images
and 6,000 retained updates are independently audited. The original failure
checkpoint's stale 6,000 counter belongs to the previous continuous arm; actual
hard weights equal initialization and no hard update ledger exists. Preserve
the file, corrected interpretation and earlier CPU-test expectation failure.
No language fit is admitted, and every original criterion remains unchanged.

The selected next [physical intervention](../references/residual_plan_motor_direction_20261118.md)
trains the previously untrained normalized XOR correction while freezing the
actual spatial-256 codec. Its condition contains generated plan bits and the
frozen base pixels, all available at deployment. Greedy identity initialization
matches the unchanged base for every possible plan. The implemented motor has
2,385,763 parameters including its frozen 1,770,449-parameter codec; 615,314 are
trainable correction parameters. This excludes all contextual language weights
and was initially untrained. Three CPU checks pass for initial writing, likelihood,
actual gradients/update/reset, fixed-codec identity and target-encoder exclusion.

Its [complete source preparation](../references/evidence/residual_motor_preparation_result_20261118.json)
retains all 68,948 saved plans/bases and original target bytes with the unchanged
physical partition. It takes 1.575 seconds with no neural calls or training.
True residual targets are not generated corrections. The separate
[learning comparison](../references/residual_motor_comparison_protocol_20261118.md)
is now prospectively registered: complete-source real gradient/update/reset,
6,000 updates, unchanged sampling/budget, all-source initial/final actual
generation and every emitted-prefix probability replay. The new admission unit
is the composite motor D(B) XOR R, with every original absolute fidelity,
exactness, empty-image and total-event condition retained; the failed base stays
failed. Actual generated correction work and remaining error complexity are
reported separately. No conditional language fit, whole-model efficiency or
autonomous score follows from preparation or registration.

The [first correction-fit execution](../references/evidence/residual_motor_admission_failure_20261118.json)
ends at the 48-GiB host entry guard, constructing no model and making zero
updates. No drawing or language quality is measured. Preserve its original
directory. A separate [admission recovery](../references/residual_motor_admission_recovery_protocol_20261118.md)
retains every model, source, schedule, quality and resource rule, with a new
output directory and one worker maximum. It first waits for two stable
50-GiB host observations ten seconds apart; the original worker's 48-GiB guard
remains. Scheduling wait is at most 300 seconds and included in a separately
enforced 2,400-second total ceiling. Do not duplicate its owner, reduce reserves,
kill foreign services or retry a worker exception automatically.

That [scheduler is terminal](../references/evidence/residual_motor_scheduling_failure_20261118.json)
after 291.299 seconds with no worker or model. Both reserve-only attempts remain
preserved. The current separately registered [physical-head resource amendment](../references/residual_motor_head_resource_protocol_20261118.md)
uses 40-GiB entry for this 2,385,763-parameter motor only: 24 GiB unchanged live
host reserve + 8 GiB unchanged maximum process RSS + 8 GiB additional entry
margin. It retains all GPU, disk, pressure, mutex, time and thread limits. The
original full-context admission source remains unchanged; no contextual model
is loaded. This is an explicit resource change before any model execution,
not a change to quality, mathematical learning rules or failed model scores.
Every source, initialization, update, generation/probability test and physical/
language criterion remains fixed. Launch at most once, qualify real complete-
source memory/gradient/update/reset before retained fitting, and preserve any
resource violation. No automatic retry or language admission follows.

The [first head-resource execution](../references/evidence/residual_motor_loading_failure_20261118.json)
passes admission but stops at serialized-plan loading, before constructing a
model or making any update. The uint8 plan bytes need widening before the
shared validator's scalar-256 comparison. The original component is unchanged.
The [registered CPU loader qualification](../references/evidence/residual_motor_loader_result_20261118.json)
recovers all 2,206,336 bytes / 17,650,688 bits exactly across all 68,948 plans
in .490 seconds, with no neural call or source change. A separate unit check
covers every byte value. Preserve this loading failure and the earlier resource
attempts, all of which leave physical quality unmeasured.

The [loaded comparison](../references/residual_motor_loaded_comparison_protocol_20261118.md)
is now separately registered. Its training body changes only the explicit
integer widening and the already selected physical-resource context. Model,
source pixels, frozen codec, initialization, objective, sampling, 6,000 updates,
actual generation/probability tests and every quality criterion stay fixed.
Use its one serial owner and new output path. Do not alter bound sources or
restart any earlier run. Only actual complete-source update/reset and final
generated-image qualification can establish a trained motor; neither code nor
loading proves language.

The loaded trial is now [terminal at optimizer replay](../references/evidence/residual_motor_replay_failure_20261118.json).
It passes the real complete-source update/reset path and generates all 68,948
initial images equal to the unchanged base, then retains two updates and fails
the exact third-update replay. No final trained-image result exists. The saved
failure state includes an unaccepted replay attempt; do not install it or call
it a qualified two-update model. The combined check did not retain which
loss/gradient/model/optimizer component differs, so the cause is unproved.

The [registered numerical diagnostic](../references/residual_motor_replay_diagnostic_protocol_20261118.md)
compares the original global deterministic-algorithm flag with strict true,
keeping actual initial tensors, source batches, loss and optimizer fixed.
It reconstructs two updates and repeats the actual third four times per mode,
recording every gradient/model/optimizer and RNG digest. All twelve trial
updates are discarded and reset. cuDNN flags alone do not cover all CUDA
operations, but neither attention nor embedding reductions are yet an isolated
cause. Require an observed unequal original replay and exact candidate replay
before a separately registered fresh full physical fit. No language, final
motor quality or efficiency follows from numerical reproducibility.

The [numerical diagnostic is complete](../references/residual_motor_replay_diagnostic_result_20261118.md).
Original-mode repeats have identical loss, norm and model weights but depth/
row/column embedding gradients differ by up to 5.820766091346741e-11, changing
six optimizer-moment tensors. Strict global deterministic mode makes all four
repeats exact, including every gradient and optimizer tensor. RNGs do not change;
all twelve trial updates reset. The 1.674-second, .310-GiB execution qualifies
this setting, not physical writing or language. Tiny numerical differences do
not establish a cause of earlier unreadable language.

A separately registered [fresh deterministic fit](../references/residual_motor_deterministic_comparison_protocol_20261118.md)
uses the committed loaded worker with this one global execution flag and new
output path. All model, data, quality, source partition, initialization,
6,000-update schedule, actual-generation/probability and resource rules stay
fixed. Do not resume the failed weights or select a checkpoint. Observe its
single serial owner and independently audit all actual final images before
admitting any conditional-language work. This fit is now terminal and audited.

The [complete correction-motor result](../references/residual_plan_motor_comparison_result_20261118.md)
fails seven of nine fixed conditions after all 6,000 updates and 768,000 training
draws. Validation F1 declines from .943690071 to .933584204, macro F1 from
.928693592 to .916099307, and exact cells from 94/4,907 to zero. Every one of
the 3,206 true spaces gains ink. Actual plan-plus-generated work averages
45.998129 events; this passes its event condition while physical quality fails.
All 137,896 initial/final images, emitted actions, complete causal traces and
6,000 retained updates are independently recounted. The strict GPU probability
and optimizer replay checks pass. Numerical correctness does not repair writing.
The monitored fit takes 434.507 seconds, including 140.133 seconds of pressure
wait, at .485 GiB peak allocation. These physical-only costs exclude preparation,
earlier codecs and any contextual foundation. Preserve the fixed base, failed
motor, every prior attempt and original criteria. No unchanged extension,
intermediate selection or conditional-language admission follows. Autonomous
language remains unmeasured for this failed physical actor.

The next [native continuous-density candidate](../references/continuous_autoregressive_language_direction_20261118.md)
returns to causal language selection using the already qualified V34 motor,
without installing the failed binary or correction actors. A newly initialized
causal image reader predicts 48 groups of continuous writing coordinates, each
with a normalized shared-component Gaussian mixture. Compare one versus eight
components, with identical common reader/local initial weights. Actual generated
coordinates feed the neural decoder; the next history contains the emitted
pixels re-encoded, never the unrendered coordinate proposal. A learned stop head
receives no target length. No inherited language body, teacher state, lexical
head or glyph bank is used. Code density is not exact pixel likelihood.

Three CPU tests cover distribution scoring, both causal levels, cached attention,
actual updates/reset, frozen codec, reproducible sampling and emitted-image
feedback. The [registered GPU preflight now passes](../references/continuous_autoregressive_preflight_result_20261118.md)
on all three original complete training cases in each arm. All six actual
trial updates reset, the fixed codec/common initialization checks pass, and an
independent audit recounts every one of the 2,208 generated images. Execution
takes 11.962 seconds at 1.336 GiB peak allocation. The candidate has 23,549,194
whole parameters, including the previously trained 7,423,361-parameter codec;
no language weights are inherited. Code NLL falls slightly in all six cases,
but updated supplied-history F1 is only about .22–.23 and most loss reduction
comes from stopping. This is execution support, not trained physical quality
or language. Do not repeat this completed preflight. Its pass permits only a
separately bound complete-corpus comparison.
Keep the original complete-answer, book, retention, erased-input, stopping,
untouched-confirmation and whole-model cost criteria. The requested 1,000-fold
efficiency remains an unproved target, not a property inferred from this design.

The [complete native comparison](../references/continuous_autoregressive_full_comparison_protocol_20261118.md)
is now implemented and separately registered. Its two new CPU execution checks
pass on the authentic source: all 13,785 complete training records remain,
with 2,571,477 next-image targets per arm. This excludes 4,762 first book images
used as causal input anchors from the earlier 2,576,239 source-position count.
The longest record has 1,101 images and no training truncation is needed.
One shared shuffled order gives 13,785 updates per arm; retain all five fixed
snapshots and the 792-stream fixed-cell autonomous plan. An independent CPU
completion audit must precede generation. The full proportional, earlier-
success, cold-independent and untouched-confirmation requirements remain later
requirements even if this development comparison passes. Use the one serial
owner in the private handoff. Do not edit bound sources, duplicate its owner,
restart on observation timeout, add unchanged epochs or infer language from loss.
Quality, complete language and efficiency remain unmeasured before evaluation.

Both native fits and the independent CPU schedule audit are complete.
The [fit receipt](../references/evidence/continuous_image_training_completed_20261118.json)
records 27,570 updates, 5,142,954 next-image exposures, 5,170,524 stop
decisions and ten snapshots. Monitored training totals 1,306.37 seconds
on one RTX 4090 D with 1.416 GiB peak allocated memory. The [completed blind evaluation](../references/native_continuous_blind_result.md) covers all 792 original outputs. Both final arms give zero of 32 readable or correct answers and zero of 12 coherent long continuations. The eight-component model reached the cap on every long normal book. Preserve the frozen failure. A training-only frozen encoder cache gives
1.53–2.86× faster warm forward/backward on three complete CPU cases with
identical measured codes, losses and gradients. This establishes neither GPU
training acceleration nor a matched-quality advantage over a text model.
The registered fits used the ordinary encoder.

The [visual word-order evaluation](../references/visual_word_order_result.md)
compares the same 552 Mandarin sentence pairs at all five native K=8 checkpoints.
Final grammatical preference is 284/552 (51.4%) versus 299/552 (54.2%) before
training and 57.2% for a matched visual bigram counting adjacent bitmaps.
Paradigm-level intervals include chance and zero improvement. Preserve all
paired margins, checkpoints and this negative result. Avoid treating a decrease
in writing-code loss as evidence that the reader learned syntax. The active
792-stream autonomous qualification is now complete and failed.
The next learning decision should account for both visual ordering and the
eventual complete-answer evidence before another heavy fit.

The [matched training-source comparison](../references/visual_passage_register_result.md)
recovers 2,127 unique modern passages with 1,047,284 direct next-image targets.
Two visual bigrams trained on equal numbers of transitions score 59.5% for
modern passage ink and 57.6% for historical book ink on 6,062 additional
pairs. The paired construction interval for the 1.9-point difference spans
−9.0 to +12.7 points. The data adapter is verified; changing register alone
has not passed the criterion for a follow-on neural fit.

The [actual 言 question](../references/native_yan_inference_result.md) produced
identical 128-cell greedy outputs from typed and pictured prompts. Greedy and
sampled output were both unreadable. At every position in both answers the
writing code changed more after drawing and re-reading than the largest change
among 9,581 distinct source images. In the sampled answer, mean and scale
explain only 0.39% of squared discrepancy. Preserve the original pixels and
directions. A scalar normalization remedy misses its prespecified criterion.
This one question is separate from the complete autonomous evaluation.

A separate [CPU passage reader](../references/visual_extractive_reading_result.md)
recovers 269/1,788 complete held-out answer rasters from the supplied passages.
Question erasure gives 18/1,788 and same-passage exchange gives 11/1,783.
This passes the registered question-dependence test for extractive reading.
Rotating answer assignments among training questions about the same passage
reduces exact answers to 59/1,788. A whole-passage interval puts the advantage
of correct supervision at 10.1--13.5 percentage points.
Preserve it as a compact reference for selecting evidence from a supplied
page. Its scope is a contiguous passage span; standalone writing about an
absent fact, historical glyph synthesis and matched-quality cost remain open.

Historical diagnostic registrations, all failed executions and frozen readings
remain protected. No development answer, reserved DRCD record, rejected reward
judge or policy fit may enter the new training objective. Source-conditioned
contrast or image likelihood is training supervision, not evidence of language
until autonomous complete answers and books meet the original criteria.

The earlier white-target comparison, both reward judges, isolated-cell teacher,
fixed encoder metric, label uniqueness and preprocessing audits remain terminal.
Do not recover old evaluators, tune rejected thresholds or reuse development
outputs as fitting data. Physical aliases and invisible source symbols remain
preserved; an exact pointwise Unicode condition's failure is not proof that
visual language is impossible. The reserved DRCD collection remains unrendered,
unscored and evaluation-only.

Every successor fit requires its own failure-specific, prospectively bound
training-only support/gradient/resource preflight on a fresh clone with all trial
updates reset. Bind complete sources, one intervention, unchanged control,
initialization, schedule, checkpoints, full evaluation, resource ceilings and
actions for either result. Direct supervised proposals need no reward judge;
policy proposals also need qualified complete-action rewards and exact emitted-
pixel probability replay. More unchanged epochs are not a selected intervention.

An ordered visual volume remains a representation hypothesis. Its completed
invertible CPU operator preserves order but does not establish compression,
faster language learning or coherent generation. Any future block must predict
future visible spans without receiving its own target block as context. Do not
replace language learning with supplied-answer reconstruction.

## Acceptance criteria

The unit of success is a **complete correct answer or coherent continuation**.
Copying, glyph fidelity, canonical bitmaps, loss, stopping, cold execution and
resource savings remain distinct claims.

1. Demonstrate complete modern-Chinese answers and coherent book continuation
   on held-out works and passage groups, using actual emitted pixels as history.
   Ask different questions about the same passage and retain question-, passage-
   and prefix-erased controls. Preserve raw outputs, natural stops, caps and
   uncertain readings; freeze review rules before generation and readings before
   new reference scoring. Canonical mismatch alone does not establish falsity.
2. Preserve the original complete-response pilot: at least 12/32 strict
   proportional answers, 4/16 both-correct pairs, 24/32 readable answers and 4/12
   coherent complete long books; all eight proportional and two fixed earlier
   successes; each QA-erasure arm at most two exact answers and a dependence gap
   of eight. Keep caps 128/640, emitted-image feedback and natural stopping.
   A physical failure leaves autonomous language unmeasured, not zero.
3. Confirm development improvements on untouched source groups. The reserved
   DRCD pool has 998 paired passages, 3,485 questions and 6,970 human answer spans.
   It remains evaluation-only, unrendered and unscored, excluded from fitting,
   reward/preference calibration and tuning. Its known-source screen does not
   prove semantic novelty or unknown inherited pretraining exposure. A separate
   committed confirmation execution is required after a qualified development
   comparison; this pool does not replace books or historical-form requirements.
4. Reproduce training and standalone inference within one 24-GiB GPU. Account for
   the entire model, inherited weights, preparation, teacher cost, wall time,
   peak VRAM and latency. Compare useful matched quality against a named local
   Qwen baseline before claiming efficiency or large-model parity. A prototype
   attached to a 7.522B inherited body is not a compact foundation.
5. Extend demonstrated language to held-out writing styles and attested
   historical or unencoded forms. Evaluate explanation, historical stage,
   glyph fidelity and provenance separately; label reconstructed forms.
   Demonstrate Chinese/English word-origin questions as part of the full goal.
6. Maintain Nature-style TeX/PDF and editable native PowerPoint figures from
   actual measurements, preserving original images and negative results. These
   artifacts support the evidence; publication does not qualify the model.

A development pilot alone does not complete this objective. Novelty, human-like
reading and broad language capability require their own evidence. The requested 1,000-fold training-efficiency gain is a stretch target.
Define the ratio as the baseline compute required to reach a fixed held-out
quality divided by the ILM compute required to reach that same quality. Use
the same task, source groups, output grading and stopping criteria. Report
wall time on identical hardware, data exposure and whole-system energy where
metered. Charge corpus preparation, visual codec fitting and teacher work
explicitly. Report inference latency per complete correct answer separately.
An existing pretrained Qwen can establish a quality and serving comparison;
estimating its training cost requires documented comparable information.
A smaller checkpoint or faster span extraction does not measure the target
ratio for a general language model.

## Completed constraints and data

The last [autonomous continuation result](../references/visual_continuation_learning_result.md)
is 7/32 strict proportional answers, 0/16 both-correct pairs, 14/32 readable
answers and no credited coherent complete long-book continuation. All 732 streams,
183 boards and 96 normal-book prefixes are reviewed. Both the original scorer and
independent four-QA-erasure audit reject the trial. Preserve all five checkpoints;
the better intermediate result does not replace its prescribed final result.

Reuse the completed [clean-history](../references/continuation_clean_history_result.md),
[trajectory](../references/visual_continuation_trajectory_result.md),
[joint-writer](../references/joint_writer_physical_result.md),
[context-gradient](../references/joint_context_physical_result.md),
[nonempty-ranking](../references/global_glyph_ranking_physical_result.md),
[whitespace](../references/ranked_writer_whitespace_result.md) and
[saved-score](../references/ranked_glyph_score_gap_result.md) findings. Books fail
with correct history, so a feedback-only explanation is insufficient. Earlier
native-reader, teacher-state, question-assignment, recognition-loss, copying,
writer, codec and released MIXAR controls are terminal; do not repeat unchanged
fits, evaluator recovery or diagnoses. Preserve contradictory readings and
original scores. The preceding full goal text is retained at Git revision
`2dda87d203ce8e137a0c5450fa9efd9c28c1f302`; historical directions are indexed in
[the decision archive](ilm-goal-decision-history.md).

Use the authenticated corpus and whole-work/passage-group splits. The original
13,785 training sequences contain 2,576,239 target glyph positions; the complete
matched continuation comparisons retain 13,766 eligible sequences. Exposures are
not unique facts or foundation-scale pretraining. Add data only with a declared
coverage or supervision reason, provenance and group splits. Preparation is not
student-language evidence.

The [current manuscript](../publication/ilm-nature/manuscript.pdf) reports independent writing in Figures 40, 41, 43 and 44 and source-grounded historical forms in Figure 42. The reading, review and cumulative redline editions contain 78, 129 and 132 pages. The [verified build](../publication/ilm-nature/evidence/passage_diversity_article_build.json) preserves earlier figure assets. The [local delivery receipt](../publication/ilm-nature/evidence/expression_preference_local_sync.json) records 1,379 files, the preserved previous package and the two current Nutstore PDF aliases. Cloud propagation is unverified.

## Work-unit and resource rules

Read this goal and `artifacts/glyph_tail_current_handoff.json` at each model work
unit. Verify actual results and live process identity before acting. Record the
new evidence and next decision at the end. Keep private runtime details out of
commits and historical result files out of live scheduling authority.

Run at most one heavy project job under `artifacts/ilm_gpu0.lock`. Before a heavy
launch inspect host RAM, swap, pressure, owned processes, tmux and ports. The
existing full-reader entry requires 48 GiB available host RAM and 20 GiB free
GPU; retain 24 GiB host RAM, 48 GiB disk and at most 22 GiB allocated VRAM, or
stricter registered ceilings. Use two CPU threads. Protect foreign projects,
reuse shared weights and stop obsolete owned runtimes after evidence capture.
An observation timeout or serial phase transition does not authorize a duplicate
or restart.

Prioritize the failure-driven model comparison and complete-language evidence
over additional source preparation, copying repairs, covers or manuscript
polish. Commit and push related source/document changes while preserving ignored
private artifacts and unrelated work. Keep the session goal active until every
language, independence, resource and written-form requirement is proved.
