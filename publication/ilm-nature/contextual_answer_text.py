"""Completed paired image answers, with failed gates and scope intact."""

SUMMARY = r'''Written language includes handwritten and historical forms outside fixed character inventories. We investigate language generation that reads and emits images, using continuous glyph feedback without a runtime tokenizer, lexical output head, font renderer or glyph bank. An externally pretrained 7.522B visual contextual body supplies inherited language knowledge; exact frozen-prefix caching permits final-layer adaptation on one consumer GPU. Across 3,176 short-answer training questions, 688,128 contextual adapter parameters and 17,848,833 writer/stopping parameters are trained for 18,000 updates. On 32 prospectively reserved questions from sixteen new contexts, contextual adaptation gives sixteen clear complete literal matches, versus eleven for a fixed-body readout control. Both different questions are correct in five versus two contexts; canonical pixel-exact complete answers number two and one. Neither condition passes the predefined complete-answer pilot or training-pixel qualification. First-glyph training F1 remains below later reference-conditioned fidelity, showing that feedback drift cannot explain every failure. A cold standalone replay verifies actual emitted-pixel feedback without answer-bank or lexical access on one fixed input. These results support partial image-conditioned answer transfer and identify distinct initial-decision and autonomous-continuation bottlenecks. Reliable open-ended writing, historical-script generalization and an efficiency advantage at matched language quality remain unresolved.'''


def frontmatter(baseline):
    marker = r'\newcommand{\PaperSummary}{'
    if baseline.count(marker) != 1:
        raise ValueError('unique final abstract macro required')
    prefix, old = baseline.split(marker)
    if not old.endswith('}\n') or '\n' in old[:-2]:
        raise ValueError('preserve the complete one-line abstract boundary')
    return prefix+marker+SUMMARY+'}\n'

SUMMARY=SUMMARY.replace(
    'First-glyph training F1 remains below later reference-conditioned fidelity, showing that feedback drift cannot explain every failure.',
    'A further balanced writer fit passes physical qualification but gives eighteen clear answers versus nineteen for its parent on a different reserved cohort, separating training fidelity from generalization.')

RESULTS = r'''\subsection*{Contextual adaptation improves autonomous image answers}
Adapting the final four contextual layers, together with the image writer and
stopping head, improves complete answers relative to a broader-data readout-only
control (Supplementary Note 62; Fig.~S41). Both conditions use 3,176 training
questions and 18,000 updates. On 32 prospectively reserved questions from 16
contexts, conservative clear complete literal matches are 16/32 versus 11/32;
both different questions are correct in 5/16 versus 2/16 contexts. Canonical
pixel-exact complete answers are 2/32 and 1/32. Neither condition meets the
predefined pilot requirement of 24/32 matches and 10/16 paired contexts, and
both retain a failed training-pixel qualification. The result demonstrates
partial image-conditioned answer transfer, not a reliable general language
model. Training first-glyph fidelity is lower than later reference-conditioned
fidelity, so generated-image feedback cannot explain every failure. A separate
cold replay reproduces the full image-feedback execution on one fixed input
without runtime lexical, font or answer-bank access.

'''

RESULTS+=r'''An additional balanced writer refinement isolates a training-fit problem.
With the learned reader fixed, batched first/later supervision raises training
F1 from 0.963524 to 0.999811 and passes physical qualification. On a different
reserved cohort, however, clear complete answers are 18/32 versus 19/32 for the
unchanged parent, with two canonical exact answers each. The one-case reading
difference is a damaged initial glyph, not an established semantic regression.
Neither passes the answer pilot (Supplementary Note 62; Fig.~S42). Stronger
reference-conditioned fitting alone has not established a generalization gain.

'''

NOTE = r'''\clearpage
\section*{Supplementary Note 62 | Paired autonomous image-answer results}

\textbf{Design and boundaries.}
The prospective contextual fit introduced in Note 61 is now complete. It
trains 688,128 rank-8 q/v/o adapter parameters in source layers 32--35 together
with 17,848,833 image-writer/stopping parameters. The matched control fixes the
entire body. Both use the same 3,176 admitted questions,23,262 answer-glyph
targets, pixel-only reconstruction, balanced terminal/continuation supervision,
seed and 18,000-update schedule. Three out of four updates sample one question
sequence; the fourth rehearses 128 genuine-copy states. No diagnostic source-head
trajectory enters training. EncoderE, decoderD and normalization stay fixed.
The body retains 7,522,463,984 inherited parameters; pretraining exposure is
unknown. A small adapted parameter count is not a small trained foundation.

\subsection*{Completed training and interruption accounting}
The cache retains every frozen-prefix position and rotary coordinate, rather
than treating a changing contextual layer's final states as fixed. Original
execution saves 810 cases before interruption; recovery completes the remaining
2,366 with zero checked cold-replay difference. A fit interruption after 2,926
executed updates retains the periodic 2,000-update checkpoint. Recovery restores
model, optimizer and RNG state and replays 926 unsaved updates. An independent
four-update serialization check reproduces weights, optimizer buffers, RNG,
sample identities and losses exactly; these verification updates are discarded.

The final contextual checkpoint retains 18,000 updates, including 13,500
question updates and 4,500 copy updates. Its resumed 16,000-update attempt takes
3,355.40\,s, including 570.54\,s pressure waiting, and peaks at 3.357\,GiB allocated
CUDA memory. This excludes prior cache work, the interrupted attempt and
inherited pretraining; it is not complete end-to-end training time. The
readout-only fit takes 341.15\,s. AdamW counter and tensor audits confirm the
completed schedules, finite weights, unchanged frozenE/D and unchanged zero
contextual deltas in the control. Both final RNG states match; an identical
complete observed sampler trace is not claimed because it was not saved for
every primary update.

\subsection*{Physical qualification remains below the training gate}
All 23,262 reference-history training predictions and 256 held-wording copy
inputs are evaluated. Every held-copy image passes through the actual full
adapted reader; original unadapted copy features are not substituted for this
test. Training mean foreground F1 is 0.963524 for contextual adaptation and
0.954784 for the control, both below the unchanged 0.98 threshold. Held-copy
mean F1 is 0.957366 and 0.968693;208/256 and 225/256 exceed 0.95. Terminal decisions
are correct for 3,130/3,176 and 3,134/3,176 states; continuation decisions for
23,030/23,262 and 22,974/23,262. Held-copy non-stop counts are 255/256 and 254/256.
Copy and stopping thresholds pass, but the joint qualification fails. Running
the prospectively fixed answer assay despite this failure does not waive it.
All qualification pixels and sixteen actually viewed example boards are kept.

\subsection*{Complete image answers and paired questions}
Each condition generates all 72 registered streams: eight training controls,
sixteen earlier development questions, their question-erased counterparts,
and 32 reserved questions from 16 distinct contexts with two questions each.
The reserved contexts exclude training and earlier development contexts under
the registered hash ordering. They are not an exhaustive historical or source
pretraining decontamination. One case has no fully supported canonical
reference form in the physical evaluator and remains in every denominator.

The actual emitted binary glyph images return throughE to determine the next
contextual state. No lexical classifier, tokenizer, glyph bank, font renderer,
reference string or external teacher participates in generation. The maximum
length is 64 glyphs; all failures, stop events and raw pixels are retained.
Every one of the 36 raw-output boards is inspected before either reference
score, with 27 ambiguous stopped streams additionally inspected as unchanged
nearest-neighbour enlargements. Conservative agent readings retain null for
uncertain forms; they are not independent human grades. Complete literal
matches require a natural stop and a clear reading equal to an allowed
reference. An intelligible longer answer is not retrospectively accepted by
changing that rubric. Canonical image equality is a separate, stricter test.

Both conditions give eight clear complete matches on eight training controls.
Earlier normal development gives 6/16 versus 4/16; question erasure gives 0/16
for both. The reserved cohort gives 16/32 versus 11/32 clear complete matches,
21/32 versus 18/32 natural stops, and 11/32 versus 14/32 caps. Canonical complete
image matches are 2/32 versus 1/32. Eleven clear correct reserved answers are
shared, five occur only with contextual adaptation, and sixteen fail in both.
These are small single-seed descriptive counts, not a population effect estimate.

Both questions are clearly correct in 5/16 contexts with adaptation and 2/16
with the readout-only control. All sixteen paired allowed reference sets are
disjoint; the successful pairs also have different actual output pixels.
This supports question use within the tested passages without establishing
open-ended language competence. Neither condition passes the registered pilot
of at least 24/32 matches and 10/16 paired contexts. The earlier question-erasure
control is not a new erasure test on the reserved cohort.

Generation wall time over all 32 reserved cases, including failures, is
130.020\,s and 144.046\,s. Timing includes image prefill, actual pixel feedback
and stopping, but excludes checkpoint loading, prompt-file decoding and output
serialization. No matched text-LLM efficiency comparison has been performed.

\subsection*{The first image decision remains a distinct failure}
For adaptation, first-glyph training F1 is 0.893582 over 3,176 positions, whereas
later reference-conditioned positions reach 0.974584 over 20,086. The control
gives 0.880031 and 0.966604. The first decision precedes any generated-image
feedback. Its failure therefore cannot be attributed entirely to a later
feedback clock mismatch. Later fidelity receives correct preceding images
and does not measure autonomous completion.

There are 870 distinct first target forms,449 occurring once as a first target.
All 3,176 physical training inputs are unique; no identical input has conflicting
targets. With 13,500 single-sequence question updates, the mean exposure is
4.25 visits per question. These descriptive observations motivate separately
testing first-decision weighting and greater batched writer exposure. They
do not prove that more training will repair generalization, or that contextual
states lack the information for a better image decoder. No result from that
subsequent repair is included here.

\subsection*{One cold standalone execution}
The prospectively fixed first reserved input, rather than a selected success,
is rerun through the standalone image-only command in an empty working
directory with an empty model cache and offline flags. All saved pixels,
hidden states, stop probabilities, feedback events and PNG bytes reproduce
exactly. System-call tracing records no IP connection attempt or forbidden
corpus, font, tokenizer or answer-bank file access. The shared source archive
can contain unused vocabulary metadata; no runtime lexical module is loaded.
This establishes one execution's interface isolation, not general accuracy.

The result leaves damaged initial forms, answer-selection/length errors and
unstable feedback continuation unresolved. It does not yet demonstrate
general writing, multiple fonts or historical scripts, a lightweight
independent foundation, or parity with a general-purpose LLM.

\clearpage
\plate{Figure_S41_contextual_image_answers}{\textbf{Partial transfer to autonomous image answers.}
\textbf{a}, Clear complete literal matches on the 32 prospectively reserved
questions; both conditions miss the fixed 24/32 pilot threshold.
\textbf{b}, First and later training foreground F1, with later predictions
conditioned on correct preceding images. These are not autonomous answer
accuracies. \textbf{c}, The actual pixel-feedback path retains a learned
contextual body and a continuous image writer, with no lexical runtime head.
\textbf{d}, The first eight reserved questions for both conditions, in fixed
index order, including damaged/capped failures. Prefix viewports are marked
where full streams exceed ten glyphs; complete original PNG bytes remain
embedded in the editable figure. All 144 output streams are archived.
Agent transcription, canonical image equality and natural stopping remain
separate. Neither the pilot nor the training-pixel gate passes.}{fig:contextual-image-answers}
'''

REFINEMENT_NOTE=r'''\subsection*{A passed physical gate does not restore new-context answers}
The next fixed experiment initializes from the completed contextual parent
and freezes its entire learned reader, contextual deltas, E, D and normalization.
Only the existing 17,848,833 writer/stopping parameters change. Final training
states are recomputed through the actual fixed learned tail; every one of the
23,262 parent pixel scores and all stopping counts reproduce before fitting.
With the reader fixed, these cached states preserve the restricted gradient
for the new writer exactly.

The additional 18,000-update fit samples 64 first and 64 later training states
on each of 13,500 question updates, with equal image-loss weights for the two
groups. It separately samples 64 terminal states for balanced stopping. The
other 4,500 updates rehearse 128 genuine-copy states each. Exposure and position
weighting change together, so their individual effects are not identified.
These are 864,000 sampled first targets, 864,000 later targets and 576,000 copy
targets, not new independent questions. No held-out answer enters fitting.

The fixed-state cache takes 1,051.67\,s, including 630.60\,s pressure waiting,
at 2.168\,GiB peak CUDA allocation. The new fit takes 156.38\,s, including
20.02\,s waiting, at 1.096\,GiB. A failed initial admission precedes model
loading and every optimizer update; its retry preserves the same recipe and
physical limits. All inherited parent training and source pretraining remain
part of the method. An independent tensor audit confirms actual AdamW counters
and bit-identical frozen modules and inherited contextual deltas.

Training mean F1 rises from 0.963524 to 0.999811; first-position F1 from 0.893582
to 0.999559 and later F1 from 0.974584 to 0.999851. Canonical exact first forms
increase from 995/3,176 to 3,111/3,176. Every training terminal and continuation
decision is correct. Through the actual full adapted reader, held-copy F1 is
0.961801 and 214/256 copies reach 0.95, versus 0.957366 and 208/256 for the
parent. All held copies have non-stop decisions. The unchanged physical gate
passes, although the sampled copied exclamation mark remains visibly damaged.

A new hash-ordered cohort reserves 32 questions from sixteen contexts excluded
from training, earlier development and the first reserved cohort. All inputs
are inspected before fitting; answer coverage and teacher success do not
filter selection. The earlier reserved cohort is now development evidence and
is not recycled as a fresh test. Both complete 72-stream conditions are
actually reviewed before either new reference score; nineteen uncertain
stopped rows are additionally enlarged without altering their pixels.

Clear complete literal matches are 18/32 for the balanced writer and 19/32 for
its unchanged parent, with both questions correct in 5/16 and 6/16 contexts.
Eighteen clear correct answers are shared. The nineteenth is readable in the
parent but has a damaged initial glyph in the refined output; this conservative
agent-reading difference is not an established semantic regression. Canonical
complete image matches remain two each. Natural stops increase from 21/32
to 24/32 and caps fall from eleven to eight, but neither answer pilot passes.
One question lacks a fully supported canonical reference and remains in all
denominators. All 144 streams and 52 review boards are retained (Fig.~S42).

On the reused sixteen normal development questions, clear matches rise from
six to seven while canonical exact images fall from two to one. Both models
give zero clear correct answers after question erasure. All eight training
controls become canonical exact under refinement. A cold standalone replay on
the new cohort's prospectively fixed first input reproduces all emitted pixels,
states, stopping and feedback, without prohibited file access or IP connection.
This one execution does not prove arbitrary correctness.

The experiment resolves a measured training-fit failure while leaving
new-context language accuracy unchanged within this small trial. Passing a
copying or reference-conditioned reconstruction gate does not establish useful
autonomous language. Sequence alignment and learning on actual generated
histories remain separate tests; further fitting of G on the same fixed states
is not supported as the sole repair.

\subsection*{Fixed line framing does not repair source advancement}
A separate source-head-only probe requests one character per line with a
newline after every character. It uses the original body, the same eight
training and sixteen earlier normal development images, and three feedback
routes. Native source units give 7/16 complete literal development matches;
exact-character-plus-LF and canonical-image-plus-LF feedback each give 1/16.
All initial states match within each image across routes. Compilation checks
4,096 primitive character/source-ID pairs and retains forty two-unit codec
exceptions. No native output obeys the strict final-LF format. Unsupported
events and caps remain failures. The preset mechanical criterion fails, so
this framing is rejected. The diagnostic executes no optimizer update,
generates no learned answer image and changes no deployed control vector.

\subsection*{Retrospective first-policy separation}
Projecting all saved first-answer states through the original lexical matrix
provides another offline instrument. On the first reserved cohort, its first
decoded character matches an allowed reference first character for 30/32
adapted states and 31/32 readout-only states. Actual first-image F1 reaches
0.95 for 14/32 and 11/32, with five and three canonical exact first images.
On the second cohort, the unchanged parent and balanced writer have identical
first states and 29/32 source first-character matches. Their physical first
images reach 0.95 for 16/32 and 17/32, with five and six exact first images.
For the balanced writer, twelve cases have a matching source first character
but fail the physical threshold.

This is retrospective analysis of already scored states, not new held-out
evidence. It generates no new image or training example. The original lexical
matrix acts on actual possibly adapted states and remains outside deployment.
CPU FP32 accumulation after BF16 matrix rounding can differ from GPU arithmetic;
the smallest recorded top-two margin is 0.15937, and CPU/GPU winner equivalence
is not independently proved. A first-character match is not complete-answer
accuracy, and physical F1 is not a Unicode identity classifier. The gap supports
testing broader multi-unit function transfer for the image writer without
establishing that such a repair will work.

'''

NOTE=NOTE.replace('No result from that\nsubsequent repair is included here.',
                  'The following refinement tests that training-fit hypothesis directly.')
NOTE=NOTE.replace('\\clearpage\n\\plate{Figure_S41_contextual_image_answers}',
                  REFINEMENT_NOTE+'\\clearpage\n\\plate{Figure_S41_contextual_image_answers}',1)
NOTE+=r'''\clearpage
\plate{Figure_S42_balanced_writer_transfer}{\textbf{Training fidelity and new-context answer transfer.}
\textbf{a}, Clear complete literal matches on a second reserved cohort,
comparing the unchanged contextual parent with the balanced writer.
The one-case difference reflects an uncertain initial glyph; neither passes
the answer pilot. \textbf{b}, Training F1 improves and the physical gate passes,
while canonical exact complete new answers remain two in both conditions.
\textbf{c}, The refinement freezes the reader and fits only writer/stopping
on exact cached states with balanced first/later exposure. Inference continues
to feed actual generated pixels back through E. \textbf{d}, The first eight new
questions for both conditions in fixed index order. Labelled prefix viewports
retain full raw PNG bytes in the editable deck; capped and damaged outcomes
remain visible. All 144 new streams are archived. These are different contexts
from Fig.~S41, and all inherited model/training costs remain disclosed.}{fig:balanced-writer-transfer}
'''
