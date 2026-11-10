"""Completed failure-localization results; no language-repair success claim."""

SUMMARY = (
    'Written language includes historical and handwritten forms outside fixed character inventories. '
    'We investigate independent image-input, image-output language generation on one consumer GPU, '
    'separating reconstruction, conditional preference and autonomous answering. '
    'A normalized causal pixel writer above an externally pretrained 7.544B visual contextual body '
    'fits 495 of 512 Chinese training answers exactly after 16,000 updates, but none of 24 normal '
    'development answers. All supplied correct development first images have lower likelihood '
    'than the corresponding erroneous emission, despite transferred paired-question preferences. '
    'A matched diagnostic reader reproduces all 32 image-memory fingerprints and returns literal '
    'reference answers on 12 of 16 normal Chinese questions, versus zero with the question erased. '
    'This helper-only text baseline localizes substantial lost capability to the learned writing interface; '
    'it is not an image-output solution. A continuous complete-glyph loop then produces some readable '
    'new short answers, but eleven of sixteen normal development streams still cap. On two selected '
    'question templates, eleven of fourteen prospective date changes yield clearly matching answer '
    'images, with two pixel-exact; all fourteen stop naturally. These non-independent manual readings '
    'support restricted input-dependent composition, not general language qualification. '
    'Historical-script understanding and an efficiency advantage at matched language quality remain unresolved.'
)

ARTICLE_REPLACEMENTS = [
    ('The present study uses one canonical font to isolate\nthis coupling before introducing handwriting or historical scripts.\n',
     'The present study uses one canonical font to isolate\nthis coupling before introducing handwriting or historical scripts.\n\n'
     'The experiments progress from initial raster readers and denoisers to\n'
     'an inherited visual/contextual body and a separately learned glyph\n'
     'interface. The earlier architectures provide controls; the final\n'
     'experiments evaluate the complete-glyph loop shown in Fig.~S37.\n'),
    (r'\subsection*{A continuous visual interface preserves the written signal}',
     r'\subsection*{An initial continuous visual interface preserves the written signal}'),
    ('Each observed cell is a $32\\times32$ ink image.',
     'In the initial control, each observed cell is a $32\\times32$ ink image.'),
    (r'\textbf{A visual language interface and its conditional generation test.}',
     r'\textbf{Initial visual language interface and its conditional generation test.}'),
    ('The schematic\nshows the implemented experimental contract, without claiming that its\ngenerated content is already correct.',
     'The schematic\nshows the initial denoising control. Its generated content remains incorrect;\n'
     'the later complete-glyph runtime is shown in Fig.~S37.'),
    ('''Complete pixel learning reproduces all 32 seen answers, but none of 24
normal development answers. Further unchanged fitting of those questions is
therefore unwarranted. Varied visual-language supervision must be tested against
readable, correct new-prompt answers, retaining independent image generation
and accounting for external pretraining.
''', '''Broader pixel training reproduces 495 of 512 answers but still fails the
supplied unseen reference images. The matched native reader retains useful
answer capability, directing the intervention toward the learned writing
interface. The complete-glyph loop supports restricted input-dependent date
composition, with actual-image feedback independently replayed. It remains
unreliable on broader questions: eleven of sixteen normal Chinese streams cap.
Wider semantic readout coverage and robust generated histories remain
necessary; clear familiar glyphs and low training loss are insufficient.
''')]

RESULTS = r'''\subsection*{The learned writing interface loses usable reading capability}
Increasing the cohort from 32 to 512 questions gives 495/512 complete
pixel-exact training answers after 16,000 updates, but no exact normal
development answers (0/16 Chinese and 0/8 English; Supplementary Note 58,
Fig.~S36). All 512 training first images are exact; none of the 24 normal
development first images matches a reference. The mismatch is therefore
already present before feedback from an earlier complete generated image,
although within-image autoregression remains a possible contributor.

Scoring fixed owning checkpoints separates an erroneous learned distribution
from a decoding-only explanation. Every supplied correct development first
image is less likely than the corresponding saved emission. At the final
state, the mean best-reference minus emitted-image log probability is
$-751.24$\,nat for Chinese and $-941.82$\,nat for English. A maximum-likelihood
decoder cannot select any of these supplied references over the existing
emission. This does not rule out other correct wordings or fonts. Relative
question conditioning nevertheless transfers: 14/16 Chinese reference events
favor their owning question and all eight paired assignment contrasts are
positive. Relative preference alone does not establish useful output.

The original source reader, using the same normal and question-erased PNGs,
reproduces all 32 ILM image-memory fingerprints. A temporary helper-only
source text head then gives 12/16 literal normal answers, zero erased answers
and 11/16 source-text answers. The two date questions receive the correct
years; the learned pixel writer's corresponding forms remain damaged.
Some literal mismatches are aliases or longer relevant wording. These are
reused development cases without independent semantic grading, and the
symbolic helper generates no ILM images. The comparison localizes substantial
lost capability to the writing interface without uniquely separating its
initial query, conditional density and image-history projection.

We consequently separate glyph formation from language continuation
(Fig.~S37). A continuous CNN encoder and decoder learn training-only semantic
anchors for 4,087 distinct canonical forms. Direct decoding is exact for
4,048 forms (foreground F1 0.999940); actual-image roundtrip is exact for
3,697 (F1 0.999634). The 80.22-second primitive run peaks at 0.728\,GiB,
excluding the large contextual body and its pretraining. Semantic alignment
is weaker than reconstruction, with mean cosine 0.77735 on the first 128
forms. Clear reconstruction neither validates unseen glyphs nor demonstrates
language understanding. The proposed continuation loop preserves the native
assistant state, predicts one complete glyph and re-encodes its actual pixels.
Fitting only its readout/stopping on 508 supported questions yields readable
new complete strings corresponding to 288 items and the year 1874
(Supplementary Note 59). Nevertheless, only 1/16 normal Chinese development
images is canonical-exact, and eleven cap; the one exact full year string
occurs in training. These are partial examples, not a semantic-accuracy estimate.

A frozen intervention replaces passage dates on two templates selected from
those outputs (Fig.~S38). Eleven of fourteen prospectively fixed new dates
have clear matching output-first transcriptions, two are canonical-exact and
all fourteen stop naturally without a target length. Three damaged/wrong
digit streams remain failures. Erased-question controls give no correct date.
Six clear outputs compose full year strings absent from supported training
answers. This supports restricted input-dependent composition through actual
image feedback, while broader unseen-answer reliability remains unresolved.
'''

NOTE = r'''\clearpage
\section*{Supplementary Note 58 | Localizing the image-writing failure}
\label{note:writing-failure}
The larger cohort contains 512 Chinese questions from 256 passages, with
1,657 complete answer blocks and 498 distinct full answer images. It reuses
the earlier 32 controls and adds 480 source-hash-selected questions. The
architecture and externally pretrained 7,543,984,691-parameter body are
unchanged; 13,855,681 parameters train. Batch four retains 64,000 draws over
16,000 updates. Each question has 124 or 128 exposures. The original
initialization, schedules, source-image bindings and numeric task vectors
are fixed before execution. The 24 Chinese/English development cases are
reused rather than sealed, and English is untrained transfer.

Final complete-answer foreground F1 is 0.978044 on the original 32 controls,
0.990639 on the 480 additions, 0.336570 on normal Chinese development and
0.059018 on English. Chinese and English erased controls give 0.225725 and
0.040126. Both training answers are exact for 239/256 passage pairs. The same
64 training questions assessed throughout yield 0, 3, 21 and 60 exact answers
at updates 0, 4,000, 8,000 and 16,000. Whole-training correct-history image
NLL falls from 9.749791 to 0.251025, 0.020290 and 0.000259; this averages
piece losses within blocks and questions, not an unscaled answer likelihood.

A resource guard ends the original process after 12,128 actual updates.
Recovery resumes the durable update-12,000 checkpoint, with optimizer,
sampler and RNG states intact. All 128 discarded per-update history records
and bytes replay exactly, but full discarded model/Adam tensor identity is
not independently established. The ledger therefore records 16,128 actual
optimizer calls. Cumulative primary active time is 42,549.17\,s, including
both processes and assessments; the completed process peaks at
18.138\,GiB. That peak is process-local rather than an independently measured
maximum over both processes. External pretraining and independent audit
compute are outside this timing.

An independent CPU audit authenticates all 896 streams from four assessments,
their PNGs and stopping events, and 2,731 actual-image feedback records.
Maximum CPU/GPU feedback error is $7.57\times10^{-6}$, below $10^{-4}$.
All 32 original control prompt prefixes remain bit-exact. Numerical and
recording consistency does not establish semantics. All 24 normal and
24 erased final development images are inspected. Source-first notes precede
the current reference comparison, but the cases are not fully blinded or
independently graded. All 560 final raw outputs and 140 display sheets remain
available, including failures. No malformed strokes are repaired.

\subsection*{Fixed-state normalized image likelihood}
For a complete first block $I$ with 256 physical eight-bit pieces,
\[
\ell(I\mid q)=\sum_{j=1}^{256}\log p(u_j\mid h(q),u_{<j}).
\]
At each owning state, compare the maximum likelihood of an individual allowed
reference with the likelihood of the actual saved emission. Do not substitute
the log-sum probability of multiple references for this MAP witness. The
same-question pre-image continuation factor cancels. No supplied development
reference wins at any of the four states. Final normal-case differences
range from $-1651.68$ to $-6.05$\,nat. This establishes a failure of the learned
first-image distribution relative to these references, not a guarantee that
all other correct renderings are impossible.

Owning-question preferences rise from 7/16 to 12/16, 13/16 and 14/16.
Positive paired assignments rise from 2/8 to 8/8, 8/8 and 8/8. All eight
paired first-image reference events are disjoint. Additive question-independent
event terms cancel in the paired contrast; an image prior inside a
multiple-reference log-sum need not cancel separately. The diagnostic makes
112 fresh PNG prefills, 448 writing queries and 896 physical-head forward
calls, with no optimizer updates or new answer generation. All reader
fingerprints and initial stop probabilities match; every saved nonempty
first image reproduces all 256 original greedy decisions. Runtime is
299.82\,s at 17.999\,GiB peak allocation.

\subsection*{Matched native reader}
Run the unmodified source reader on all sixteen Chinese normal PNGs,
question-erased PNGs and original source-text prompts. Numeric controls and
visual/contextual computation match the image actor; source adapter outputs
are zero. All 32 image-memory fingerprints match the actual ILM path before
decoding. A helper-local source projection and previous-symbol embedding
lookup produce 48 text responses, capped at 64 source units. They are never
attached as runtime image-writer components. All normal and text streams
stop; one erased stream reaches the cap. Literal counts are 12/16, 0/16 and
11/16 for normal, erased and text routes. Raw aliases and longer answers
are retained without assigning formal semantic scores. The helper uses
233.37\,s including 40.04\,s of pressure waiting, at 18.970\,GiB peak allocation.
Its outputs and hidden states are prohibited as development training data.

\subsection*{A continuous semantic glyph primitive}
Training-only source character frequency and ASCII supply 4,096 anchored
characters; nine identical bitmap aliases merge, leaving 4,087 forms. The
initial anchor requires font coverage and an exact single-source-unit round
trip. The canonical Noto CJK form uses a common baseline on a binary
$32\times32$ canvas. Font and source IDs occur only in preparation.
The CNN encoder E, convolutional decoder D, continuous contextual readout G
and stop predictor contain 47,127,810 neural parameters in total. Only E and
D train in this primitive stage. The checkpoint contains global affine
statistics but no character-indexed prototype or image table.

Let $v_c$ be a training-only inherited input vector and $z_c$ its normalized
coordinate. E learns MSE and cosine alignment to $z_c$ from one-pixel-shifted
images. D reconstructs canonical pixels from $z_c$ plus small Gaussian noise
and from E's predictions. Foreground/background-balanced BCE weights ink
and background separately; it is a reconstruction loss rather than a
normalized full language likelihood. AdamW uses 6,000 batch-128 updates,
learning rate $3\times10^{-4}$ with cosine decay to one tenth, FP32 parameters
and BF16 small-network autocast. Seed and all source bindings are retained.

The completed run takes 80.22\,s including 20.02\,s of pressure waiting,
peaking at 0.728\,GiB. Direct and roundtrip mean F1 exceed 0.999, but exactness
and semantic alignment remain imperfect. Minimum F1 is 0.974359 for direct
decoding and 0.956835 for roundtrip. The displayed 64 indices are evenly
spaced, with unrepaired target/direct/roundtrip triplets. This is seen-identity,
single-font rendering, not historical or unseen-character generalization.

The evaluated language loop is
\begin{align*}
h_0&=T(I_{\rm prompt}),\qquad z_t=G(h_t),\\
I_t&={\bf1}[\sigma(D(z_t))\geq\tfrac12],\\
h_{t+1}&=T(E(I_t);\mathrm{KV}_{\leq t}),
\end{align*}
with the global affine restoration of E's normalized semantic coordinates
before contextual input. A learned stop receives no reference length. The
native assistant state replaces the earlier learned initial writing query.
The primitive-only checkpoint explicitly refuses language generation.
Supplementary Note 59 distinguishes its reconstruction from the subsequent
language fitting and frozen intervention.

\clearpage
\plate{Figure_S36_writing_failure}{\textbf{Training fit does not recover useful unseen writing.}
\textbf{a}, Exact complete images on the same 64 training examples at four
states; all 24 normal development answers remain inexact. \textbf{b}, Mean
best-reference minus actual-emission first-image log probability for sixteen
Chinese cases, in nats. All supplied references lose. \textbf{c}, Matched
native-reader literal counts, with all 32 image-memory fingerprints identical
to the ILM path. This is a helper text baseline, not image generation.
\textbf{d}, Two unrepaired final date-like pixel outputs; the helper's correct
years do not repair them. Relative question preference, literal text matching
and pixel exactness have distinct denominators and interpretations.}{fig:writing-failure}

\clearpage
\plate{Figure_S37_semantic_glyph_loop}{\textbf{Clear glyph formation precedes language validation.}
\textbf{a}, A continuous readout predicts one complete glyph from the native
assistant state; E re-encodes the actual emitted pixels before continuation.
No runtime glyph table, tokenizer, source output head or font renderer is
present. \textbf{b}, Primitive encoder/decoder training uses offline semantic
anchors and images; the later 508-question stage fits only readout/stopping.
\textbf{c}, Target/direct/roundtrip triplets at 64 evenly spaced indices from
4,087 prepared forms. Direct reconstruction is 4,048/4,087 exact. Sharp
familiar glyphs do not establish answer accuracy or unseen forms. The broader
Chinese assessment still has eleven capped failures.}{fig:semantic-glyph-loop}
'''

NOTE += r'''\clearpage
\section*{Supplementary Note 59 | Partial glyph-language transfer and changed dates}
\label{note:glyph-language}
Four training questions contain unsupported anchor characters and are
explicitly recorded rather than substituted; development cases are not
filtered. The 508 supported questions supply 3,855 gold glyph events and
508 terminal states. The contextual body and primitive encoder/decoder stay
frozen. Cache the original FP32 assistant state, followed by native causal
states after E encodes each gold previous glyph image. The current target
never enters the state predicting it. Bulk and one-glyph-at-a-time states
agree within $2.29\times10^{-5}$ on the first sequence. Caching takes
731.67\,s at 17.789\,GiB peak; all 4,363 states are retained by hash.

Train 17,848,833 readout/stopping parameters for 6,000 batch-128 AdamW updates,
learning rate $2\times10^{-4}$ with cosine decay to one tenth, and seed
202611102. Sample glyph and terminal events uniformly from the cache. The
loss sums semantic MSE, cosine discrepancy weighted 0.2, balanced pixel BCE
through D, and unweighted stop BCE. This is a reconstruction/continuation
objective, not a normalized language likelihood. Fit time is 42.47\,s at
0.658\,GiB peak, excluding native caching and inherited pretraining.

All eight fixed training controls are exact and stop. Normal Chinese
development gives 1/16 canonical-exact (F1 0.354426), five stops and eleven
caps; erased gives 0/16 (F1 0.075368), one stop and fifteen caps. All forty
raw streams are inspected before the current canonical-reference comparison.
Readable complete answers corresponding to 288 items and 1874 with Chinese
suffixes have F1 0.987315 and 0.986612, respectively; neither is pixel-exact.
Those full strings are absent from supported training answers. The exact
2001 year string appears in training. Canonical images are evaluator-only.
Damaged Latin forms and the channel-name response are not repaired or graded
correct. Autonomous assessment takes 332.59\,s, including 10.01\,s of pressure
waiting, at 17.646\,GiB peak. These are reused development data without
independent human grading.

For the frozen date test, choose the two previously observed year-question
parents, then fix all values before inference: 1874, 1936, 1648, 1789, 2001,
2007, 2019 and 1993. Replace every occurrence of the answer year in each
passage, preserving question ink and original wrapping. Verify exact original
rendered PNGs and unchanged strip width. Altered passages are counterfactual
reading conditions, not historical claims. Keep all sixteen normal and
sixteen erased images; fourteen normal conditions are new date interventions
and two are original baselines. The frozen actor receives only the image and
its unchanged 64-glyph safety cap, performs zero updates and never receives
the requested year or desired length.

All sixteen normal streams stop after five glyphs. Among fourteen new dates,
eleven have clear matching output-first manual transcriptions, two are
canonical-exact and mean F1 is 0.970408. Both 1789 variants have a damaged
second digit, and the coast 1648 variant has wrong/damaged digits; none is
corrected toward gold. Nine new conditions request complete strings absent
from supported training answers; six of these produce clear matches. No new
erased condition gives a correct date; thirteen cap and one stops after an
entity-name-like response. Mean erased F1 is 0.048683. The parents are
selected from prior outputs, and manual readings are non-independent; this
is not a sealed or general semantic benchmark. Runtime is 229.23\,s at
17.605\,GiB peak. Repeating the original two PNGs in this cold process
preserves complete pixels, stop probabilities and feedback hashes exactly.

A separate small-bridge audit re-encodes all 2,781 actual emitted glyphs from
the seventy-two language/intervention streams. Every cold GPU embedding hash
is bit-exact; maximum CPU/GPU error is $1.79\times10^{-7}$, below $10^{-4}$.
The audit loads no native contextual body or teacher, takes 30.80\,s and peaks
at 0.351\,GiB. Its 48\,GiB host-entry, 32\,GiB runtime and 5\,GiB GPU cap are
separate small-stage limits, not changes to completed training protocols.
These checks establish execution/feedback consistency, not semantic correctness.

All model weights and external pretraining remain part of the provenance.
The bridge's small fitting cost does not include acquiring the large body's
knowledge. Broad answer reliability, English, unknown glyph identities and
historical writing remain unvalidated. The evidence supports a restricted
image-input/image-output composition mechanism that can be improved, not a
general-purpose model equivalent to a contemporary LLM.

\clearpage
\plate{Figure_S38_date_interventions}{\textbf{A frozen glyph loop follows changed dates on two selected templates.}
\textbf{a}, Replace every answer-year occurrence in the passage, preserve
the question and geometry, and perform zero updates. Original PNG baselines
reproduce exactly. \textbf{b}, All sixteen normal outputs, including two
baselines, in fixed preparation order. Labels state requested years, not
transcriptions. Fourteen new date conditions give eleven clear output-first
readings, two canonical-exact images and fourteen natural stops. Both 1789
variants and the coast 1648 variant remain uncertain/wrong and count as
failures. All erased outputs are retained; none gives a correct date.
Parent selection uses prior successes and manual readings are non-independent,
limiting this result to exploratory input-dependent date composition.}{fig:date-interventions}
'''
