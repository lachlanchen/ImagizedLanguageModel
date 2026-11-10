"""Completed Note 61 results; contextual adaptation has no answer result yet."""

RESULTS = r'''\subsection*{Improved glyph copying does not restore complete-answer generation}
Training on genuine copy-prompt states separates physical writing from language
decisions (Supplementary Note 61; Fig.~S40). Removing source-anchor regression from the
writer objective improves held-wording glyph F1 from 0.951778 to 0.972856 and
the number reaching F1 0.95 from 194/256 to 229/256. Complete development-answer
transfer nevertheless fails: the resulting image-feedback model gives no
canonical exact answer among sixteen reused questions. On a fresh C3 cohort,
neither the original nor the improved writer produces a clearly correct
complete answer phrase among 32 conditions. A separate source-policy diagnostic
isolates a sequence-interface problem: with identical image prefills, native
source-unit feedback gives twelve literal matches among sixteen questions,
whereas both exact-character and image-encoded character feedback give three.
These helper trajectories use a source lexical head and are not ILM answers.
The comparison motivates adapting sequence dynamics, but does not establish
that frozen contextual states cannot support an independently trained writer.

'''

NOTE = r'''\clearpage
\section*{Supplementary Note 61 | Copying, feedback granularity and complete answers}

\textbf{Scope.}
These experiments retain the original complete-glyph checkpoint as the
language baseline. The copy curriculum changes its contextual writer; the
separate feedback diagnostic changes how a source policy advances. Their
denominators, resources and capabilities must not be pooled. Every reported
image answer is the actual generated binary pixel stream, with no glyph
substitution or nearest-font repair. Conservative transcription is an agent
reading, not independent human grading. Source pretraining exposure is unknown.

\subsection*{A matched writing objective improves copying}
The genuine-copy cache contains 8,172 fitting states and 256 held-wording states,
covering 4,086 nonempty known forms. The held wording is a different instruction
template, not an unseen glyph-identity test. Another 3,855 original reference
history states provide rehearsal. Fitting changes the 16,791,552-parameter G
writer alone for 18,000 updates; the body, image encoder E, decoder D and
stopping head stay fixed. With source-anchor MSE, cosine and pixel losses,
held-wording foreground F1 is 0.951778, but only 194/256 cases reach 0.95,
below the fixed 80\% requirement. Six held images are canonical pixel-exact.

Removing the source-anchor losses, while retaining the same architecture,
data, seed and schedule, gives F1 0.972856, 229/256 cases reaching 0.95, and
42/256 canonical exact images. It passes the unchanged physical requirement.
The two fits take 101.77 and 90.54\,s, respectively, excluding their common
9,651.28\,s genuine-state cache and inherited model pretraining. This is a
writer-objective comparison, not evidence that language knowledge was learned
in ninety seconds. The unexecuted G-plus-decoder arm fails resource admission
before fitting and supplies no algorithmic result.

The pixel-only writer then generates all forty original/control streams.
Eight training controls are canonical exact, but the sixteen normal development
questions give zero canonical exact answers, mean foreground F1 0.341013,
five natural stops and eleven caps. The preserved baseline gives one canonical
exact answer, F1 0.354426, and the same stop/cap counts. Its restricted year-edit
evidence remains separate. Stronger held-wording copying has not solved the
complete-answer task.

\subsection*{A fresh multiple-choice cohort still fails phrase generation}
Sixteen C3 documents provide original and rotated option orders under two
instructions: emit an option letter, or emit its complete answer phrase.
Both fixed checkpoints generate all 64 conditions, giving 128 actual image
streams. The evaluator inspects every output before reading current reference
strings, including enlarged views of ambiguous short outputs. The source's
noncommercial research terms accompany the archived outputs; source input
pages and reference strings are not redistributed in that archive.

For letter instructions, conservative clear complete reference matches are
6/32 for the baseline and 11/32 for the pixel-only writer; both option orders
are correct for 1/16 and 3/16 documents, respectively. Canonical exact counts
are zero in both conditions. For complete phrases, both models have zero
canonical exact and zero clear complete reference matches among 32 conditions.
The baseline stops in 4/32 phrase conditions and caps in 28; the pixel-only
writer stops in 8/32 and caps in 24. The letter counts are correlated conditions
over sixteen documents, not evidence of independent general reading accuracy.
Malformed strokes and uncertain readings remain failures; no evaluator repairs
a missing crossbar or infers an answer from the question image.

\subsection*{Source-unit and glyph feedback induce different trajectories}
The helper comparison uses eight training controls and sixteen reused normal
development questions, each under three routes. The prompt geometry and first
contextual state are identical across routes, with checked maximum absolute
prefill difference zero. The same native lexical decision head selects the
next source unit. Native-unit feedback supplies its complete embedding;
the exact-character route supplies the source embedding of the first complete
character; the image route supplies E of that character's canonical image.
Partial-Unicode or unsupported winners terminate explicitly, without fallback.

On the sixteen development questions, literal complete reference matches are
12/16 with native units, 3/16 with exact-character vectors and 3/16 with canonical
image feedback. All native routes terminate. Exact-character feedback has nine
terminal events, five unsupported events and two caps; image feedback has ten,
four and two, respectively. The equal match counts do not imply equal paths.
Each route gives five literal matches among the eight training controls.
All 72 trajectories and failures are retained. The helper takes 353.62\,s,
including 40.04\,s pressure waiting, and peaks at 20.041\,GiB allocated memory.
It executes no optimizer update and generates no learned ILM answer image.

A source unit can contain several characters. Replacing its feedback with
one character changes both represented content and the sequence's advancement.
The exact-character intervention therefore shows that encoder approximation
alone cannot explain this diagnostic's loss of performance. It does not prove
an information-theoretic ceiling for a separately trained glyph policy, nor
does it establish that any particular body adapter will repair that policy.
No diagnostic trajectory is reused as a training example.

\subsection*{Exact frozen-prefix factorization for a prospective repair}
The next candidate changes rank-8 q/v/o adapters only in the final four layers
of the actual 36-layer source body, jointly with G and the stopping head.
The frozen first 32 layers, visual reader, E and D remain unchanged. For a
reference-image history, let
\[
 H_{32}=F_{0:31}(V(x),E(y_{<t})),\qquad
 h_t=F_{32:35,\theta}(H_{32})_t.
\]
Because no trained parameter enters $H_{32}$, caching it preserves the gradient
with respect to $\theta$ exactly. All prefix positions and rotary coordinates
are retained; final-layer attention is recomputed over the complete causal
prefix. Caching final states of changing layers would not have this property.
The fit uses balanced image reconstruction through D and a separately balanced
terminal/nonterminal loss, without source-anchor regression.

The corrected preflight checks zero maximum split-replay error and zero earlier
state change when a future reference glyph is perturbed. Two real optimizer
updates change contextual adapter-up and writer parameters, with finite nonzero
adapter-up gradients and unchanged frozen parameter versions. There are 688,128
trained contextual and 17,848,833 trained writer/stopping parameters; peak
allocation for this two-update check is 3.208\,GiB. These updates are discarded.
An initial static protocol count of 32 total layers is preserved as a correction;
the implementation dynamically selected the correct final four throughout.
The prospective main candidate uses 3,176 admitted training questions and a
fixed 18,000-step schedule, with a matched broader-data readout-only control.
No main-fit complete-answer result is reported in this note. Gradient flow,
physical copying and a small trainable parameter count do not establish an
independent lightweight language model or eliminate inherited pretraining cost.

\clearpage
\plate{Figure_S40_copying_and_feedback_steps}{\textbf{Copying, source steps and complete-answer failure.}
\textbf{a}, Counts of held-wording known-form copies reaching foreground F1
0.95 under matched writer objectives; the fixed gate requires at least
205/256 cases. \textbf{b}, Literal matches under three feedback routes with
identical image prefills. This source-head helper is not the image-only ILM
runtime. Its output is not used for training, and equal counts do not imply
identical trajectories. \textbf{c}, Actual C3 outputs for fixed documents
0, 5, 10 and 15, retaining both models, option orders and instruction styles.
Where a stream exceeds ten glyphs, the display explicitly marks its first-ten
prefix and reports the full length and cap/stop outcome. The editable deck
retains the complete, unchanged output PNG. No stroke or glyph is replaced.
Neither model gives a clear correct complete phrase among all 32 phrase
conditions. Agent transcription and canonical exact scores are separate;
no independent human grade or contextual-adaptation answer success is claimed.}{fig:copying-feedback-steps}
'''
