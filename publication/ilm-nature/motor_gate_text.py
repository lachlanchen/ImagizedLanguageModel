"""Completed motor/geometry controls; no unfinished curriculum result included."""
RESULTS=r'''\subsection*{Sharp known-form generation does not recover uncertain glyph identity}
Additional controls distinguish a useful motor from correct conditional
writing (Supplementary Note 60; Fig.~S39). On sixteen
normal Chinese initial states, an allowed reference first form is nearest
in semantic cosine for nine, but only one first image is pixel-exact.
A spectral calculation gives a 0.060109 lower bound on uniform anchor-code
MSE for a rank-2048 affine output; distance to the trained output span and
bias is 0.305180. These are regression limits, not pixel-error bounds.
Synthetic-state readout transfer and an added contrastive loss both fail
their physical glyph requirements. Noise-trained decoding and two joint
motor flows also fail fixed mixed-code tests. A 3.89M-parameter spatial
flow generates the 4,087 known clean-condition forms with mean foreground
F1 0.998260, but mixed-condition F1 is 0.810537. None of these rejected
motors undergoes autonomous language assessment. Confident known-form
generation remains distinct from recovering the intended form and producing
a correct complete answer.

'''

NOTE=r'''\clearpage
\section*{Supplementary Note 60 | Semantic output geometry and motor ambiguity}

\textbf{Scope.}
All controls in this note preserve the complete-glyph language checkpoint
reported in Supplementary Note 59. They use its existing 4,087 canonical
forms and offline source-derived semantic anchors. No unknown glyph
identity or historical script is tested. Source-model pretraining remains
part of the provenance. The small fitting costs below do not include
acquiring that language knowledge.

\subsection*{First-form proximity is not a correct answer image}
Recompute the original frozen readout and decoder on the saved first state
of each of forty language streams. Every nonempty saved first image replays
exactly. On the sixteen normal Chinese questions, an allowed reference
first form ranks first in cosine to the readout vector in 9/16 cases and
within the first three in 11/16; only 1/16 first images is pixel-exact.
The reference initial forms for four cases have rank one but physical
foreground F1 of approximately 0.60--0.64. This bank-based ranking is an
evaluator-only diagnostic: it never selects, repairs or replaces an
image produced by the runtime. It is neither complete-answer accuracy nor
an independently graded semantic score.

\subsection*{An affine output span limits exact anchor regression}
The original readout ends in
\[
G(h)=B\phi(h)+b,\qquad B\in\mathbb{R}^{4096\times2048}.
\]
Every output lies in the bias plus a subspace of dimension at most 2048.
Let $z_c\in\mathbb{R}^d$ be the normalized training anchor for form $c$,
with $d=4096$, $M=4087$, and centered covariance
\[
C=\frac{1}{M}\sum_{c=1}^M(z_c-\bar z)(z_c-\bar z)^\top.
\]
For eigenvalues $\lambda_1\geq\cdots\geq\lambda_d\geq0$, the optimal
rank-$r$ affine approximation satisfies
\[
\inf_{a,\,\dim U\leq r}\frac{1}{Md}\sum_c
 \left\|z_c-\Pi_{a+U}z_c\right\|^2
 =\frac{1}{d}\sum_{j=r+1}^{d}\lambda_j.
\]
A double-precision CPU calculation gives minimum uniform per-coordinate
MSE 0.060109 at rank 2048, capturing at most 93.9876\% of anchor variance.
Rank 3072 reduces this bound to 0.005000. The minimum distance to the
\emph{actual frozen trained span and its bias} is 0.305180. This is not
G's prediction error on a language dataset; it measures the wider anchor
set's distance to a fixed affine output space. The calculation takes 5.60\,s
on four CPU threads, with no neural forward call or CUDA initialization.
A nonlinear decoder may still produce useful images from a lower-dimensional
space, so these bounds do not prove that rank caused the observed pixel or
language failures.

\subsection*{Broader teacher-function coverage does not qualify the readout}
A temporary native output matrix labels genuine training histories and
validates synthetic hidden-state calibration. For each primitive form, a
bounded sequence of diagonal-variance half-space corrections forces its
single-source-unit anchor to be the actual full-head winner. All 20,435
requested samples pass that final winner check: four fitting bases and one
held-out base per form. Median standardized RMS displacement is 0.151012,
maximum 0.323332. These are constructed hidden vectors, not additional
passages or language examples. No development state enters fitting.

Train only G for 6,000 batch-128 AdamW updates at $2\times10^{-4}$, using
semantic MSE, cosine discrepancy weighted 0.2 and balanced pixel BCE through
frozen D. Half each batch consists of accepted synthetic states; half uses
genuine teacher-supported states. On 4,087 held-out synthetic states,
mean physical glyph F1 rises from 0.333969 to 0.522572, but none of the
candidate images is pixel-exact. On 774 genuine states withheld from these
additional updates, F1 falls from 0.896007 to 0.829770 and exact glyphs fall
from 638 to 180. The initial checkpoint already trained on all 508 original
questions, including these histories: this latter route measures retention,
not independently held-out language generalization.

A loss-only variant restarts from the same original checkpoint and adds
cosine-anchor cross-entropy at temperature 0.07 and weight one. With the
same data, sampler and 6,000 updates, it gives F1 0.489500 on synthetic
states and 0.795106 on genuine states, with 115 genuine glyphs exact.
Neither variant meets the prospective requirements of mean F1 at least
0.95 and at least 80\% of forms reaching F1 0.95 on both routes.
An independent cold replay reproduces all 9,722 physical measurements for
each variant bit-exactly. Their fixed raw-image sheets contain damaged
composites; no lexical correction is applied.

\subsection*{Mapped native agreement along gold image histories}
An offline audit reuses the earlier native-head winners on all 4,363 original
training-history states, verifying their exact alignment with the causal
cache. A winner is mapped to its first complete form, a terminal label or an
unsupported label. Reference-form agreement is 368/508 (72.44\%) before the
first glyph and 2632/3347 (78.64\%) across continuation states; terminal
agreement is 434/508 (85.43\%). After one gold glyph, next-form agreement is
328/507 (64.69\%); after two or three it is 756/891 (84.85\%). Later positions
represent different question subsets, whose first-state baselines are retained
in the report. No neural forward or new teacher call is made by this audit.

All mapped reference forms and the stop label agree in 138/508 gold-history
sequences. Correct preceding glyph images were supplied at every position:
these are not 138 autonomous answers. Alternative wording, formatting,
source-unit segmentation and image-feedback errors remain possible causes of
disagreement. No matched exact-source-embedding counterfactual is evaluated.
The pattern does not establish cumulative feedback damage as a sufficient
explanation, nor does it establish reliable image encoding or semantic accuracy.
It leaves the genuine-input readout intervention and all its gates unchanged.

\subsection*{Motor formation under known clean and mixed conditions}
The original D is accurate near individual anchors. A decoder-only control
trains 10,792,449 parameters for 6,000 batch-128 updates, leaving G, E, affine
buffers and stop exactly fixed. Training corruptions comprise Gaussian noise,
two unit-RMS-anchor mixtures and exact anchors. Random-noise assessment
at standard deviation 0.6 improves mean F1 from 0.965706 to 0.992049, but
hard mixed conditions still fail. A mixture condition combines a target's
unit-RMS direction at weight 0.55 with a different form at 0.45, with fixed
inventory offset 1049. This controlled stress test is not a model of every
malformed language readout vector.

Two subsequent generators use conditional joint flow matching with unweighted
velocity MSE, smoothed endpoints and final EMA weights. The 168,366,080-parameter
dense flow evolves all 4,096 motor coordinates, then uses the original D;
it uses 32 integration intervals. The 3,892,737-parameter spatial flow reuses
a 32-pixel U-Net, with 32 base channels and 256 context channels, conditioned
on the semantic vector and the original D's continuous ink plan; it uses
16 intervals. Both train for 8,000 batch-128 updates. Each generates four
candidates, re-encodes their actual binary pixels with E, and selects by
cosine to the condition. No reference or runtime prototype bank selects an
image. Source-body weights are not loaded for these motor fits.

\begin{center}\small
\begin{tabular}{lrrr}
\hline
Motor & Clean F1 & Mixed F1 & Fitting time (s)\\
\hline
Original D & 0.999940 & 0.761720 & ---\\
Noise-trained D & 0.999752 & 0.822751 & 26.60\\
Dense motor flow & 0.906284 & 0.715449 & 129.67\\
Spatial image flow & 0.998260 & 0.810537 & 136.35\\
\hline
\end{tabular}
\end{center}
Each row assesses all 4,087 known forms. The original-decoder mixed output
bitmaps are verified identical across these controls, although conditioning
normalizations and training recipes differ. These are not matched-compute
learning comparisons. The dense and spatial generators' reference-best
candidate F1 on mixtures is only 0.735920 and 0.823132, respectively; those
oracle diagnostics never select actual output. Poor generation is therefore
not explained by the learned selection rule alone.

The spatial motor passes its clean-condition requirement: all 4,087 forms
exceed F1 0.95 and 2,314 are pixel-exact. All three candidate motors fail
their prospective mixed-condition requirements, and no autonomous language
assessment uses them. Their peak fitting allocations are 0.548, 3.313 and
0.889\,GiB, respectively, excluding external pretraining and the language body.
All candidate pixels and errors remain available. The fixed 64-triplet sheets
were actually viewed; individual human grades for every glyph are not claimed.

A separate zero-update check applies four anchored E/D passes to the forty
original first states. The original decoder's exact normal first forms rise
from one to three, but the five cases reaching F1 0.95 do not increase and
mean F1 slightly decreases. The noise-trained decoder also leaves that count
at five. All 160 first-state outputs were viewed. Sharpening a few digits
without producing more correct first forms does not establish language
improvement.

\subsection*{Reproduction and interpretation}
Protocols, source hashes, numerical measurements and every raw output are
retained under \path{references/evidence/semantic_readout_distillation_v1/},
\path{references/evidence/contrastive_glyph_readout_v1/},
\path{references/evidence/robust_glyph_decoder_v1/},
\path{references/evidence/semantic_motor_flow_v1/} and
\path{references/evidence/spatial_semantic_motor_v1/}.
The fixed-state and spectral diagnostics are in
\path{references/evidence/semantic_glyph_geometry/},
\path{references/evidence/glyph_motor_states_v1/} and
\path{references/evidence/glyph_readout_rank_v1/}.
The cached-label audit retains every state and question summary under
\path{references/evidence/cached_glyph_teacher_audit_v1/}.
These controls reject the specified recipes, not every possible transfer,
normalizing flow or pixel model. They motivate real contextual coverage and
continued autonomous-image validation, while leaving broad language generation,
unknown writing identities and historical-script understanding unresolved.

\clearpage
\plate{Figure_S39_motor_uncertainty}{\textbf{Known-form motor accuracy is distinct from recovering uncertain glyph identity.}
\textbf{a}, Evaluator-only first-form diagnostics on the same sixteen normal
Chinese initial states: nine nearest allowed reference forms versus one
pixel-exact first image. No nearest-form operation is deployed.
\textbf{b}, Minimum uniform anchor-code MSE in the frozen trained output span
and bias, compared with the best rank-2048 affine span. This is not a pixel
or language-error bound. \textbf{c}, Physical F1 over all 4,087 known forms
for three separately trained motors. Every mixed-condition gate fails; costs
and recipes differ. \textbf{d}, Complete unrepaired 32-pixel cells at four
fixed, evenly spaced non-empty inventory indices. The original D, noise-trained
D, selected dense-flow and selected spatial-flow outputs use the declared
55/45 mixtures. Source targets are actual canonical bitmaps, not replacement
transcriptions. Clean known-form success does not qualify complete answers.}{fig:motor-uncertainty}
'''
