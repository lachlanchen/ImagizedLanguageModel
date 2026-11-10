# Next intervention: transfer the native decision geometry to glyph writing

The complete-glyph loop has a restricted working subset: it tracks eleven
of fourteen date changes on two selected templates, while eleven of sixteen
broader Chinese responses still cap. Preserve that verified checkpoint and
every failure. Do not resume unchanged density training or describe the
restricted test as a general language model.

The [target-coverage audit](evidence/semantic_glyph_readout_coverage.json)
finds only 1,074 distinct next-glyph targets in the 508 supported questions,
and only 266 distinct initial glyphs. The primitive knows 4,087 image forms.
Three development first-reference characters have no positive target event
anywhere in language training. Two more occur later but never as initial
targets. Only eight questions have any reference whose entire character set
appears in these targets. This coverage does not by itself explain failures:
several fully covered references fail too. Development labels are used only
for this post-assessment diagnosis, never to select training examples.

The native contextual body already gives useful source-head answers on the
same images. Its output projection and input embeddings are untied in the
authenticated configuration. The glyph decoder is anchored to input vectors;
G must learn the contextual-state-to-glyph map from sparse supervision. A
random readout plus 508 answers need not preserve the much broader native
output decision function. Untied weights alone do not prove a geometric
cause, and the stronger generalization claim remains to be tested.

## Proposed bounded experiment

Distil the native output function during training, then remove that teacher.
Keep the image encoder, glyph decoder, source body, numeric controls and
inference loop fixed. Choose all training anchors by the existing training-only
inventory, not by development answers. Decode source units only in offline
preparation to associate a valid first complete glyph with a teacher output.
Record partial-byte, missing-font and unsupported-prefix exclusions. The
runtime architecture and checkpoint must still contain no lexical matrix,
character-indexed prototype table, font, tokenizer or answer bank.

For a native hidden state h, the temporary teacher selects
\[
j(h)=\arg\max_i w_i^\top h.
\]
Use only training-image contextual states as genuine examples. Broaden
first-glyph coverage with counterfactual **hidden-state calibration**, not
fabricated claims of new language knowledge. Let k be a source unit chosen
from the full training-anchor coverage and j the current teacher winner.
With diagonal training-state variance S, one minimum-Mahalanobis correction
against that current winner is
\[
d=S(w_k-w_j),\qquad
h' = h + \frac{[m-(w_k-w_j)^\top h]_+}
 {(w_k-w_j)^\top d+\epsilon}d.
\]
Recompute the complete teacher argmax after each bounded correction. There
is no claim that finitely many such steps must reach every requested class.
Accept a calibration sample only after the actual teacher chooses its
recorded label. Record distances from genuine states, acceptance, discarded
attempts and coverage. Include genuine states and stopping events; never
use development hidden states or responses as training data.

Train the existing neural readout to the accepted continuous glyph anchor
and actual glyph pixels, using the same reconstruction/stop losses. The
teacher's source IDs are preparation labels only. Then apply the same
508-question fitting recipe and evaluate autonomous actual-image feedback.
This changes readout initialization/data, not the model's physical output
alphabet. Its extra teacher compute must be counted; it is not a matched-budget
comparison with the existing 6,000-update fit.

First qualify the compiler on separately held-out calibration states with
teacher labels and strict physical-image measurements. That measures teacher
function transfer, not language understanding. The scientific acceptance
test remains complete, readable answers to question images, correct stopping,
the existing erased controls and retention of the current date behavior.
Do not use an exact teacher head, first-character lookup or cached source
answer to repair deployed output. All language inference must still use the
same learned continuous bridge and actual emitted pixels.

This is an unimplemented candidate, not a result or a mathematical guarantee.
It should be tested before an expensive expansion of native prompt caching.
If teacher transfer works but generated histories still fail, isolate the
image encoder's semantic error and exposure shift next. If it fails even on
held-out teacher states, change the readout design before spending further
body-inference time. No new physical law is needed or claimed.
