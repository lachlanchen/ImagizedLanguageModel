# Source-only one-character line-clock diagnostic

The balanced writer has passed physical qualification but not the new-context
answer pilot. Preserve its18/32 versus19/32 parent result; do not change that
trial. This separate diagnostic tests whether explicit character-level source
framing reduces the previously observed native-unit/character-step mismatch.
It is not an ILM result, uses a source lexical head, generates no learned
answer image, performs no optimizer update and supplies no training examples.

Use the exact same eight training and sixteen earlier normal development
input images as the completed feedback diagnostic. Do not use either reserved
32-question cohort. Load the original authenticated source body with zero
contextual deltas and the unchanged image encoderE. Compile a new fixed task:
answer the question, but put exactly one character on each line and a newline
after every character, with no other text. Keep those controls separate from
all existing numeric task vectors and checkpoints.

Compare three routes over all24 images: native source units; first complete
character source embedding plus one fixed newline vector; and that character's
canonical image throughE plus the same newline vector. The latter routes
advance two source positions per emitted glyph and decide the next glyph after
the newline. Native-unit feedback advances according to its own actual units.
All three initial image-prefill states must be bit-identical within each case.
Partial Unicode, unsupported first forms and caps remain explicit failures.
Record source-unit formatting violations without dropping their cases.

Offline compilation must check the newline's single-unit codec and report
all primitive characters for which character+newline tokenization differs
from two separate source units. These exceptions do not filter evaluation.
This is source-format alignment, not a claim of a tokenizer-free compiler.
A future independent image runtime could use the constant numeric newline
advance without a lexical head, but that runtime is not established here.

The cap is64 source units in the native route and64 complete forms in each
character route. Report that unequal cap interpretation, all unsupported cases,
full strings and stop events. For literal scoring remove only structural LF
characters, not spaces, explanations, punctuation or answer substrings. Also
report whether every native output line contains exactly one codepoint with
a final newline; codepoints are not a general theory of visual graphemes.

A mechanical-alignment signal requires native development accuracy at least
10/16 and exact-character-plus-LF accuracy at least max(10,native_correct-1).
Image-plus-LF accuracy is descriptive, not a substitute for independent learned
image output. If the mechanical condition fails, do not present this framing
as a working remedy or silently try new phrasings on the same cohort.

Commit code and controls before source-head execution. Use the single GPU0
mutex, existing48 GiB entry/24 GiB live/22 GiB GPU limits and bounded low-PSI
waiting. Keep existing qualified runtime code and controls unchanged. All
source-body knowledge and pretraining costs remain inherited.
