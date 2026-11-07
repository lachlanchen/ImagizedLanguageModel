# Can retained writing and source-language histories be aligned?

Prospective CPU data audit, 7 November 2026. Earlier Show-o2 state alignment
supervised pre-answer prompt states. It did not supervise the Qwen writer's
image-history transitions. A new history-alignment objective remains an
untested hypothesis. This audit determines what the existing traces can
actually support; it does not optimize, invoke or qualify a language model.

Use all 32 training cases shared by the completed V77 source helper and V80
correct-history writer diagnosis. Authenticate both source reports and their
verification receipts, individual traces and reference images. Require
identical owning physical-prompt hashes and BF16 reader-memory hashes. Check
the native writer's initial state against the helper's initial state after
the same BF16 rounding. Retain failures rather than selecting favorable cases.

The writer advances by two learned image embeddings per 64 physical columns.
The source helper advances by native language pieces. Neither equal ordinal
step numbers nor a nearest string prefix establishes identical history.
For every original image boundary j, including initial and terminal states,
test the actual first 64j answer columns against every complete Unicode text
prefix of the canonical reference rendered in the original answer's font,
canvas and draw origin. Require no unobserved ink after the boundary in the
candidate prefix. Preserve the full reference's draw origin: independently
recentering every prefix would introduce a separate layout change.

Decode the retained source-piece prefixes with the existing ByteLevel decoder,
using only tokenizer strings from the shared GGUF metadata. Materialize no
source weight, vocabulary projection or contextual body. Bind the tokenizer
string table by hash; disclose that the complete archive is not rehashed in
this metadata-only audit. A language prefix is eligible only when its bytes
decode to complete UTF-8 and its corresponding saved pre-decision state
exists. Check the decoded complete source output against its retained text.

Match a saved source state to an image-history state only when both have the
same canonical prefix text and that prefix passes the exact raster test.
Keep all matching source positions if the correspondence is ambiguous.
Report coverage across all boundaries, separated into initial, intermediate
and terminal positions. Distinguish boundaries with no exact canonical-prefix
raster from boundaries with such a raster but no matching source trace state.
The latter can arise from different source outputs or unavailable states;
they must not be labelled a model failure or silently removed.

Save canonical-prefix PNGs, original reference PNGs, source pieces, complete
state arrays and all matching indices. Report normalized state distances only
on physically matched prefixes. Different vector distances do not establish
a semantic or causal mechanism, especially with different step counts and
rotary positions. An independent NumPy/PIL verifier must reconstruct UTF-8
prefixes from the saved pieces, rescan the physical matches and recompute
the retained state distances and all coverage totals. It does not rerun
either contextual model or independently implement font shaping.

Run with CUDA hidden, at most four CPU threads and the existing Python 3.12
renderer environment. There are zero model forwards and zero optimizer
updates. No source outcomes select a new training set, checkpoint or gate.
The active contextual adaptation comparison and its 23 bound files remain
unchanged. This audit cannot establish a repair of free image-language
generation or completion of the independent ILM goal.
