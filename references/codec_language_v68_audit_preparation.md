# V68 autonomous audit: image preparation

Implementation detail for the frozen V68 training/evaluation protocol. This
preparation makes no actor call and changes no selected case or threshold
using model outputs. It runs on CPU while the sole GPU-0 trainer continues.

Select the prescribed 64 CMRC article pairs and eight admitted Dolly examples
per category, using the exact ordering and generator in the V68 protocol.
Authenticate the existing development report, source JSONL files and original
393-pair source. Match every first reference and complete prompt against the
actual packed development cache. Preserve all admitted original pixels.

For Chinese question blanking, retain the original complete Pango layout and
apply a white foreground attribute only to the UTF-8 byte interval belonging
to the question. This changes ink color without reflow, replacing characters,
moving context or changing glyph advances. Keep the original canvas and draw
coordinates. Verify the result contains no added foreground ink and preserves
the original layout geometry. Original source strings are used only by this
offline preparer; the actor receives a physical prompt-image tensor alone.
English blank controls are all-white images of the same physical dimensions.

Store normal/blank prompt strips and reference strips as uint8 tensors, with
separate source-bound manifest records. The evaluator must fold and clone the
selected prompt before generation and must never provide reference tensors,
lengths or IDs to the actor. References may be read only for scoring after
generation. Alternative Chinese references are rendered completely using the
same configuration or explicitly logged as excluded, without changing the
primary selected questions. English remains recorded-reference agreement,
not an exhaustive semantic evaluation.

Require 192 selected questions: 128 Chinese from 64 pairs and 64 English from
eight categories. Persist case hashes, source bindings, renderer provenance,
the exact selection, exclusions and zero model calls. The final split stays
unopened. New cases, changed layouts, output repair and partial answers are
outside this preparation.
