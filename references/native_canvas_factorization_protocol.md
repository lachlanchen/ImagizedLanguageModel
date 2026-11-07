# Separate the encoded glyph field from the language context

The completed canvas control holds the dense checkpoint and seeds fixed:
Chinese copying is 13/16 on 1024-by-256 strips and 0/16 after white bottom
padding to 1024 square. English copying changes from 16/16 to 7/16. The
attention-only learning intervention preserves calibrated copying but gives
0/32 QA in both languages. Neither result identifies which part of the
padded-image path must be repaired.

This prospective diagnostic uses the exact original dense checkpoint
`016429ec5454cdd681693d881df4c6b7d52fb43dcae7b9e15857d2d9cd32b483`,
the same 32 copying requests and their original seeds. The later attention
checkpoint is not used. No parameters are learned, and no new weights are
downloaded. All input pixels, normalization, generic request, BF16 body,
FP32 sampled codec and 96-decision top-one/EOS readout stay fixed.

Let S be the 1,024 visual features from a strip, Q the 4,096 features from
its padded square, and Q_top the first 1,024 square features, covering the
same upper 256 pixels. Every request's ink is verified to lie inside that
upper strip. The four arms are:

| Arm | Visual features | Native language-context layout |
| --- | --- | --- |
| strip_short | S | Original short prefix/mask |
| square_long | Q | Original long prefix/mask |
| square_short | Q_top | Original short prefix/mask |
| strip_long | S followed by Q's remaining 3,072 features | Original long prefix/mask |

Before any hybrid call, replay all 32 original strip decisions and all 32
original square decisions exactly against the retained canvas-control
outputs. A mismatch stops the run. Assert bit-exact equality of nonvisual
headers and suffixes between the two source prefixes. Keep the original
source masks and row-major order; only the declared feature block and
context layout are exchanged. There are exactly 128 primary readouts.

The hybrid prefixes need not correspond to any real image. The square's
nominally white-region features may carry global content. Changing the
context also changes automatic language-body positions, the suffix location
and available keys. Equal seeds do not imply identical sampled noise across
different tensor shapes. This tests recovery under representation-path
interventions, not unique causal attribution to one internal operation.

Report exact stopped copying by language, every unmodified response,
input/seed/prefix hashes, the actual source replays, invocation costs and
complete model size. Retain the existing Chinese/English normalization and
EOS criteria. Do not choose examples, alter scoring or run more optimization
from these development outcomes. Recovery would justify a targeted next
interface test, not natural QA, continuous image answers, general language
competence, matched-quality efficiency or completion of the ILM goal.

The copying strings are synthetic. The source model/implementation are
NUS Show Lab's Show-o2, Apache-2.0. All source PNGs and previous decisions
remain bound to `references/evidence/native_canvas_control_result/`.
