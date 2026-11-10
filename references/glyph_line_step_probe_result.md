# Explicit line framing does not repair the glyph clock

The fixed source-only diagnostic completes all 72 trajectories on the same
eight training and sixteen earlier normal development images. It uses an
original source lexical head, no optimizer update and no learned answer image.
The [complete records](evidence/glyph_line_step_probe_v1/scores.json) retain
every string, unsupported event and cap. These are helper-policy results,
not ILM language accuracy.

| Development route | Complete literal matches | Native end | Unsupported | Cap |
|---|---:|---:|---:|---:|
| Native source units, character-per-line request | 7/16 | 15/16 | 0/16 | 1/16 |
| Exact first character plus fixed LF | 1/16 | 2/16 | 9/16 | 5/16 |
| Character image through E plus fixed LF | 1/16 | 4/16 | 8/16 | 4/16 |

Training controls give 7/8, 4/8 and 4/8 literal matches. All three within-case
image-prefill states are bit-identical. Compilation checks 4,096 primitive
character/source-ID pairs and records forty exceptions where character+LF
encoding differs from two separate source units. No exception filters the
evaluation cohort.

None of the native outputs meets the strict requested one-codepoint-per-line
format **with a final LF**. Appending LF after every forced character therefore
does not reproduce the native policy's own segmentation or termination. Some
routes repeat question fragments or continue after an otherwise plausible
answer. Only structural LF characters are removed for literal matching;
spaces, punctuation, explanations and partial outputs remain unaltered.

The preset mechanical signal requires at least 10/16 native matches and
at least max(10,native-1) exact-character-plus-LF matches. Both fail. This
framing is rejected as a remedy; the runtime and its original controls remain
unchanged. The experiment does not prove that all continuous sequence
interfaces are impossible. It demonstrates that a formatting request plus
manual position advancement is insufficient here.

The run takes 429.67 s under the common single-GPU mutex. Full resource
measurements, source identities, compiler exceptions and every trajectory
hash are archived. Its knowledge, lexical head and body are inherited; no
small independently trained language model or new training corpus results.
