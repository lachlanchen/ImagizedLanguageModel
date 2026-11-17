# Layer conditioning fails the complete-record support comparison

Adding conditional scale and shift inside the fitted writer does not pass its
fixed physical support test. Both arms complete twelve real updates on the
same three complete training records. The candidate's pooled foreground F1
falls from **.348600259** to **.339850462**; unchanged conditioning reaches
**.342911305**. Three of four physical support conditions fail. No full fit is
admitted and no autonomous language score changes.

| Complete training record | Target glyphs | Common initialization | Final control | Final modulation |
|---|---:|---:|---:|---:|
| Book | 512 | 0.347489942 | .340999451 | .337504122 |
| Short QA | 5 | 0.543458371 | .770659239 | .876016260 |
| Long QA | 35 | 0.346559428 | .328940041 | .325599015 |
| Pooled | 552 | **.348600259** | **.342911305** | **.339850462** |

The complete case values are retained in the
[hash-bound receipt](evidence/layer_conditioned_writer_preflight_result_20261118.json).
The short-QA gain is retained as a component effect, not discarded because
the pooled test fails. Its five output masks are not exact source masks; its
improvement does not establish a complete correct answer or generalization.
The book and long QA deteriorate relative to control. Candidate exact source
masks number fifteen, versus twenty-one for control and seven initially.
Both final arms have zero false blanks and four false-ink outputs at the four
true source-space positions. No whitespace condition is relaxed.

Both arms start with bit-identical velocities and complete generated images.
Their common explicit layer loop has zero measured discrepancy from the
original writer on the real probe. The initial base gradients agree within
the fixed tolerance; every new modulation parameter receives finite nonzero
gradients. All twelve training-noise/time hashes match across arms. Each update
uses a whole source, accumulating its valid-glyph mean gradient before stepping.
Each arm processes 2,208 fitting positions; an additional 512-position gradient
probe per arm has no optimizer step. The final prescribed state, rather than
an intermediate, is measured.

Every trial update resets exactly. The saved source writer, reader, contextual
body, codec and length/stop head remain unchanged. Correct source-image history,
span/update counts and next-span lengths are supplied. Actual generated
feedback, independent prompt grouping and natural stopping are unmeasured.
These training-support records do not qualify held-out language.

Execution takes **129.921 monitored seconds**, with no pressure waiting,
**6.957 GiB** peak allocated GPU memory and 52.614 GiB minimum sampled available
host RAM. The outer supervisor records 140.142 seconds, including that interval.
The original complete actor has 1,465,381,142 parameters; modulation adds
6,297,600. This is an inherited foundation and a short writer-only trial,
not small-model foundation training or a useful-quality efficiency result.

The [CPU exporter](../scripts/export_layer_conditioning_preflight.py)
recounts every one of the 2,208 generated tiles across both endpoints and arms,
checks exposure and initialization equality, and exports all twelve sheets
without changing a pixel. The [complete images](evidence/layer_conditioned_writer_preflight_20261118/)
retain failures and the short-QA improvement. The raw source-image/context
cache, arrays, ledgers, registration and terminal receipt remain hash-bound.

This rejects the bounded conditioning intervention at the registered budget.
It does not establish that conditional modulation cannot work at another
scale, nor select an extra unchanged fit, loss weight or checkpoint. Together
with the negative reader-consequence comparison it leaves the integrated
conditional language-to-writing problem unresolved. A replacement algorithm
is not yet qualified or selected for another full fit. The original complete
language, independence, historical-form and resource goal remains active.
