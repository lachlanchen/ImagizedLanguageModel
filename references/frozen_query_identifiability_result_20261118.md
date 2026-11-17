# Identical query states explain an alignment floor, not the whole failure

The [complete saved-query analysis](evidence/frozen_query_identifiability_20261118.json)
groups all 33,941 fitting rows into 31,125 distinct BF16 query inputs. It finds
2,682 duplicate groups containing 5,498 rows and 3.7008% of the original native
feedback mass. Every duplicate group contains different target vectors.

Their exact-arithmetic, free-function relative-error floor is **.016774741**.
The direct residual and moment formula agree within 3.26e-19. This bound is
below the fixed .10 reader gate, so identical-query ambiguity alone does not
make that gate impossible. It also does not show that the present finite
network, precision or optimizer can reach it. The current nonlinear ridge
error is .271039 on the same fitting support.

Of the duplicate groups, 2,675 combine different slots of the same physical
span, seven contain same-image/same-slot encoding alternatives, and none
combine distinct physical span images. These categories describe the observed
groups; their definitions are retained in the evidence. Original source labels
and all conflicting rows remain unchanged.

The implementation explains a concrete slot failure. Its cross-attention feeds
the feedback network directly, without a residual carrying the query. For a
one-glyph span, the one-key attention softmax equals one, independent of query
slot. Different native-unit slots can therefore receive exactly the same
feedback input. A shared nonlinear output map cannot distinguish those inputs.
This does not mean that all multi-slot spans collide or that all language
information is lost. Positional/contextual dynamics may behave differently;
the bound concerns the native-vector auxiliary task on this frozen cache.

The [prospective CPU protocol](frozen_query_identifiability_protocol_20261118.md)
uses the original masses and same BF16 source embedding targets, checks every
duplicate group's optimum independently and preserves all group contributions
privately by hash. It performs no neural call, rendering, fitting, teacher
request or answer reading. Runtime is **13.127 seconds**, peak process RSS
**1,466,040,320 bytes**, within the sixty-second/two-GiB limits.

The earlier pixel-span audit assumed an independent prediction per image and
slot; its .000749235 floor is a different, less restricted problem. This new
cache-input bound does not replace that result or establish a language-quality
bound. It supports retaining slot identity as a future reader requirement,
while not promoting that issue to a complete explanation for the much larger
current error. Any new feature-learning or slot intervention requires separate
registration, complete sources, resource support and fixed evaluation.
