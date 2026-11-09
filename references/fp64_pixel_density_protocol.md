# Full physical-head precision control after normalization-only rejection

The normalization-only candidate restores the original 1e-4 cache gate but
fails its separate requirement that both paths improve their worst error
against the independent reference. Its full-path maximum error changes from
7.28034e-5 to 7.45783e-5. Preserve that failed control without relabelling it.
FP32 dense maps and attention still accumulate rounding error; improved
normalization alone cannot guarantee a smaller maximum over every coordinate.

The targeted successor evaluates all arithmetic of the 2,666,944-parameter
physical head in FP64, retaining its FP32 parameters and unchanged state-dict
layout. Activations, dense-map casts, LayerNorm, attention, physical positions
and logits use FP64. Gradients flow through casts into FP32 stored parameters.
The inherited multi-billion-parameter reader is not loaded or modified in
this component control. It retains its original precision in any later actor.

Use the same captured two states, complete teacher prefixes, independent FP64
reference and original GPU traces. The same registered checks apply: exact
original replay; unchanged parameters; cache error at most 1e-4; exact greedy
agreement between both candidate paths and with original outputs; exact
target causality; lower maximum error to the reference for both paths; and
finite, present gradients after a complete-image NLL backward. Save every
metric before enforcing the checks. Retain failures without an automatic
rerun or a search over precision modes.

Measure original/candidate two-block forward latency with five warmups and
twenty timed calls and record actual peak allocation. One project-locked
GPU0 job has the same 120-second and 2 GiB limits. This is a numerical repair
and component cost measurement, not language training or a model efficiency
claim. Only after qualification may a new native learning preflight adopt
this exact head implementation while preserving the original failure records.
