# Validate higher-precision normalization on captured native writing states

The captured FP32 decoder exceeds the 1e-4 cache/full-prefix logit criterion
despite identical autonomous pixel choices. Independent FP64 equations account
for the discrepancy. This control changes only the nine LayerNorm operations
in the physical decoder to FP64 arithmetic, casting their results back to
FP32. All parameter values, shapes, names and storage types remain unchanged.
Linear maps, attention, activations and output logits retain FP32 computation.
There is no logit rounding, probability adjustment or tolerance increase.

Use the exact captured two states, complete 256-piece targets and decoder
weights. First replay the original GPU teacher traces and both original
autonomous greedy paths bit exactly. Then test the candidate against the
unchanged 1e-4 maximum absolute cache/full-prefix criterion, exact autonomous
greedy agreement between its paths, equality to the original emitted blocks,
exact target-suffix causality and lower maximum error to the independent FP64
reference for each teacher path. Check that every parameter has a present
finite gradient after actual full-image NLL backward. No optimizer update is
performed, and these local block controls receive no language-quality credit.

Retain both full traces and every metric even if a criterion fails. Do not
search normalization modes or automatically retry. Measure two-block forward
latency with five warmups and twenty timed calls per implementation; this
component timing excludes the inherited reader and is not an ILM efficiency
comparison. One GPU0 job has a 120-second limit and 2 GiB peak allocated
memory cap, under the existing workstation admission and project lock.
Only a qualifying numerical control can motivate integration into a new
native learning preflight. It cannot qualify language generalization.
