# Exact native-target conflicts in the saved reader queries

The full-support nonlinear head and direct ridge refit remain above the native
alignment gate. Before choosing another feature-training intervention, compute
the exact loss floor imposed by identical *saved query inputs* to any shared
feedback function. This is a new input equivalence relation, not a repetition of
the completed source-pixel alias or old native-history diagnoses.

The reader uses cross-attention without adding the learned query afterward.
With a one-glyph span, each query's attention softmax has one key and weight
one, so the output is independent of query slot. A multi-unit source can
therefore assign different embedding targets to identical feedback inputs.
This code observation alone does not establish how much loss it explains.

Read all 33,941 prepared source/slot rows, original corpus masses, BF16 native
targets and all 29,740 cached physical-span query triples. Group rows by exact
BF16 query bytes. Keep every source alias and slot. For group g, coefficient
c_i=m_i/max(||e_i||²,1e-8), report the free-function optimum

\[
 \bar e_g=\sum_i c_ie_i/\sum_i c_i,\qquad
 L_{min}=\sum_g\sum_{i\in g}c_i\|e_i-\bar e_g\|^2.
\]

Independently check each duplicate group's direct residual against its moment
formula. Singleton groups have zero free-function loss; no fitted neural model
is involved. Count duplicate rows/mass, conflicting native targets, same-image
cross-slot groups, cross-image groups and encoding-only groups. These classes
may overlap and must not be added as disjoint partitions. Report the ten largest
loss contributors by query digest and source row/slot/image/native identifiers;
do not decode or repair any answer. Save all group contributions privately with
a hash; public counts retain the complete denominator.

Verify registered prepared/cache/source identities and preserve originals.
No neural forward, model update, new rendering, teacher request, development
record, DRCD record or transcription enters this analysis. Use two CPU threads,
no CUDA, at most 2 GiB process RSS and a sixty-second deadline.

If the exact-query floor is at least .10, shared-feedback-head fitting cannot
meet the fixed alignment gate on these inputs, even with perfect optimization;
the next change must distinguish relevant query inputs. If below .10, exact
identity conflicts alone do not rule out that gate. This does not prove that a
finite nonlinear map can reach it; separately registered feature learning or
query changes still need real support. Neither branch is language evidence.
