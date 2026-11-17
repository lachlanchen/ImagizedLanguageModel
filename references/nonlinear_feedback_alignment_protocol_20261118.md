# Nonlinear native-feedback alignment on the complete training support

The completed compact/context/KL recipes do not provide usable visual language.
The existing reader ends in LayerNorm followed by a 384-to-2,048 linear map.
Its predictions lie in a fixed affine family with at most 383 varying
directions. The completed selected-source analysis measures alignment-error
floors near .57 for the earlier fitted heads. That is a coordinate restriction,
not a language impossibility theorem or a proof of the current failure's cause.
The full-context and predictive-KL changes do not remove it.

Exact pixel aliases are a different restriction. The completed whole-schedule
[feedback audit](visual_feedback_identifiability_result.md) finds nine ambiguous
groups among 29,740 visual spans and a relative-error floor .0007492351. Reuse
that result; do not rerun its diagnosis or demand recovery of invisible source
encoding choices. Its aliases and source targets remain in this comparison.

## One architectural intervention

Let n(r) be the old LayerNorm of a frozen reader query, and Wr+b its old
normalized output. Add

\[
 f(r)=W n(r)+b+B\,\operatorname{GELU}(A n(r)+a),\qquad B=0\text{ initially}.
\]

The branch width is 4,096; it adds 9,965,568 parameters. The original native
mean/scale buffers still follow f, unchanged. The nonlinearity removes the
specific old affine-family constraint; neither the actual rank after learning
nor alignment quality follows from parameter count. Zero B preserves the initial
function. The first gradient opens B; A/a can receive gradients after B changes.
This is a standard nonlinear residual, not a new mathematical or physical law.

The control refines only the original feedback LayerNorm/linear pair. The
candidate refines the same pair plus the new branch. Both start from the exact
final learned-context reader. All earlier image features, attention, queries,
group/count heads, codec, source statistics, writer and contextual body remain
frozen. Both use the same corpus weights and optimizer schedule. No predictive
KL, new label loss, synthetic states or negative examples are added.

The complete deployed footprints would be 1,465,381,142 and 1,475,346,710
parameters, with 1,409,410,048 inherited contextual parameters. No complete
successor is deployed by this experiment. Small calibration costs do not
establish foundation pretraining or useful matched-quality efficiency.

## Complete-source preparation and fixed features

Read every authentic training shard and its original 1,024-position batch
schedule. Reconstruct each native source sequence, packed glyph span and its
feedback-loss mass. Preserve every source appearance in the sufficient weights;
keep aliases as separate targets sharing the same visible input. Expected
counts are 13,785 records, 8,172 scheduled batches, 5,749,578 span appearances,
5,844,979 native feedback positions, 29,755 native sequences and 29,740 physical
spans. These are fitting data, not held-out language evidence.

For each unique physical span, cache the frozen encoder's three pre-feedback
queries from its actual pixels and fixed codec. Native IDs and targets never
enter the encoder. The same cached query/slot is used by every alias. Use BF16
forward arithmetic and fixed batches of at most 64 spans sorted by glyph length
then image digest. Check the first eight cached spans against singleton
encoding, requiring relative RMS <=.01; record its limited scope. Verify the
complete encoder and codec unchanged. No native contextual body or teacher body
is loaded for this cache or fitting stage.

Let m(s) be the sum over source appearances of 1/(B U_b), where B=8,172 and U_b
is the actual batch's native-update count. Each slot of sequence s gets mass
m(s). Mass over all native update rows sums to one. Targets are the same BF16
compact-source embedding rows as the original trainer. For prediction v_i and
target e_i, retain the original relative error and weighted full-support loss:

\[
 R_i=\|v_i-e_i\|^2/\max(\|e_i\|^2,10^{-8}),\qquad L=\sum_i m_iR_i.
\]

No target is replaced by a mean or class identity. Cached features change the
optimization setting from complete contextual training; this experiment isolates
feedback-coordinate fitting and cannot establish contextual language learning.

## Preflight, fitting and fixed measurements

First verify exact control/candidate initial outputs on every cached query and
the same initial weighted loss. On each of the three complete prescribed
training records (513/846/1,042 glyphs), take two real head-only AdamW updates
on all its native positions. Require finite loss/gradients, a nonzero new-output
gradient, a nonzero new-expansion gradient by the second step, and exact reset
of each trial head. These checks qualify this calibration execution only.
Any failed execution or support check stops the single pipeline without retry.

Only after that preflight, fit both heads serially for exactly twenty complete
epochs over all native-sequence/slot rows. Use batch 256, fixed CPU permutations
from seed 202611185+epoch, AdamW learning rate 2e-4, betas (.9,.999), epsilon
1e-8, decay .01 and clip one. For uniformly permuted rows, a batch loss is
N/batch_size times its weighted sum, including the actual final batch size.
Every row appears once per epoch. Record each permutation digest, updates,
coverage and full weighted error after every epoch. Retain only initial/final
heads; no checkpoint selection, extra epoch or threshold adjustment.

The fixed reader-stage support condition is final candidate weighted error <=.10
and at least .10 below the final affine control, with error <=.10 on each of
the three complete training records. Preserve every error and alias. A pass
permits preparing a separately registered conditional-writing evaluation; it
does not admit full language training or qualify an ILM. A failure rejects this
calibration as sufficient alignment support, without disproving all nonlinear
readers or authorizing unchanged extra fitting. Copying, readable images,
complete language, stopping and efficiency remain separate claims. All original
language thresholds and reserved DRCD protections stay unchanged.

## Source, ownership and resource contract

CPU preparation uses two threads, no CUDA, a 900-second deadline and at most
2 GiB process RSS. Bind its complete source/shard hashes, input snapshot,
implementation, numerical CPU checks and derived data before neural execution;
commit and push the registration first. Reuse shared weights, without copying
the source archive. All derived vectors and target IDs remain private training
artifacts and never enter the deployed visual model.

The single cache/preflight/fit pipeline owns GPU0 through the existing mutex.
Entry requires 48 GiB available host RAM, 20 GiB free GPU0, 48 GiB disk and PSI
some <=.5/full <=.1. Retain 24 GiB host RAM, cap this head-only pipeline at
8 GiB allocated VRAM, 1,800 monitored seconds and 300 seconds of pressure
waiting. Use two CPU threads, with an external 1,860-second deadline and
thirty-second termination grace. Inspect processes, tmux and GUI ports before
launch; preserve foreign services. Preserve partial failures and fixed heads,
and release the exact owner after evidence capture. This registration does
not authorize a second pipeline or another full language-model fit.
