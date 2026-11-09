# Learning the continuous reader queries: a bounded native preflight

The completed native-flow run adapts connector/writer projections while its
reader features and 128 continuous query vectors stay frozen. It produces no
correct QA answers. The codec preserves four tested answer images, and merely
changing guidance does not repair four tested questions. These observations
motivate testing a gradient path through the reader; they do not establish
that frozen queries uniquely caused the failure.

For an input page x and shared continuous queries Q, use the existing frozen
visual/language reader R, final trained connector C and image writer v:

\[
s_Q(x)=C(R(x,Q)),\qquad
L(Q)=M\{v(\epsilon,t_1,s_Q(x),r(x)),\epsilon-E(y)\}.
\]

At native sigma one the writer frame is exactly independent noise. The
correct answer supplies supervision only. The derivative traverses the
frozen writer, connector and reader to Q. Frozen weights still transmit
gradients to inputs. This changes the previous detached reader interface.
The task vectors are shared across questions, with no per-question vectors,
changing word embeddings, vocabulary head, teacher answer or OCR objective.

The principle of optimizing continuous vectors through a frozen pretrained
body is related to [continuous prompt learning](https://aclanthology.org/2021.acl-long.353/).
Here the native queries follow the image/task context and use the release's
existing hierarchical conditioning path. This is not a new learning theorem,
foundation training from scratch or a demonstrated language improvement.

Start from the exact completed final native-flow adapter export. Freeze all
reader, connector, adapter, writer and codec weights. Clone the existing
128 by 2048 queries into 262,144 FP32 optimization masters; cast them to the
native BF16 at the same reader interface. The whole retained model remains
approximately 6.24B parameters. Small tunable state is not a small foundation.

Use only the first seen passage pair already fixed by the completed field
probe: TRAIN_3802_QUERY_0 then TRAIN_3802_QUERY_1. Use their registered input
pixels, feature/codec/common-noise seeds and sigma-one native timestep. No
held-out input enters either update. Before optimization, require exact
equality of the ordinary detached and differentiable feature paths, and the
actual native field against the saved final-model field for the first case.
Preserve both results; do not change tolerances after observing a mismatch.

Run two actual AdamW updates, rate 5e-5, betas (.9,.95), zero weight decay and
gradient norm clipped at one. Only the query master enters the optimizer.
Require a finite nonzero query gradient and changed queries, unchanged frozen
weight versions, and peak allocated memory below 23 GiB. Use native
non-reentrant checkpointing through the reader and writer, with caching
disabled during learning. No full-model or linear-projection weight changes
are allowed. Persist the query masters, optimizer states, gradients, actual
noise/target/prediction tensors and complete bindings.

After the second update, copy the BF16 queries to their original native
parameter slot. Use the unchanged positive-branch guidance-one native sampler
to generate both complete 1024-square answer pages, at the same two input
images and seeds as the completed guidance diagnostic. Preserve actual
decoder tensors and every PNG. The existing pre-update outputs are matched
controls, not new samples. Score and inspect both new original pages offline.
These are seen training examples. Two updates and any improved appearances
cannot prove generalization, a reader repair, semantics or a usable ILM.

This preflight tests real differentiability, memory fit and image deployment
of query learning. It does not authorize an unchanged book-scale run. A
subsequent training-fit experiment needs its own fixed budget and autonomous
seen/held-out paired QA/copying assessment. Stop on a changed binding,
nonfinite value, numerical replay failure or memory breach. Preserve failed
attempts without automatic retry, canvas change or quality reroll.

Use one GPU0 child through the existing shared lock; preserve GPU1. Prepare
on CPU and commit the exact helper, runner and this protocol before execution.
The completed 19-source experiment remains byte-identical. No model or weight
download, active manuscript promotion or Office job is part of this preflight.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. python scripts/preflight_native_query_learning.py \
  --stage prepare --output artifacts/native_query_learning_preflight_v1
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/preflight_native_query_learning.py \
  --stage run --output artifacts/native_query_learning_preflight_v1
```
