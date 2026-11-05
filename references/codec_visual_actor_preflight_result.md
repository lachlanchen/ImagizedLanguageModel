# Complete codec-language actor: GPU execution passes

The [registered preflight](codec_visual_language_actor_preflight_plan.md)
executes the actual 27-layer reader, 36-layer contextual body, trainable
continuous output projection and frozen V34 codec on physical GPU 0. All 24
execution checks pass. The model has 7,539,239,282 parameters, of which
9,351,937 train. Its contextual vocabulary module is absent.

Measured peak allocated memory is 16.953 GiB; peak reserved is 17.381 GiB.
Actual Adam-state backward covers both the 918-position sequence with 109
answer blocks and the 512-by-1024 prompt-page witness. Finite, nonzero gradients
reach the output projection, structural queries, stop head, first query adapter
and last value adapter. Frozen reader, base and codec parameters receive none.

Changing future pixels leaves earlier query states, predicted writing codes
and stop logits exactly unchanged. Independent visual encoding is also exactly
isolated. Cached/full context relative maximum error is 1.292e-5; the output
code error is 1.903e-5, both below the existing 0.02 tolerance. All trainable
parameters, optimizer state, sampling and global RNG states replay exactly
across a restored real optimizer update. Two generated blocks are finite and
deterministic with actual image feedback.

Setup takes 68.949 s; the whole preflight takes 78.022 s. The completed process
and owned tmux session exit normally, and GPU 0 returns to its idle allocation.
Raw data are under `artifacts/codec_visual_actor_preflight_20261105/attempt01/`.
Report SHA-256: `960a79cd02eab2b9f084a926dc137bef275356cec3ae9df26a567acc33d7f0ba`.
The [source-bound receipt](evidence/codec_visual_actor_preflight_result.json)
checks the committed source identity and report consistency; it is not a
second independent GPU execution.

Only two distinct optimizer updates precede the generated examples. Their
raw strokes are fragmented, and no semantic quality is evaluated. This result
qualifies execution of the new actor, not instruction following or useful
language. Its next step is full-corpus causal instruction training, followed
by autonomous answer-image and question-control evaluation. The independent
image-language goal remains open.
