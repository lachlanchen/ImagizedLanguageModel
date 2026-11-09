# Shared reader-query fit across native noise levels

The preceding two-update preflight verifies real gradients, changed BF16
query coordinates and a 15.281 GiB memory fit. Its two image answers are
incorrect. The completed 1,536-update adapter experiment also fails QA.
This experiment tests whether sustained learning through the reader interface
can fit a small paired-question task. It does not substitute that task for
the full independent glyph-language-model objective.

For question/passage image x, answer image y and shared reader queries Q:

\[
s_Q(x)=C(R(x,Q)),\quad z_y=E(y),\quad
z_\sigma=(1-\sigma)z_y+\sigma\epsilon,\quad
L(Q)=\|v(z_\sigma,t_\sigma,s_Q(x),r(x))-(\epsilon-z_y)\|_2^2/d.
\]

Only the native 128-by-2048 queries learn. Reader, vision encoder, trained
connector/writer adapters and codec remain frozen. They transmit gradients
to the FP32 query masters. The native writer and frozen VAE generate the
answer image. There is no lexical head, answer-class selector, changing text
condition, OCR training reward or font-based output renderer. The retained
6.24B pretrained body remains explicit; this is not foundation training from
pixels. Continuous-vector optimization through a frozen body is related to
[Prefix-Tuning](https://aclanthology.org/2021.acl-long.353/); the release's
native query positions and hierarchical conditions differ from that method.
No new theorem or efficiency result is asserted.

Use the exact preceding final adapter export and original native queries,
not the two-update query checkpoint. Train only TRAIN_3802_QUERY_0 and
TRAIN_3802_QUERY_1, the previously fixed first seen passage pair. Their answer
sets differ. All images and source/protocol bindings remain hash-checked.
The separate frozen target encoder caches the two answer latents; no target
is used as an inference condition. Questions enter through image pixels.

Perform exactly 128 AdamW updates, alternating the two questions for 64
rounds, rate 5e-5, betas (.9,.95), no weight decay, norm clipping at one.
Use a fresh native 1,000-point shifted training scheduler. The first round
and every eighth round use sigma exactly one for both questions (16 updates).
The other 56 rounds use deterministic uniform native indices in [1,999]
(112 updates). This deliberate full-noise anchoring is not an unbiased draw
from the release's original uniform schedule. Each round uses a shared fresh
Gaussian seed for its two questions; the first round reuses the registered
field-probe noise for numerical replay. No final sampling seed enters updates.
The full schedule is written and committed before model execution.

Before the first update require bit-exact ordinary and differentiable reader
features, including checkpointed forward mode, and bit-exact first full-noise
prediction versus the completed final field probe. Stop on any mismatch,
nonfinite tensor, changed binding, frozen-weight version change, allocated
memory at or above 23 GiB, or 45 minutes of execution. No tolerance change,
automatic retry, output selection or reroll is allowed. Record every update;
save full actual loss/gradient tensors at updates 1,2,32,64,96,128, and query
plus optimizer checkpoints at 32,64,96,128. Two final registered full-noise
fields provide a matched numerical diagnostic, not an answer score.

After learning, install the BF16 queries in their native parameter slot.
Use the same guidance-one, fifty-step, 1024-square native image sampler.
The registered existing four pre-update normal QA pages are reused unchanged
as matched controls (the first seen and first held-out passage pairs). New
generations are fixed before execution:

| Cohort | New images |
| --- | ---: |
| Initial copying: first seen/held Chinese and English strings | 4 |
| Final normal: original first seen/held passage pairs | 4 |
| Final copying: matched initial copying strings/seeds | 4 |
| Final seen questions at a second unseen generation seed | 2 |
| Final question-erased seen pair, identical seeds | 2 |
| Final passage-erased seen pair, matched seeds | 2 |

All 18 original generated pages and decoder tensors must be preserved,
authenticated, scored with the existing offline OCR engine and inspected.
Reference calibration remains separate. All reported answer counts use the
unchanged canonical OCR-exact rule. No OCR result enters an update.
Seen fit requires both normal answers correct, and seed robustness requires
both answers correct at the second seed. Held-out paired answering requires
both held-out answers correct. Question and passage dependence each require
the seen normal exact count to exceed its corresponding erased count.
Copying retention requires the final exact count not to fall below the
initial count. These are small developmental gates, not broad language or
historical-script qualification. All these examples have been reused in
development; held-out means excluded from this optimizer, not an untouched
benchmark. Loss reduction, field alignment and visible pixel changes alone
cannot satisfy the answering gates.

Use the existing verified LocalLLM image-generation Python environment and
GPU0 lock. Preserve GPU1 and every other project's processes. A lightweight
wait stage may observe resources for at most two hours, polling every thirty
seconds; it must not load torch/model weights or hold the GPU lock while
waiting. It may start one child run once the existing cold-load GPU/RAM/swap
thresholds and lock are available. Expiry records a terminal wait timeout,
not a failed training result. A failed model attempt is never automatically
restarted. After a successful model child exits and frees GPU memory, the
coordinator may start the separate CPU scoring child. Raw-image review still
requires actual inspection. Record actual PID and phase in the private
runtime handoff.

Prepare on CPU, commit and push the runner and this protocol, then run:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. python scripts/train_native_query_fit.py \
  --stage prepare --output artifacts/native_query_fit_v1
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/train_native_query_fit.py \
  --stage wait --output artifacts/native_query_fit_v1
```

The 22 executed predecessor source/protocol files remain byte-identical.
No weight download, book-scale run or active publication promotion is part
of this experiment. Its measured outcome will determine the next model
change and the subsequent manuscript revision.
