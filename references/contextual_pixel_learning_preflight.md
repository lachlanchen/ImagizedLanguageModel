# Contextual image-span learning executes within one 4090

The [prospective execution checks](contextual_pixel_learning_protocol.md)
pass. They establish that the existing image-span objective can train the
native contextual adapters while preserving the completed source interface.
They do not establish an improvement in reading, generation or efficiency.
The [numeric receipt](evidence/contextual_pixel_learning_preflight.json)
retains every checked gradient norm and the measured resource cases.

Preparation authenticates all 3,211 CMRC short training records, admitting
3,193 QA images and reusing 64 training copies. Eighteen rows are excluded
before model calls for font/layout or representability limits. Training
article titles and paragraph hashes do not overlap the unchanged development
set. The first preparation invocation omitted the documented renderer preload
and failed before rendering or model calls; the corrected invocation retains
that failed attempt separately.

The fixed-grid sampling forward is bit-exact with the existing operation.
Its deterministic adjoint differs from CPU autograd by at most
1.1102230246251565e-16 and passes finite differences, including border samples.
Four actual native forwards reproduce cached visual descriptors, task memory,
physical units and geometry exactly. Enabling checkpointed training at zero
adapters also preserves those descriptors.

All 231 trained tensors receive finite gradients. Each of 108 contextual up
matrices receives a nonzero gradient initially; each of 108 down matrices
does so after an update. Frozen weights receive no gradient and remain
outside Adam. A saved real update replays model, optimizer, CPU/CUDA RNG and
metrics bit-for-bit. Preflight updates are not production training.

| Measured quantity | Value |
| --- | ---: |
| Complete runtime parameters, including external pretraining | 7,531,277,171 |
| Trained parameters in the prospective adapted arm | 8,571,522 |
| Existing span-head parameters | 2,378,370 |
| Contextual adapter parameters | 6,193,152 |
| Source setup | 58.997 s |
| Six full forward/backward invocations and resume checks | 15.283 s |
| Peak allocated VRAM | 18.043 GiB |
| Peak reserved VRAM | 21.346 GiB |

The two new training-question updates take 2.415 and 2.394 seconds. With
allocated Adam moments, the largest-sequence case has 1,208 positions and
takes 2.690 seconds; the most-ink case has 596 units and takes 2.385 seconds.
These are single-example execution costs, not model-quality or matched-task
efficiency results. The full body remains externally pretrained.

The process exits after qualification and releases GPU0. GPU1 is untouched.
Production training requires a separate fixed-budget comparison against the
continued frozen-head control, with the same data/order and final assessment.
No new manuscript claim of useful language learning follows from this check.
The complete independent image-language objective remains unachieved.
