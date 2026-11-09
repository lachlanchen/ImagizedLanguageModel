# Reader-query learning runs, but does not yet answer correctly

The two-update continuous-query preflight completes on GPU0 at 15.281 GiB
peak allocated memory. Both autonomous answer images remain incorrect
(0/2 exact answers; clean-reference OCR 2/2). This establishes a working
gradient and image-generation path, not language learning or generalization.
The [complete preceding assessment](native_visual_flow_complete_result.md)
also remains unsuccessful: zero exact seen and held-out QA answers after
1,536 adapter updates, with copying rising from 0/64 to 5/64.

The new [protocol](native_query_learning_preflight_protocol.md) changes one
learning interface. The completed adapter run detached its frozen reader
features and kept the 128 continuous reader queries frozen. Here gradients
traverse the frozen writer, connector and reader to shared 128-by-2048 query
vectors. Only 262,144 FP32 query coordinates enter AdamW. All other weights,
including the final trained adapters, remain frozen. This is a hypothesis
about a trainable interface; the old failure does not establish that frozen
queries caused it.

| Measured observation | Result |
| --- | ---: |
| Actual optimizer updates | 2 |
| Nonzero gradient coordinates, update 1 | 262,116/262,144 |
| Nonzero gradient coordinates, update 2 | 262,107/262,144 |
| Changed FP32 coordinates after both updates | 262,144/262,144 |
| Coordinates changed after conversion to native BF16 | 153,094/262,144 |
| Generated exact answers after both updates | 0/2 |
| Clean-reference exact OCR | 2/2 |
| Complete execution time, including loading and generation | 99.423 s |
| Peak allocated GPU memory | 15.281 GiB |
| Retained pretrained runtime parameters | 6,240,279,775 |

The unchanged helper and differentiable/checkpointed reader reproduce the
native features exactly before optimization. The first full-noise velocity
prediction also reproduces its saved native baseline bit-exactly. The
optimizer records two updates, and all saved training tensors are finite. The actual
writer inputs equal independent noise, with the answer image supplying only
the supervised velocity target. Saved-tensor MSE recomputes exactly as
0.003543323604390025 and 0.022178594022989273. These are different examples,
so their two losses are not a learning curve.

Both seen examples belong to one previously used Chinese passage pair.
After the second update, the model generates complete 1024-square pages
with the original guidance-one sampler, fifty steps, and the same input
images and generation seeds as the matched preceding outputs. The output
pixels change, but short malformed fragments remain in place of the correct
answers. All four original before/after pages and both reference pages were
inspected without repairs. Review had source and OCR access and assigns no
independent manual accuracy. No held-out cases are tested in this preflight.

The [CPU audit](evidence/native_query_learning_preflight/audit.json) verifies
the checkpoint, optimizer steps, numerical losses, changed query coordinates,
original input/seed bindings and decoder-tensor-to-PNG authentication. The
[execution report](evidence/native_query_learning_preflight/execution.json),
[fixed OCR report](evidence/native_query_learning_preflight/ocr.json),
[raw review](evidence/native_query_learning_preflight/raw_review.json) and
[archive hashes](evidence/native_query_learning_preflight/provenance.json)
preserve the measured outcome. The original full pages are under
[images/](evidence/native_query_learning_preflight/images/). Dataset
attribution and the original CMRC 2018 licence accompany the archive.

The deployed reader still retains language/image pretraining. Optimizing a
small query state does not create a 262K-parameter foundation model or show
language acquisition from pixels alone. A subsequent bounded training-fit
experiment must demonstrate correct autonomous answers, question dependence,
and held-out behavior before the candidate can be described as a useful ILM.
Two updates neither establish such learning nor rule out this interface.

To reproduce the offline audit of the preserved local run:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. python scripts/audit_native_query_learning.py \
  --output artifacts/native_query_learning_audit_new
```

The audit performs no optimizer update, neural model loading or new image
generation. The original 19 experiment sources and all three query-preflight
source/protocol files retain their execution-bound hashes.
