# Native visual-flow training completes; answer quality remains unqualified

The fixed image-pair run completes all **1,536 optimizer updates** on one
RTX 4090 D. Its final two monitored training answers remain incorrect.
The separate complete initial/final assessment has started. Its first completed
[final normal-question cohort](native_visual_flow_normal_result.md) gives
0/32 seen and 0/32 held-out source-accepted OCR-exact answers. The complete
source/final controls remain in progress.

The [complete training evidence](evidence/native_visual_flow_training/README.md)
retains every original fixed monitoring image, both complete input/reference
pairs, the original training report, independent checkpoint audit and full
raw-review bindings. All 24 monitored pages were inspected without transcript,
contrast or score repair. They are seen examples from the same passage, with
two questions and fixed seeds, rather than a generalization cohort.

| Actual completed execution | Measurement |
| --- | ---: |
| Training input/target image pairs | 384 |
| Natural Chinese QA examples | 256 from 128 articles |
| Chinese/English copying examples | 128 |
| Complete permutations | 4 |
| Actual AdamW updates | 1,536 |
| Trainable FP32 adapter parameters | 37,380,992 |
| Adapter tensors / verified optimizer states | 852 / 852 |
| Complete retained runtime parameters | 6,240,279,775 |
| Whole training attempt | 11,450.839 s (3.18 h) |
| Sum of recorded update elapsed times | 9,954.717 s |
| Peak allocated / reserved GPU memory | 15.697 / 16.822 GiB |
| Actual fixed monitoring images | 24 |

The whole attempt includes source loading, saving and all 24 native 50-step
previews. Update times include input/target encoding and synchronized learning
work. Neither timing is an estimate of pure arithmetic cost. Earlier source
acquisition, preflights and the subsequent assessment are separate costs.
These measurements do not establish efficiency at matched language quality.

## What was optimized

The [registered protocol](native_visual_flow_learning_protocol.md) uses the
released DeepGen reader, connector, image transformer and VAE. Language/image
pretraining is inherited. Rank-16 adapters in the connector and generator
receive the native unweighted velocity-MSE gradient; the reader and codec stay
frozen. The target supplies training supervision, while conditioning features
come only from the input image. All 384 training examples occur exactly four
times, and no held-out example enters that optimizer schedule.

Both canvases are 1024 square. Original 512-by-256 answer images are padded
with white pixels, retaining their writing scale. The deployed source/final
comparison uses the same compiled generic condition and no changing text
embedding, answer transcript, external teacher or font lookup. Unused word
tables, output head and tokenizer are removed; the pretrained language body
remains. This is adapter learning on a pretrained system, not foundation
training of a 37M-parameter language model from pixels.

## Independent audit and interpretation

The [CPU audit](evidence/native_visual_flow_training/training_audit.json)
reads and hashes one atomic final checkpoint snapshot. Every actual optimizer
state reaches 1,536; adapter and AdamW tensors are finite, with matching shapes.
The complete schedule and training-image bindings reproduce. CPU, CUDA, time
and noise RNG states are present. All 24 original monitoring PNGs authenticate
against their actual saved decoder tensors using the source pixel conversion.
The audit runs no neural model, optimizer step or independent neural replay.

The final checkpoint SHA256 is
`87a1420b200c892e99224b37f4dffa878dd24eab2b08508c7ecdc03821ad3ae8`.
Its monitored answers do not reproduce `第四部` or
`唐太宗李世民如何成为天子`. Incorrect short strings remain even after these
examples have appeared in training. This observation does not identify a
unique causal bottleneck or quantify unseen-question accuracy.

The [complete assessment protocol](native_visual_flow_assessment_protocol.md)
fixes 384 fresh generations across final/source normal questions, erased-input
controls and copying. Its input-only generator has started from the exported
final checkpoint. Final/source counts, question dependence and copying
retention must wait for the complete authenticated cohorts and raw review.
Optimization, finite gradients and glyph-like appearance cannot qualify the
independent ILM objective.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. python scripts/audit_native_flow_training.py \
  --training-run artifacts/native_visual_flow_learning_v1 \
  --output artifacts/native_visual_flow_training_audit_v1
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. python scripts/publish_native_flow_training.py \
  --training-run artifacts/native_visual_flow_learning_v1 \
  --audit artifacts/native_visual_flow_training_audit_v1 \
  --output references/evidence/native_visual_flow_training
```

Use the existing model Python for the tensor audit, and fresh output folders.
The publisher requires complete hash-bound raw monitoring review. Full
checkpoints, deployment adapters and decoder tensors remain in the ignored
experiment directories; the public export includes original PNGs and audit
records without distributing those weights or private runtime logs.
