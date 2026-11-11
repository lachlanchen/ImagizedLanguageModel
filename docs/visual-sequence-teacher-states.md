# Teacher context states for full visual sequences

The completed run covers all 13,785 training sequences. A local Qwen3-VL
teacher performed 3,337 production causal forward batches over their aligned source
units. Batches keep each passage independent, with right padding masked out.
The schedule covers 5,844,979 real input positions and 103,253 padded positions.
The resulting 108 shards contain 2,380,620 BF16 context states. Preparation
verified every position on CPU. See the
[preparation evidence](../references/evidence/visual_sequence_teacher_preparation_20261111.json)
and [completed cache evidence](../references/evidence/compact_visual_span_gpu_start_20261111.json).

The cache stage took 1,783.41 seconds, including 770.72 seconds waiting for host
memory pressure, with 13.396 GiB peak allocated GPU memory. Two additional
diagnostic forward calls checked causal isolation and padding. Perturbing the
reference suffix left the earlier states bit-identical; right padding gave
0.01250 relative RMS difference, within the fixed 0.02 tolerance. The teacher
receives source text offline. These measurements concern training preparation;
visual language ability requires the student's generated answers.

| State supervision | Rows |
| --- | ---: |
| Book continuation spans | 2,243,532 |
| Complete QA answer spans | 123,303 |
| End of sequence states | 13,785 |
| Total | 2,380,620 |

For a book passage, the first visible span supplies context and the states
after subsequent span boundaries supervise the continuation. For QA, the
first saved state comes from the end of the prompt. It supervises the first
answer span. Each following state conditions on the full reference history
through the previous span. The last state supports the learned stopping
decision. The original teacher update ranges are retained when a visual
span covers several source units.

This is offline text prefill. The saved targets describe the source text
trajectory, and later training must test transfer from actual image input.
The authenticated model loader retains the 36 contextual layers and reads
embeddings for the observed training units. It uses the existing shared
archive and leaves the vocabulary output projection out of the cache stage.
Every saved state has width 4,096. The BF16 state arrays occupy about
19.5 GB before file metadata. A subsequent fit can read them by shard.

The GPU stage verifies causal isolation by perturbing the actual reference
suffix of a training QA sequence and measuring its prompt states. It also
checks the effect of right padding on real positions. It publishes each
completed shard atomically and authenticates those shards on resume. A
single model owner uses the existing project GPU mutex.

## Compact contextual body

The local Qwen3 1.7B archive provides a smaller pretrained body for the
span model. Its retained contextual weights contain 1,409,410,048 parameters,
28 layers and width 2,048. Those weights alone take 2.82 GB in BF16. That
figure excludes training state, activations and the image encoder and writer.
The inspected archives have identical merge tables and equal spellings for
all 26,710 training units. Their complete spelling tables differ. The
observed match supports reusing this corpus alignment for source-derived
training targets. The compact fit has started on one 4090. Its completed
language evaluation remains pending.

Use the teacher states to supervise the smaller visual body over complete
spans. The image encoder must learn from visible glyphs and the writer must
generate the following glyph images. Keep the teacher and its lexical
matrices in the offline training process. Measure the standalone reader,
writer and image feedback together. See the
[visual span analysis](../references/visual_chunk_coverage_result.md).

## Run the teacher stage

Prepare a new run with CUDA hidden. The current verified preparation is in
`artifacts/visual_sequence_teacher_v1/preparation.json`.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. python scripts/cache_visual_sequence_teacher_states.py \
  --stage prepare --run artifacts/visual_sequence_teacher_reproduction
```

For a new prepared run, execute its cache on an available project GPU0.
The optional `--maximum-shards` bounds the number of newly collected shards
in an attempt. A later invocation resumes the same preparation.

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. \
  python scripts/cache_visual_sequence_teacher_states.py --stage cache \
    --run artifacts/visual_sequence_teacher_reproduction
```
