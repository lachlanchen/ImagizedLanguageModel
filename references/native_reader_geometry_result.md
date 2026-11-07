# Native image geometry changes the source reader's answers

**A verified interface problem contributes to the repeated garbage outputs.**
The frozen Show-o2 HQ source answers 0/32 English passage questions exactly
on the original 1024-square inputs. Resizing the same complete pages to the
official MMU demo's 432-square geometry raises exact stopped answers to
**11/32**, without training, replacement cases, OCR conditioning or another
checkpoint. This is inherited reading capability recovered by an interface
change. It does not demonstrate a trained independent image-answer ILM.

| Same English questions and matched erasures | Source, 1024 | Source, 432 | 4,096-update adapters, 432 | Dense curriculum, 432 |
| --- | ---: | ---: | ---: | ---: |
| Normal exact stopped answers /32 | 0 | 11 | 5 | 5 |
| Normal answers reaching EOS /32 | 3 | 32 | 27 | 26 |
| Both questions correct /16 pairs | 0 | 3 | 1 | 1 |
| Different normal outputs /16 pairs | 3 | 14 | 15 | 14 |
| Normal answers changed after question erasure /32 | 13 | 30 | 31 | 28 |
| Exact answers with passage erased /32 | 0 | 2 | 0 | 0 |

Nine of the eleven correct source answers cease to be correct when the
passage is erased. Two remain correct without it. The latter cannot establish
use of the passage. Different outputs and question sensitivity alone also do
not establish correctness. The existing learned adapters score below the
frozen source at native geometry; they are not a validated repair.

## What changes, and what stays fixed

The official pinned image transform bicubically resizes the shorter edge to
432 and centre-crops a square. Since every original English input is square,
the crop is the identity. Every source/control pixel participates in the
resize. No answer span is cropped, repositioned or supplied to inference.
The prepared index records both original and transformed SHA256 values.

Native 432 inputs have 729 visual positions. The original 1024 interface
interpolates positions to 4096. That interface runs numerically, but its
language quality was not established before adapter learning. This diagnostic
changes image pixels and the spatial sequence jointly; it does not isolate
position embeddings from scale, sequence length or other geometric effects.
It identifies an effective interface correction on this source/cohort, not
the unique cause of every earlier model failure.

All four arms use the same source weights, BF16 body, FP32 native Wan codec,
input-only generic instruction, per-passage latent/readout seeds and top-one
96-decision readout. Each arm includes all eighty original normal/erased
conditions. The unadapted arms have no new parameters or optimizer steps.
The adapters are the two completed checkpoints, evaluated without more
training. Their original 1024-square results remain [separate](native_visual_alignment_result.md).

Original source-page OCR has 0.001522 character error. After resizing, the
32 complete normal pages have **0.004207** error; sampled native-codec
reconstructions have **0.006210**. Full-page OCR is exactly equal on 10/32
resized inputs and 0/32 reconstructions: low character error is not perfect
fidelity. These offline checks do not enter model inputs or filter cases.
All source and reconstructed pages are retained.

The source's eighty calls take 131.576 seconds at 1024 and 14.872 seconds
at 432, excluding loading, with peak allocated VRAM of 11.129 and 6.738 GiB.
The diagnostic runs on one RTX 4090 D. This comparison concerns this
resolution change, not parity or language-efficiency superiority over an LLM.
The source is 2,830,777,056 parameters plus a 126,892,531-parameter codec
and inherited external pretraining.

## Reproduce the completed diagnostic

Reuse the [existing source setup](../docs/native-visual-access.md),
[English preparation](../docs/native-english-transfer.md) and completed
adapter checkpoints. Use fresh output paths. GPU invocations must remain
sequential, after the preceding job has terminated.

```bash
export ILM_NATIVE_PYTHON="$PWD/artifacts/external/native_unified_source/runtime/bin/python"
env CUDA_VISIBLE_DEVICES= PYTHONPATH=. python scripts/check_native_reader_geometry.py --stage prepare --output artifacts/native_english_transfer/native432_prepared_v1
env CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=. "$ILM_NATIVE_PYTHON" scripts/probe_native_unadapted_reader.py --output artifacts/native_english_transfer/unadapted_hq_v1
env CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=. "$ILM_NATIVE_PYTHON" scripts/probe_native_unadapted_reader.py --inputs artifacts/native_english_transfer/native432_prepared_v1/inputs.json --output artifacts/native_english_transfer/unadapted_hq_native432_v1
env CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=. "$ILM_NATIVE_PYTHON" scripts/check_native_reader_geometry.py --stage codec --output artifacts/native_english_transfer/native432_codec_v1
env CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=. "$ILM_NATIVE_PYTHON" scripts/train_native_visual_alignment.py --stage evaluate --inputs artifacts/native_english_transfer/native432_prepared_v1/inputs.json --checkpoint artifacts/native_visual_access/learning_v1/checkpoint.pt --output artifacts/native_english_transfer/predecessor_native432_v1
env CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=. "$ILM_NATIVE_PYTHON" scripts/train_native_visual_alignment.py --stage evaluate --inputs artifacts/native_english_transfer/native432_prepared_v1/inputs.json --checkpoint artifacts/native_visual_alignment/learning_v1/checkpoint.pt --output artifacts/native_english_transfer/dense_native432_v1
env CUDA_VISIBLE_DEVICES= PYTHONPATH=.:scripts python scripts/score_native_reader_geometry.py --output references/evidence/native_reader_geometry_result
```

The [protocol and post-source amendment](native_reader_geometry_protocol.md)
precede their respective calls. The [complete export](evidence/native_reader_geometry_result/result.json)
binds 128 files: all 320 model responses and reports, eighty resized input
pages, 32 codec reconstructions, OCR observations and executed source scripts.
Original inputs and annotated alternatives are bound to the existing English
export. The scorer checks every condition, input hash and seed, and retains
the existing strict answer criteria. No post-result semantic rescoring replaces
those criteria.

SQuAD 1.1 passages retain CC-BY-SA-4.0 attribution to Rajpurkar et al.,
EMNLP 2016. The public benchmark may be present in source pretraining; this
is a small diagnostic, not an independent benchmark replication. Show-o2
source/model are NUS Show Lab, Apache-2.0. No Chinese native-scale transfer,
new image-answer training or vocabulary-free inference is established here.

The next learning run requires a calibrated image interface and a frozen
source comparison. Neither more original-scale supervision nor attaching
another writer is justified as an established solution. Useful independent
image-input/image-output language generation remains the open requirement.
