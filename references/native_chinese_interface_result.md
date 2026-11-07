# Chinese geometry recovery does not recover answer correctness

The frozen HQ source gives **0/32 exact Chinese answers** on both the original
1024-square pages and the native 432-square resize. The same frozen language
body answers **16/32** exactly from the original visible text. All sixteen
text-control passage pairs receive different responses and four pairs have
both answers correct. These are auxiliary text-head observations, not an
independent image-output ILM.

| Fixed Chinese passage-pair assessment | Original image 1024 | Image 432 | Original text |
| --- | ---: | ---: | ---: |
| Normal exact stopped /32 | 0 | 0 | 16 |
| Normal stopped /32 | 0 | 29 | 32 |
| Both questions correct /16 | 0 | 0 | 4 |
| Different normal outputs /16 | 0 | 8 | 16 |
| Normal outputs changed by question erasure /32 | 0 | 26 | 32 |
| Shared erased output matching either reference /16 | 0 | 0 | 0 |

The original image condition repeats an unrelated numeric answer and reaches
the 96-decision cap on every normal question. At 432 the responses contain
Chinese and source-related fragments, but their facts remain wrong. For
example, the first fish-distribution question receives `Pondicherry` instead
of `印度西太平洋区`; the next question receives `13元` instead of
`印度Pondicherry`. The original-text control answers both exactly. Changing
the output from repeated numbers to plausible Chinese does not qualify
understanding.

All three arms preserve the same sixteen held-out passage pairs, 32 questions,
sixteen shared question-erased conditions and per-passage latent/readout seeds.
No optimizer or learned adapter is used. The text control reconstructs the
visible transcript and verifies its hash against the original page manifest;
it supplies no annotated answer. It is a privileged modality control with its
own generic text instruction, not the deployed image learner or an exactly
identical multimodal interface. The model retains external pretraining and
its Unicode output head.

## Readability and codec reconstruction

Independent Tesseract `chi_sim+eng`, PSM 6, with one CPU thread gives the
following complete-page Levenshtein character errors after NFC/Chinese
whitespace normalization:

| Thirty-two complete normal pages | Character error | Exact full transcripts |
| --- | ---: | ---: |
| Original 1024 images | 0.089771 | 0/32 |
| Whole-square 432 resize | 0.084320 | 0/32 |
| Native sampled FP32 codec reconstructions at 432 | 0.705221 | 0/32 |

The first resized/reconstructed pages were visually inspected. Reconstruction
introduces visibly broken or confused fine strokes. This is a reconstruction
readability failure, not a proof of how much information remains in the
encoder's latent state. Recognition remains imperfect on the clean pages too.
No recognition output enters model conditioning, chooses cases or repairs
the generated responses.

The source contains 2,830,777,056 parameters plus a 126,892,531-parameter Wan
codec, with no new parameters. Reading uses BF16 body weights, a sampled FP32
codec and cached top-one decoding. Complete original-image, native-image and
original-text arms take 109.830, 23.708 and 9.539 seconds excluding loading.
The native-image timing includes saving 32 codec reconstructions and is not
a like-for-like inference-speed comparison. Allocated peaks are 11.129,
7.242 and 5.830 GiB on one RTX 4090 D.

## Evidence and next test

The [complete export](evidence/native_chinese_interface_result/result.json)
binds 89 files: 48 resized pages, 32 exact-latent codec reconstructions,
all 144 native responses and reports, full source transcripts, separate
references, 96 raw OCR observations and the executed script. Original pixels
remain bound to the preceding public input evidence. All failures are kept.
The [protocol](native_chinese_interface_protocol.md) precedes model calls.

```bash
export ILM_NATIVE_PYTHON="$PWD/artifacts/external/native_unified_source/runtime/bin/python"
env CUDA_VISIBLE_DEVICES= PYTHONPATH=. python scripts/probe_native_chinese_interface.py --stage prepare --output artifacts/native_chinese_interface/prepared_v1
env CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=. "$ILM_NATIVE_PYTHON" scripts/probe_native_chinese_interface.py --stage evaluate --output artifacts/native_chinese_interface/source_v1
env CUDA_VISIBLE_DEVICES= PYTHONPATH=. python scripts/probe_native_chinese_interface.py --stage score --output references/evidence/native_chinese_interface_result
```

Output paths must be fresh. GPU work is sequential under the project lock.
CMRC passage images preserve CC-BY-SA-4.0 attribution to Cui et al.,
EMNLP-IJCNLP 2019; Show-o2 source/model are NUS Show Lab, Apache-2.0. These
repeatedly inspected public cases do not constitute a sealed benchmark.

The English resolution recovery cannot be assumed to transfer to Chinese.
Before further optimization, the next control should retain full-resolution
glyph pixels and test visual encoding within native spatial windows. A
text-input success identifies a modality gap but does not specify a successful
repair. Independent continuous image-answer generation remains unachieved.
