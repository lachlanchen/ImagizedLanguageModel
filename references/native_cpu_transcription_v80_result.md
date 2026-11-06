# CPU recognition fails calibration on the current writing format

6 November 2026. The fixed Tesseract 5.3.4 diagnostic fails its calibration
criteria. Neither Chinese nor English is qualified for actor transcript
scoring. No ILM answer images were submitted to this recognizer, and no
semantic accuracy result follows from this experiment.

| Calibration input | Exact normalized transcript | Successful requests |
| --- | ---: | ---: |
| Chinese training reference | 30/128 (23.438%) | 128/128 |
| English training reference | 78/128 (60.938%) | 127/128 |
| Blank strip, expected empty | 15/16 | 16/16 |
| Random ink, expected empty | 12/16 | 16/16 |

The registered reference threshold is 95% for each language. All blanks must
be empty, at least 90% of noise controls must be empty, and recognition must
succeed on every calibration input. The failed English request reports
`Image too large: (32800, 160)`. That input and failure remain in the
denominator. No image was cropped, omitted or retried with different settings.

The 288 cases contain 287 unique images. The completed attempt takes 49.557
seconds including preparation, uses two CPU workers and initializes no CUDA
context. All 287 recognition responses are retained. An independent replay
of the raw TSV reproduces every calibration count; all 288 enlarged images
are checked against their original target or negative-control pixels.

The first attempt failed because the response-cache directory had not been
created. Its identity, original source and failure record are retained. No
valid calibration aggregate was produced; some unpersisted OCR responses may
have been lost. The one-line directory fix uses a separate output directory
with the identical calibration manifest, recognizer settings and thresholds.
Its producing revision is `b74c498`.

The [protocol](native_cpu_transcription_v80_protocol.md),
[implementation](../scripts/probe_native_cpu_transcription_v80.py) and
[verification receipt](evidence/native_cpu_transcription_v80.json) bind the
data, executable, language weights, source revisions and saved response hashes.
The original GPU training, generated images and physical-image gates are
unchanged. The tested CPU instrument is unsuitable for language scoring in
this format. A neural recognizer would need its own calibration after the
active GPU run and original audit finish.

The completed run used:

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 PYTHONPATH=. \
artifacts/external/mixar_env_py312/bin/python scripts/probe_native_cpu_transcription_v80.py \
  --run artifacts/native_chinese_v80_20261106 \
  --out artifacts/native_chinese_v80_20261106/cpu_transcription_v80r2
```

Completed output directories are immutable. Replication requires a new
`--out` directory; the failed first attempt also remains preserved.
