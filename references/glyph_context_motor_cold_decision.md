# Fixed cold replay for the actual-context motor

The candidate's standalone inference entry point is
`scripts/infer_glyph_context_motor_image.py`. It takes one physical input
image, the authenticated parent/motor checkpoints, a fixed seed and an output
cap. It does not call the evaluation runner's training-data validation or load
a training manifest. It keeps the final EMA, frozen neural E/D and stopping,
one sampled motor trajectory per glyph and actual emitted-pixel feedback.
Code availability is not a passed runtime-independence audit.

Before inspecting any new candidate answer pixels, select the same first
fixed-cell normal book case used by the ancestor's cold control:
`fixed_cells__book_01_normal`. Use its original physical image, manifest seed
and128-glyph cap, with no supplied reference or target length. After completing
the registered generation stages, run that case in a new traced process using
the standalone entry point. Bind the source and both checkpoint hashes.

Compare every actual output pixel, contextual state, stop probability,
continuous feedback embedding hash, sampled noise and normalized motor feature
against the retained original case. Verify the displayed PNG and stopping
status. Inspect file and network access traces for forbidden training,
reference, vocabulary-weight, font, OCR, bank or teacher access. Source weights
and fixed numerical controls remain permitted and disclosed. An optional
downstream transcription must never alter output pixels.

Retain failed attempts and report setup/generation/pressure costs separately.
A successful replay is one execution and independence check; it cannot turn
an unreadable or incorrect answer into a language success. If the selected
case is capped or damaged, retain that exact failure. Do not choose another
case for a better demonstration. Keep one heavy project worker and the
existing full-reader resource ceiling; do not launch this replay while either
generation stage is live. Cold inference is not yet run or passed.

The CPU verifier is `scripts/verify_glyph_context_motor_cold.py`. It refuses
incomplete generation stages or a remaining project GPU owner, requires its
sources to be committed, and launches the standalone child from an empty
working directory with an empty offline model cache. The shared authenticated
source archive remains available. Every nested event/tensor is compared,
including the unused final sample produced at a cap. All other project
training/reference artifact opens are rejected except the exact input, two
checkpoints, fixed numerical controls and GPU mutex. The file trace cannot
alone prove tensor-level exclusion of vocabulary payloads inside the shared
archive; the bound loader and runtime provenance supply that separate check.

Run only after both generation processes have exited:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 PYTHONPATH=. \
  python scripts/verify_glyph_context_motor_cold.py \
  --output artifacts/glyph_context_motor_cold_v1
```

The verifier saves the invocation, child log, raw access traces, exact replay
outputs and verification receipt. An existing destination is preserved. Unit
checks exercise deep sampled-noise mismatches and directory-relative syscall
paths; those checks do not substitute for the actual GPU replay.

The first actual replay completes its selected book stream and reproduces all
nested tensors and PNG bytes, but fails the access gate. The common training
admission calls `owner_gone`, opening an old trainer's `status.json` and
`events.jsonl`. They are ownership metadata, but remain forbidden project
training artifacts under this protocol. Keep that attempt and its failed
verification; do not exempt those paths to pass the audit.

The standalone entry point now uses the same GPU mutex,48GiB entry reserve,
24GiB runtime reserve,20GiB free-device entry and22GiB allocation ceiling,
checking live processes and memory pressure instead of old training ledgers.
Only standalone admission changes; the candidate weights, image preparation,
sampler, seed, stopping, feedback and all216 completed streams are preserved.
Repeat the same fixed case into a new destination
`artifacts/glyph_context_motor_cold_v2`. A completed retry still needs its full
access/pixel audit before being described as passed.
