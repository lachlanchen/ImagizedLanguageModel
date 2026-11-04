# V62 answer-only vision transcription diagnostic

Registered before examining any V62 development output. K=1 training has
completed; K=8 is still training. The locked V62 raw-image audit and its
decisions remain unchanged. This is an auxiliary instrument, not a replacement
metric selected after model results. V61's Tesseract reference calibration was
only 40/256, so its transcript scores were not a reliable semantic evaluation.

## Calibrate the instrument first

Use the cached local `qwen3-vl:8b-instruct-q4_K_M`, manifest SHA-256
`0533d74300e4f9bc367d675d4e64ffd073d50ff16a2b4096cc2e8a1cf8c96319`,
model layer SHA-256
`1329cd5ab37e5ef6faef37cd8ff8660ffc96ed5b47b34fddf8eb0912e6c82466`.
Reuse the LocalLLM model store without pulling or duplicating weights.
This pretrained evaluator is external to the ILM and is not part of its actor
or its local training budget.

Prepare 128 training-only reference images. Group the original 3,211 CMRC
training questions by article title; sort groups and records, sample 128 groups
without replacement using NumPy seed 20266370, then one record per group using
the same RNG. Render the first retained answer with the unchanged V62 canonical
Noto-26 interface. Add sixteen all-white strips and sixteen independent random
ink strips (probability .15), lengths one through sixteen cells, seed 20266372.
These negative controls test guessing on absent/unstructured writing.

Pack actual source image cells with no gutter, enlarge fourfold with nearest
neighbour and add sixteen white pixels on every side. The transformation
preserves every input bit. Give the evaluator only those images and the fixed
instruction below, with anonymous content-hash filenames. No source question,
paragraph, reference text, model name or arm label enters its request.

> Transcribe only the writing visible in the image. Copy the characters and
> punctuation exactly. Do not answer a question, complete missing writing, or
> guess damaged characters. Return only a JSON object with keys "text" and
> "readable". If there is no legible writing, return {"text":"","readable":false}.

Use temperature 0, top_p 1, seed 20266371, context 2048, maximum 256 generated
tokens, thinking disabled and one request at a time. Require a valid JSON
object with a string `text` and boolean `readable`. Do not retry with changed
prompts or parameters based on calibration performance. Normalize transcripts
and gold with NFC and whitespace removal only; preserve punctuation and case.

Calibration requires >=95% exact reference transcripts, all sixteen blank
controls empty, >=90% random-ink controls empty, and valid JSON throughout.
Otherwise report an invalid instrument and do not promote its development
transcripts into a language-quality claim. Keep raw responses and errors.
Choosing a different reader/settings requires a new calibration protocol.

## Independent resource and leakage boundaries

Run the evaluator only after both ILM arms, export/replay and the locked audit
finish. Launch a separate owned Ollama process on physical GPU 0 and an unused
loopback port, with `CUDA_VISIBLE_DEVICES=0` and the existing shared model store.
Verify the pipeline is terminal and GPU 0 is free first. Never send inference
to the existing shared LocalLLM/Ollama service, which may occupy GPU 1.
Keep at most one owned evaluator instance; stop its exact server/runner after
capturing evidence and verify GPU memory release. Do not keep an idle service.

## Auxiliary V62 analysis, conditional on calibration

Freeze the calibrated instrument before inspecting actor images. Anonymize and
deduplicate the raw primary images from K=1, K=8 and their blank-question
conditions; the evaluator receives no arm or target labels. Empty actor tensors
are empty writing, not the white display placeholder treated as a glyph.
Do not repair pixels. Associate transcripts with source records only after
recognition. Compare with all retained answer references using exact normalized
text, normalized Levenshtein error and bounded character accuracy. Use the same
256 primary IDs and article-grouped paired bootstrap, 2,000 draws, seed 20266379.
Record normal-minus-blank and K8-minus-K1 differences, raw outputs, evaluator
identity, resources and its own calibration. Do not score a new sealed test.

This can assess legibility and reference-answer agreement under an audited
reader; it cannot establish general reasoning, etymology competence, historical
script understanding or model parity. It changes no registered V62 primary
gate, rejection or pass. The ILM's deployed inference remains independent of
OCR, this vision model, source text and any external language-model call.
