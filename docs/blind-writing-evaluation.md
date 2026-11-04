# Independent evaluation of generated writing

The ILM produces pixels. A separate vision reader can help measure whether
those pixels carry the intended answer, provided that reader first passes a
registered calibration. This diagnostic changes none of the primary V62
raw-image scores or qualification gates.

The [protocol](../references/v62_blind_vision_transcription_plan.md) fixes the
local Qwen3-VL 8B evaluator, weight hashes, instruction, decoding settings,
128 training-reference images, 16 blanks and 16 random-ink controls. It
requires at least 95% exact reference transcriptions, every blank empty,
at least 90% random-ink controls empty and valid JSON throughout. A failed
instrument produces a failed calibration report without scoring actor images.

Run only after both matched ILM arms, their offline deployment replays and the
locked image audit finish:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. \
  python scripts/audit_correlated_native_transcripts_v62.py
```

The harness refuses to start while the primary pipeline is active, GPU 0 is
occupied, available host memory is below 24 GiB, or swap use exceeds 75%.
It starts one owned Ollama server on loopback port 11437 using the existing
`../LocalLLM` model store. It verifies the cached model hashes and never pulls
or duplicates weights. The shared LocalLLM service is not used.

Recognition requests contain a fixed transcription instruction and anonymous
answer-image bytes. Source questions, reference answers, actor identities and
condition labels stay outside those requests. The harness uses Ollama's
[vision messages](https://docs.ollama.com/capabilities/vision) and
[JSON schema output](https://docs.ollama.com/capabilities/structured-outputs).
Raw responses, invalid JSON and transport errors are retained; failures never
become empty correct answers. Identical image/request hashes permit cached
resume without changing settings. A completed diagnostic is immutable.

Results go to
`artifacts/correlated_native_density_v62_20261105/vision_transcript_diagnostic/`.
The cleanup receipt records owned process identities, remaining live processes,
port state and GPU memory after stopping the owned server and runners. The
pretrained evaluator remains external to ILM inference and to its training
budget. Even a successful diagnostic establishes only answer legibility and
reference agreement on the registered cases.

Implementation checks on 5 November 2026: two payload/schema tests pass; both
scripts compile; an actual attempted invocation during K=8 training exits at
the pipeline guard before creating an evaluator service or making model calls.
The completed reader calibrates at 122/128 reference answers, with every blank
and random-ink control empty. Three invalid actor responses prevent its complete
fixed-case score. The [post-hoc sensitivity result](../references/v62_transcript_sensitivity_result.md)
reports bounded missing-score intervals without dropping cases or retrying
outputs. K=8 has zero exact transcript agreements and no resolved character
question gain. The evaluator server/runner exit and release GPU 0 memory.
