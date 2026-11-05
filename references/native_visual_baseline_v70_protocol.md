# V70: native source-model capability on fixed visual questions

Registered before preparing cases or making source-model requests. V68 and V69
remain failed image-language results. The earlier eight-case native source
check covers text only. This diagnostic asks whether the authenticated source
VLM answers the same written-image questions through its original interface.
It changes no ILM weights, architecture, output or qualification decision.

## Fixed cases and conditions

Use the first 16 Chinese cases in the existing V68 audit, comprising eight
source-article pairs, and the lexicographically first case in each of its eight
English categories. Select by the original fixed audit order/category, never
by source-model or actor scores. All 24 are reused development cases. Sealed
data remain unopened, and none of these diagnostic answers may be used as
training targets.

Prepare complete RGB PNGs from the exact normal and matched-layout blank
actor pages. Check their bits against the independent NumPy layout from the
V68 verifier. Never resize or crop the client images. The native server's own
image preprocessing remains part of the source interface; it is not assumed
identical to the custom ILM reader. Bind matched source strings and retained
reference strings to the authenticated development source files and existing
audit construction. References are for offline scoring only.

For each fixed case, request four routes in order:

1. `image`: normal PNG with one fixed external task instruction;
2. `image_only`: the same PNG with empty text content;
3. `blank`: the original blank-question PNG with the same task instruction;
4. `text`: the exact source-formatted prompt as text, with that instruction.

All routes use the same native chat API and JSON schema containing only a
string `answer` field. The fixed instruction is:

> Read the question or instruction in the supplied material. Answer it using
> any accompanying passage. Return only the requested answer, without an
> explanation or transcription of the full material. If there is no question
> or instruction, use an empty answer.

The request constructor accepts only route, input PNG bytes and/or source
prompt text. It cannot receive references, case identifiers or model outputs.
The source API receives no reference answer, filename or score. The schema and
native chat template provide source-model scaffolding even for `image_only`;
this route is not claimed equivalent to the ILM's learned structural queries.

## Source, resources and execution

Use exactly the existing shared `qwen3-vl:8b-instruct-q4_K_M` archive, manifest
SHA-256 `0533d74300e4f9bc367d675d4e64ffd073d50ff16a2b4096cc2e8a1cf8c96319`
and model SHA-256
`1329cd5ab37e5ef6faef37cd8ff8660ffc96ed5b47b34fddf8eb0912e6c82466`.
Do not pull, copy or replace weights. Start one isolated owned Ollama service
on `127.0.0.1:11437`, physical GPU 0, using the existing lifecycle helper.
The shared service on port 11434 and other projects' GPU 1 work remain untouched.
Require available host RAM at least 24 GiB, swap below 75%, and GPU 0 below
256 MiB before startup. Keep sampled source residency below 20 GiB and stop
the exact owned process tree after capture, verifying port closure and memory.

Use temperature 0, top-k 1, top-p 1, repetition penalty 1, seed 20267001,
4096 context positions, and a uniform maximum of 512 generated source tokens.
Disable thinking and streaming. These tokens belong to the offline source
helper, not ILM inference. Preserve all 96 full requests and raw responses,
request/image hashes, timing, termination reason and measured GPU residency.
Invalid JSON or length termination remains a recorded failure. Transport
errors stop the run and preserve completed work; they are not empty answers.
Partial request reuse requires exact identity. Completed results are immutable.

## Scoring and interpretation

Score on CPU after the owned source server exits. Report valid JSON, completed
termination, exact agreement after trimming only outer whitespace, and bounded
character edit accuracy against all retained references. Character accuracy is
`max(0, 1 - edit_distance / max(reference_length, answer_length))`, choosing the
best recorded reference. Exact agreement also requires completed generation.
Do not normalize spelling, remove punctuation or reinterpret answers after
seeing them. Invalid responses receive no exact or character credit. English
reference agreement is not exhaustive semantic correctness.

Report all counts by source and route, and both-exact outcomes for the eight
Chinese pairs. With this small reused sample, make descriptive claims only.
Independently reconstruct inputs and requests, parse raw responses, recompute
scores, and inspect cleanup. Do not use source logits or answers as evidence
that the ILM itself can generate them.

Native image success would show source capability accessible through that
native interface and motivate a controlled transfer/interface intervention.
Text success with image failure would motivate auditing visual preparation
and task alignment. Neither result identifies a unique repair, proves feature
sufficiency, or qualifies an independent image-language model. The final actor
must still operate from pixels to generated pixels without a teacher call,
OCR, vocabulary head, source strings or a character inventory.
