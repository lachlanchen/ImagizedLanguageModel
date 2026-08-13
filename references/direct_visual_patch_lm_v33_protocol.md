# Direct Visual Patch Language Model V33: Preregistered Protocol

Date frozen: 2026-08-13

Status: frozen before implementation and measurement

## Fixed Question

Can a roughly 116M-parameter causal student, initialized from the PIXAR
transformer core and trained on one RTX 4090, map rendered Chinese prompt pixels
to autonomously generated, prompt-dependent Chinese answer pixels without any
token, Unicode, OCR, vocabulary, glyph-table, codebook, VAE, or runtime teacher
inside the student path?

## Fixed Student Boundary

Allowed model inputs:

- `pixels: float[B,1,32,32*L]`;
- `patch_mask: float[B,L]`; and
- generation limits and scalar sampling controls.

Allowed model outputs:

- direct next-patch pixel logits;
- generated raster patches and concatenated strips;
- scalar stop probabilities; and
- continuous hidden states for diagnostics.

Forbidden in student methods, saved student batches, and deployed generation:

- strings, tokenizer calls, token IDs, byte IDs, Unicode values, character IDs;
- OCR, text decoding, vocabulary logits, glyph lookup, candidate retrieval;
- discrete visual codes or nearest-neighbor codebooks;
- target answers passed after autonomous generation starts; and
- calls to PIXAR, Qwen, or another external model at runtime.

Offline rendering, split construction, evaluator OCR, and evaluator candidate
matching are allowed and must be labeled evaluator-only.

## Fixed Architecture

- binary monochrome patches: `32 x 32`;
- hidden width: `768`;
- causal layers: `12`;
- attention heads: `12`;
- key/value heads: `12`;
- SwiGLU intermediate width: `3072`;
- RMSNorm epsilon: `1e-5`;
- RoPE base: `10000`;
- maximum total patches: `96`;
- direct input projection: `1 x 32 x 32 -> 768`;
- direct output projection: `768 -> 1 x 32 x 32`;
- scalar stop head: `768 -> 1`;
- no embedding table and no language-model head.

The selected initialization is PIXAR's 12-layer checkpoint with weight SHA-256
`ae4f899bbbb0bfaa90ee033c6d1dc5aeb3f50b323f726a0df241be8104682eb9`.
Core transformer tensors load exactly. Input and output projection tensors are
bicubically resized from `8 x 16` to `32 x 32`; this transformation and its
norm ratios must be recorded. The weights are a local research dependency and
must not be committed or redistributed.

## Fixed Data

- public continuation: `data/visual_grammar/chinese_wikisource_public_domain.jsonl`;
- instruction adaptation: `data/raw/alpaca_zh.json`;
- train/development/sealed partitions are deterministic SHA-256 partitions;
- development and sealed fonts are absent from training;
- answer pixels begin on a patch boundary, but text inside each region receives
  a deterministic 0--31 pixel horizontal offset;
- all student tensors are produced after rendering and contain no text labels.

The instruction source makes the selected research checkpoint non-commercial.
Every receipt must preserve source and rights metadata.

## Fixed Optimization

Evidence configuration:

| Stage | Updates | Effective batch | Core state | Adapter LR | Core LR |
|---|---:|---:|---|---:|---:|
| visual calibration | 2,000 | 64 | frozen | `3e-4` | `0` |
| public continuation | 8,000 | 64 | trainable | `1e-4` | `1e-5` |
| instruction continuation | 12,000 | 64 | trainable | `8e-5` | `8e-6` |

Common settings:

- AdamW, betas `(0.9, 0.95)`, weight decay `0.05`;
- BF16 on one RTX 4090;
- gradient norm clip `1.0`;
- 500-update linear warmup and cosine decay within each causal stage;
- fixed seed `20_263_300`;
- EMA `0.999` over trainable parameters;
- answer-only causal loss for instruction records;
- pixel BCE plus `0.25` Sobel edge loss and `0.25` ink Dice loss;
- stop BCE weight `0.10`;
- no evidence run may exceed 20 GiB allocated CUDA memory.

Smoke and exploratory runs must be labeled and cannot satisfy the update-count
gate.

## Mandatory Stage-A Gate

Before Stage B, reconstruct at least 2,048 held-out patches containing unseen
text and development fonts. Continue only if all conditions hold:

- finite outputs and checkpoint;
- mean binary pixel F1 at least `0.90`;
- mean edge F1 at least `0.90`;
- evaluator-only OCR character accuracy at least `0.95` on reconstructed strips;
- blank-patch false-ink rate below `0.01`; and
- PIXAR core parameters remain byte-identical to initialization.

Failure stops the evidence run. Lowering the gate requires a new protocol.

## Final Evaluation

Evaluate raw and EMA weights on the full development split, then evaluate the
selected route once on the sealed split. Report:

- autonomous OCR character accuracy, CER, exact answer accuracy, and length
  accuracy;
- pixel F1, edge F1, blank rate, and ink coverage;
- correct-prompt, shuffled-prompt, blank-prompt, and suffix-only results;
- held-out wording/paraphrase and held-out font results;
- prompt-answer counterfactual pair assignment;
- teacher-forced next-patch accuracy as a diagnostic only;
- random-core initialization control;
- generation examples selected by deterministic sample IDs, never by visual
  appeal;
- parameter count, update time, examples and patches seen, peak VRAM, and
  generation latency; and
- model, source, data, and upstream checkpoint hashes.

## Acceptance Gates

Boundary and integrity:

- all fixed update counts completed;
- finite model and autonomous generation;
- peak allocated CUDA memory below 20 GiB;
- all forbidden student-boundary fields absent;
- autonomous generation signature accepts pixels and masks only;
- external checkpoint absent as a runtime dependency.

Raster generation:

- nonblank answer rate at least `0.95`;
- autonomous OCR character accuracy at least `0.25`;
- autonomous CER below `0.90`;
- length accuracy at least `0.50`;
- autonomous character accuracy no more than `0.15` below teacher-forced
  character accuracy.

Prompt-conditioned language:

- exact accuracy exceeds the most-frequent-answer baseline by at least five
  percentage points;
- correct prompts exceed shuffled prompts by at least ten character-accuracy
  points;
- correct prompts exceed blank prompts by at least ten character-accuracy
  points;
- paraphrase character accuracy is at least `0.70` of original-prompt accuracy;
- counterfactual pair assignment exceeds `0.65`; and
- held-out-font character accuracy is at least `0.70` of the training-font
  diagnostic.

Passing only reconstruction, teacher forcing, or raster gates is not evidence
of language. Any unavailable required control makes the run incomplete rather
than positive.

## Decision Labels

- `accepted-bounded-proof`: every integrity, raster, and language gate passes;
- `readable-not-semantic`: integrity and raster pass, language fails;
- `semantic-not-autonomous`: prompt controls pass only with teacher forcing;
- `visual-interface-failure`: mandatory Stage-A gate fails;
- `invalid-run`: protocol, boundary, finite-state, resource, or required-control
  conditions fail.

