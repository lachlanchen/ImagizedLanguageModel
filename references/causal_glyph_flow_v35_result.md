# Causal Glyph Flow V35: Measured Result

Date completed: 2026-08-14

Decision: `not-qualified`

Sealed split: **not opened**

## Claim Tested

V35 tested whether a transferred causal pixel-language prior, a qualified
continuous writing codec, and a continuous anchor/flow writer could form an
independent raster-input/raster-output Chinese language model on one RTX 4090.
The student received only binary writing patches and a visual-position mask.
Its primary output was a generated binary raster, and every generated patch was
decoded, thresholded, and re-encoded before entering later causal context.

This was a transfer-based mechanism test, not from-scratch pretraining and not
a claim of parity with a general text LLM. PIXAR supplied the causal
transformer initialization; V34 supplied the frozen continuous codec. Their
origin remains explicit and neither result is relabeled as this project's work.

## Frozen Training Run

- protocol: `references/causal_glyph_flow_v35_protocol.md`;
- protocol SHA-256:
  `d7a4d49270676cd82c55e22ddd73466966e0b96723970f76fe66fa2381bd3718`;
- model parameters: `129,092,738`, including the frozen
  `7,423,361`-parameter V34 codec;
- causal core: 12 layers, width 768, 12 attention heads, 96-patch maximum
  context;
- writer: deterministic latent anchor plus a 3-block, width-512 conditional
  rectified-flow head;
- precision: BF16 with gradient checkpointing;
- physical batch: 8, with 8-way gradient accumulation;
- seed: `20263500`;
- updates: `22,000 / 22,000`;
- training time: `9,972.458` seconds (`2.770` hours);
- peak allocated CUDA memory: `3,112,735,744` bytes (`2.899 GiB`);
- checkpoint:
  `artifacts/causal_glyph_flow_v35_20260814/checkpoint_latest.pt`;
- checkpoint SHA-256:
  `ca30872ffdc84d3719068d27ad456da9629428eed6a37ca9eaf62f40c3acb0b1`;
- training-summary SHA-256:
  `51755958dc30bd556802b5aa3a8cbe2fcaddefe1a7e3da99dcab917fa0f7981c`;
- all 22,000 JSONL metric rows were finite; and
- training ended normally rather than by signal.

The three stages completed exactly as planned:

| Stage | Updates | Examples |
| --- | ---: | ---: |
| Visual-interface alignment | 2,000 | 128,000 |
| Public causal continuation | 8,000 | 512,000 |
| Instruction and copy | 12,000 | 768,000 |

Stage C contained 576,000 instruction, 96,000 deterministic copy, and 96,000
public-continuation examples. The public manifest contained 6,743 train, 135
development, and 139 sealed records. The instruction manifest contained 5,747
train, 159 development, and 202 sealed records.

## Alignment Gate

Stage A passed before language training continued. On 2,461 held-out serif
patches, mean adapter MSE was `0.008510` against a maximum of `0.035`, and mean
cosine similarity was `0.979348` against a minimum of `0.95`. Codec ink and edge
F1 were `0.998201` and `0.997789`. The causal core and codec hashes remained
unchanged during alignment, and the runtime-boundary audit passed.

## Optimization Diagnostics

Training objectives improved, but these values did not decide qualification.
The table reports means over the first and last 200 updates of each language
stage.

| Stage/window | Loss | Anchor cosine distance | Flow MSE | Visual loss | Stop loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| Public, first 200 | 3.973425 | 0.952614 | 1.921636 | 2.494742 | - |
| Public, last 200 | 2.295065 | 0.665218 | 1.071353 | 0.903651 | - |
| Instruction/copy, first 200 | 2.422177 | 0.711615 | 1.076310 | 1.003673 | 0.275562 |
| Instruction/copy, last 200 | 2.218105 | 0.638079 | 1.033725 | 0.901127 | 0.020145 |

These reductions show that the objectives optimized. They do not establish
correct autonomous language generation.

## Evaluation Integrity

The evidence evaluator used the completed checkpoint, fixed development data,
EMA as the primary state, and seed `20263535`. It evaluated:

- all 135 public, 128 copy, and 159 instruction teacher-forced examples;
- 32 public, 32 copy, 32 instruction, and 31 paraphrase autonomous cases;
- correct, cyclically shuffled, blank, and final-quarter-only prompt
  conditions;
- 16 copy counterfactual pairs;
- anchor and eight-step flow writers; and
- raw weights as a diagnostic control.

The audit took `586.475` seconds and used `728,368,128` bytes (`0.678 GiB`)
peak allocated CUDA memory. The checkpoint audit, finite-state audit, strict
pixel-only boundary, and visible decode-threshold-reencode feedback receipt all
passed. The report is evidence eligible.

- development report SHA-256:
  `3d14e15d6f4f8677de864a793cb76efcb35a9294e1c5f7d28a647b66aa6617ba`;
- report self-excluded SHA-256:
  `77a8b5213f7a90919142401503c7b987f67016c37c1e506169f3b37a738d6122`;
- Tesseract: 5.3.4, `chi_sim+chi_tra`, post-hoc evaluator only; and
- development renderer: Noto Serif CJK Regular, SHA-256
  `93069d8e9e45d515cc421c971a79e6a5777704b348e36a9ef86578bf58adef77`.

The smoke audit first exposed a BF16 bookkeeping defect: a BF16 stop
probability was assigned to an FP32 report buffer. Commit `85652de` casts that
reported value to FP32. The change does not modify the stop comparison,
generated pixels, model weights, or completed checkpoint. A mixed-dtype
regression test was added, the model source was added to evaluator receipts,
all 34 V35 tests passed, and smoke was rerun in a fresh directory before this
production audit.

## Writer Selection

EMA anchor character accuracy on the fixed writer-selection set was `0.4241%`
with `78.125%` nonempty-OCR readability. Flow character accuracy was `0%` with
`87.5%` nonempty-OCR readability. Flow required at least a 3 percentage-point
accuracy gain without losing more than 2 points of readability, so it was not
promoted. All primary results therefore use the anchor writer.

The term `readable` in this evaluator means that post-hoc OCR returned a
nonempty string. It does not mean that the generated sentence was correct or
that human readability was established.

## Development Results

### Teacher-forced prediction

| Stream | Anchor cosine | Decoded ink F1 | Decoded edge F1 | Flow MSE |
| --- | ---: | ---: | ---: | ---: |
| Public | 0.266929 | 0.339685 | 0.475395 | 1.138855 |
| Copy | 0.286852 | 0.327055 | 0.451494 | 1.115074 |
| Instruction | 0.287481 | 0.329959 | 0.465625 | 1.100967 |

Public decoded ink and edge F1 both missed their fixed `0.70` gates.

### Autonomous generation

| Stream/condition | OCR character accuracy | Nonempty OCR | Nonblank raster | Target ink F1 | Target edge F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Copy, correct | 0.3125% | 87.50% | 100% | 0.09214 | 0.15495 |
| Copy, shuffled | 0.1953% | 87.50% | 100% | 0.08645 | 0.15427 |
| Copy, blank | 0% | 84.38% | 100% | 0.07670 | 0.13437 |
| Public, correct | 0% | 71.88% | 100% | 0.21265 | 0.32411 |
| Instruction, correct | 0.1116% | 75.00% | 100% | 0.17346 | 0.29229 |
| Instruction, shuffled | 0.5565% | 75.00% | 100% | 0.13709 | 0.23488 |
| Instruction, blank | 0% | 100% | 100% | 0.14888 | 0.24943 |
| Paraphrase, correct | 0.3002% | 87.10% | 100% | 0.15300 | 0.26212 |

The copy target-through-codec OCR ceiling was only `63.676%`, below the fixed
`70%` audit floor. Generated copy accuracy retained only `0.491%` of that
ceiling, versus the required `60%`. Correct copy prompts improved OCR accuracy
over shuffled prompts by only `0.117` percentage point and over blank prompts
by `0.313` point, against fixed margins of 20 and 25 points. Copy
counterfactual target preference was `62.5%`, below the `75%` gate, and mean
counterfactual character accuracy was zero.

The instruction target-through-codec ceiling passed at `78.858%`. Autonomous
correct-prompt accuracy was only `0.112%`, below the `8%` gate. It was `0.445`
percentage point worse than shuffled prompts and only `0.112` point above blank
prompts, missing the required `+2` and `+3` point margins. The output changed
under shuffled and blank prompts, and the wording-shift diagnostic fired, but
those changes were not bound to the correct answer.

## Gate Decision

The visual-causal route passed `5/12` checks. The semantic-raster route passed
`5/9` checks. The exact decision is `not-qualified`.

What V35 establishes:

- a 129M-parameter raster-input/raster-output student can be trained and run on
  one RTX 4090;
- the packaged runtime is independent of tokenizers, Unicode IDs, OCR,
  retrieval, codebooks, and teacher calls;
- continuous anchor and flow writers can close the visible raster feedback
  loop; and
- generated pixels respond to prompt interventions and remain nonblank.

What V35 does not establish:

- correct visual copying;
- causal prompt-to-answer binding;
- semantic Chinese instruction following;
- useful autonomous language generation;
- Qwen/GPT parity; or
- compute efficiency over a text-token language model.

The failure is not “no pixels were generated.” It is a binding failure:
generated raster strips contain corrupted glyph-like marks, and their content
is usually unrelated to the requested target. Lower training loss and
nonempty OCR therefore cannot substitute for semantic counterfactual evidence.

## Sealed and Standalone Decisions

Because development did not reach `visual-causal-qualified`, the sealed split
was not opened and no sealed images were rendered.

A standalone EMA artifact was exported only with the explicit
`--allow-unqualified` diagnostic override:

- path:
  `artifacts/causal_glyph_flow_v35_20260814/ilm_v35_ema_standalone.pt`;
- SHA-256:
  `eaf8a52398f82237d79d9f2cdaa7b4933f4b5a4e51872c647d63dd136bef099d`;
- model-state SHA-256:
  `e03ac2ae166b16e71bf249e80abaa9daf1970991db5f1e8ecbe94dec119d7ffd`;
- embedded status: `not-qualified`;
- selected writer: `anchor`;
- clean standalone-state audit: passed; and
- redistribution authorized: false.

The redistribution flag is false because the local PIXAR weight archive does
not state a weight license. The source revision is MIT, but source and weight
licensing are recorded separately.

## Packaged Inference Diagnostics

The standalone artifact completed both supported routes:

1. rendered text prompt `照写：天地 答：` to a 23-patch PNG; and
2. that generated PNG as a native image prompt to a 31-patch PNG.

Both outputs and feedback states were finite, and OCR was absent from model
inference. Optional OCR sidecars were created afterward and confirmed that the
content was incorrect. These diagnostics validate packaging and the image
interface, not language capability.

## Next Mechanism Decision

Do not scale this exact objective. V35 shows that a reconstruction-qualified
local glyph latent plus transferred causal structure is not enough: the writer
must learn an answer-level semantic plan before local raster realization.

The next experiment should preserve the strict raster boundary and external
foundation transparency, but separate:

1. a prompt-conditioned, multi-patch visual semantic plan trained with strong
   answer-level contrastive and counterfactual objectives;
2. a representation-rich visual encoder rather than a reconstruction-only
   glyph codec as the sole predictive target; and
3. a renderer conditioned on that plan, with explicit sequence-level
   consistency and stop supervision.

External pretrained components remain allowed when they improve the route and
their provenance, license, runtime presence, and contribution boundary are
reported. “Independent ILM” means a self-contained visual deployment path, not
that every foundation must be reimplemented from scratch.

