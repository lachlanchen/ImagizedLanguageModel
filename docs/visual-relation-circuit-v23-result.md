# Visual Relation Circuit V23: Frozen Result

Date: 2026-08-13

## Verdict

V23 is **accepted for the fixed two-pair same/other visual grammar**.

It is the first experiment in this repository to pass a complete
image-prompt-to-image-answer evidence chain. The student receives six
`32x32` raster frames, resolves a relation among the visible labels, operation,
query, and source glyphs, and emits one new `32x32` answer image. On the single
authorized frozen evaluation over 98 previously unseen Chinese identities, it
reaches:

- `0.99829` binary-choice accuracy;
- `0.99609` query-counterfactual switch accuracy;
- `0.99707` operation-counterfactual switch accuracy;
- `0.99606` minimum switch accuracy on held-out operation/label compositions;
- `0.99463` generated-image identity top-1;
- `0.78478` pixel F1; and
- `0.93994` frozen-retina target cosine.

Every preregistered frozen gate passes. Evaluation was performed once, without
checkpoint selection or threshold changes.

![V23 measured frozen result](../publication/ilm-image-native/figures/visual_relation_circuit_v23_result.png)

This is a bounded causal proof, not an open-ended language model. The six frame
roles and the same/other relation algebra are fixed by the architecture. The
model does not yet parse arbitrary sentences, retrieve facts, explain
etymology, continue a book, or generate a page or movie. Its positive result is
more specific: continuous writing images can support learned visual equality,
learned operation semantics, causal binding, unseen-form routing, and readable
image output without a deployed token or Unicode channel.

## Fixed Visual Task

Every prompt has shape `[B,6,1,32,32]` and fixed image order:

```text
[label_1, glyph_1, label_2, glyph_2, operation, query_label]
```

The visible operation is `同` (return the glyph bound to the queried label) or
`异` (return the other bound glyph). The answer has shape
`[B,1,1,32,32]`. Each episode also renders three exact interventions:

1. change only the query-label image;
2. change only the operation image; and
3. swap the two label/glyph pairs.

The first two interventions must switch the answer; the pair swap must leave it
unchanged. Source glyphs use independently varied noncanonical renderings, and
the target uses a separate canonical face. Exact source-pixel copying therefore
does not satisfy the output target.

The 1,024-form bank is split by the fixed salt
`visual-relation-circuit-v23` into 817 train, 109 development, and 98 frozen
identities. The combinations `(异, 天/地)` and `(同, 左/右)` are absent from
training but present during evaluation. Every individual label pair and
operation is observed in training.

## Image-Only Student Boundary

The deployed path contains:

1. the frozen V16 convolutional retina;
2. continuous cosine comparison between query and label retinal states;
3. a learned image-only operation gate;
4. a differentiable weighted route over the two visible source-glyph images;
   and
5. a separately qualified, frozen image-to-image canonicalizer.

No string, token ID, Unicode ID, OCR transcript, character or operation label,
target slot, answer index, discrete visual codebook, glyph lookup, evaluator
score, or external language model enters `forward`. Strings used by the offline
renderer are removed before the model call. The output is a float ink image,
not a class or text decoder result.

For normalized retinal states, the relation circuit computes

\[
c_i=\langle R(q),R(l_i)\rangle,\qquad
m_i=\operatorname{softmax}_i(\tau c_i),
\]

\[
s=\sigma(U_\theta(R(o))),\qquad
w_i=s m_i+(1-s)(1-m_i),
\]

\[
x_r=w_1g_1+w_2g_2,\qquad \hat y=C_\psi(x_r).
\]

The operation reader and positive temperature contain `25,602` trainable
parameters. Candidate, query-blind, and operation-blind arms have identical
parameter names and shapes. The `1,122,081`-parameter canonicalizer is trained
first and then frozen, preventing relation and rendering modules from jointly
hiding an answer-index shortcut.

## Stage A: Canonicalizer

The canonicalizer selected step `1,000` during the only evidentiary run and
completed its fixed 1,200 updates. Validation used 1,024 images derived from
512 development episodes and all 109 unseen development identities.

| Development gate | Measured | Required | Result |
|---|---:|---:|:---:|
| Pixel F1 | `0.75829` | `>0.72` | pass |
| Identity top-1 | `0.99512` | `>0.80` | pass |
| Target cosine | `0.93747` | `>0.85` | pass |
| Raw-source pixel F1 | `0.54097` | diagnostic | - |
| Gain over raw source | `+0.21732` | `>0.12` | pass |
| Source-shuffled pixel F1 | `0.33456` | diagnostic | - |
| Gain over shuffled source | `+0.42374` | `>0.25` | pass |
| Source-shuffled identity top-1 | `0.01172` | diagnostic | - |
| Identity gain over shuffled source | `+0.98340` | `>0.70` | pass |
| Mean ink fraction | `0.16233` | `[0.03,0.50]` | pass |

Training took `214.10` seconds and peaked at `0.70023 GiB` allocated CUDA
memory on one RTX 4090.

## Stage B: Selected Development Candidate

All three arms completed the fixed 600 updates. The relation-aware candidate
selected step `600`; query-blind selected step `400` by its control-only rule;
operation-blind selected step `600`. Candidate validation covered 512 paired
episodes and 2,048 generated variants.

| Candidate development gate | Measured | Required | Result |
|---|---:|---:|:---:|
| Binary choice | `0.99463` | `>0.95` | pass |
| Query switch | `0.99023` | `>0.90` | pass |
| Operation switch | `0.99023` | `>0.90` | pass |
| Held-out minimum switch | `0.98054` | `>0.85` | pass |
| Pair-swap identity consistency | `1.00000` | `>0.99` | pass |
| Pair-swap output L1 | `0.00000` | `<1e-6` | pass |
| Identity top-1 | `0.98730` | `>0.75` | pass |
| Pixel F1 | `0.75476` | `>0.68` | pass |
| Target cosine | `0.93163` | `>0.82` | pass |
| Query-output L1 | `0.19114` | `>0.12` | pass |
| Operation-output L1 | `0.19045` | `>0.12` | pass |
| Query-label visual match | `0.99805` | `>0.98` | pass |
| Operation-gate accuracy | `1.00000` | `>0.98` | pass |
| Same/other gate separation | `1.00000` | `>0.80` | pass |

Candidate training took `117.18` seconds and peaked at `1.40609 GiB`. The
canonicalizer and relation candidate together add `1,147,683` learned
parameters to the frozen retina.

## Fresh Paired Controls

The fresh audit uses 1,024 new development episodes, seed `22260830`, 4,096
prompt variants, and four independent identity-bank views. It is not a replay
of the checkpoint-selection validation.

| Fresh paired metric | Relation-aware | Query-blind | Operation-blind |
|---|---:|---:|---:|
| Binary choice | `0.99902` | `0.50098` | `0.50024` |
| Query switch | `0.99805` | **`0.00000`** | `0.06152` |
| Operation switch | `0.99805` | `0.01563` | **`0.00000`** |
| Identity top-1 | `0.99512` | `0.29468` | `0.29932` |
| Pixel F1 | `0.75630` | `0.54066` | `0.54037` |
| Query-output L1 | `0.19147` | **`0.00000`** | `0.01090` |
| Operation-output L1 | `0.19124` | `0.00171` | **`0.00000`** |
| Pair-swap output L1 | `0.00000` | `0.00000` | `0.00000` |

The exact-zero blind-factor results are structural invariants, not merely low
scores. The candidate gains `+0.99805` query-switch accuracy over query-blind
and `+0.99805` operation-switch accuracy over operation-blind. All fixed paired
margins pass, and the frozen split remains uninstantiated at this stage.

## Opaque Visual Review

After the paired gate passed, the review builder created 48 opaque development
cards: 36 seen-combination and 12 held-out-combination cases. Each card showed
only the six prompt images, two visible source choices, and the generated
answer. It contained no target image, transcription, correctness label, or
frozen identity.

The `codex-visual-review` agent inspected only those pages and committed its
A/B choices before the scoring program opened the sealed key. It scored
`48/48` overall and `12/12` on held-out compositions, exceeding the fixed
requirements of `44/48` and `11/12`. This is a blinded agent visual review, not
a human-subject study or evidence of broad readability.

## Single Frozen Evaluation

Only after development selection, paired controls, and opaque visual review
passed did the frozen evaluator instantiate the 98 frozen identities. It ran
once on 1,024 episodes, 4,096 variants, and a 392-image identity bank.

| Frozen gate | Measured | Required | Result |
|---|---:|---:|:---:|
| Binary choice | `0.99829` | `>0.95` | pass |
| Query switch | `0.99609` | `>0.90` | pass |
| Operation switch | `0.99707` | `>0.90` | pass |
| Held-out query switch | `0.99803` | diagnostic | - |
| Held-out operation switch | `0.99606` | diagnostic | - |
| Held-out minimum switch | `0.99606` | `>0.85` | pass |
| Identity top-1 | `0.99463` | `>0.75` | pass |
| Pixel F1 | `0.78478` | `>0.68` | pass |
| Target cosine | `0.93994` | `>0.82` | pass |
| Query-output L1 | `0.19218` | `>0.12` | pass |
| Operation-output L1 | `0.19181` | `>0.12` | pass |
| Query-label visual match | `0.99951` | `>0.98` | pass |
| Operation-gate accuracy | `1.00000` | `>0.98` | pass |
| Pair-swap identity consistency | `1.00000` | `>0.99` | pass |
| Pair-swap output L1 | `0.00000` | `<1e-6` | pass |

Frozen evaluation took `4.74` seconds and peaked at `0.49819 GiB` allocated
CUDA memory. The top-level receipt records that frozen images were instantiated
for this authorized run; the nested generic metric keeps its pre-gate sentinel
at zero so the unchanged candidate-threshold function can be reused. No model
selection or threshold change occurred during frozen evaluation.

## What V23 Establishes

V23 establishes, for this fixed visual grammar:

- a real raster-prompt to raster-answer interface;
- learned query-to-label equality from continuous visual states;
- learned same/other operation semantics from an operation image;
- causal composition across multiple visible prompt roles;
- generalization to unseen Chinese identities and held-out relation
  compositions;
- readable image generation through an independently selected visual writer;
- exact pair-order invariance and exact blind-factor controls; and
- a complete selection, paired-control, opaque-review, and frozen protocol on
  one RTX 4090.

V23 does **not** establish:

- arbitrary sentence or page understanding;
- free-form image-answer generation;
- historical glyph retrieval or etymology knowledge;
- book continuation, multi-frame answers, 3D writing, or movies;
- autonomous language memory stronger than symbolic baselines;
- Qwen-8B parity; or
- end-to-end efficiency superiority over token language models.

The next proof should remove the fixed six-role syntax. It should consume a
variable-length 2D visual instruction stream, discover multiple visual spans,
compose their roles through causal memory, and emit at least two answer-image
frames while rereading its own first frame. Its controls must separately blind
instruction, evidence, and generated-history fields. Only after that bounded
stream passes should the project increase page resolution, knowledge breadth,
historical-form retrieval, geometric depth, or time/movie output.

## Reproduction Receipt

The evidentiary stages were run sequentially on one RTX 4090. Generated
checkpoints and logs remain ignored by Git.

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/train_visual_canonicalizer_v23.py \
  --pvf-checkpoint artifacts/predictive_visual_field_v16_memory_pilot/checkpoint_step_0002200.pt \
  --out artifacts/visual_canonicalizer_v23_evidence

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/train_visual_relation_circuit_v23.py \
  --pvf-checkpoint artifacts/predictive_visual_field_v16_memory_pilot/checkpoint_step_0002200.pt \
  --canonicalizer-checkpoint artifacts/visual_canonicalizer_v23_evidence/checkpoint_selected_development.pt \
  --route-mode relation_aware \
  --out artifacts/visual_relation_circuit_v23_evidence/relation_aware

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/train_visual_relation_circuit_v23.py \
  --pvf-checkpoint artifacts/predictive_visual_field_v16_memory_pilot/checkpoint_step_0002200.pt \
  --canonicalizer-checkpoint artifacts/visual_canonicalizer_v23_evidence/checkpoint_selected_development.pt \
  --route-mode query_blind \
  --out artifacts/visual_relation_circuit_v23_evidence/query_blind

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/train_visual_relation_circuit_v23.py \
  --pvf-checkpoint artifacts/predictive_visual_field_v16_memory_pilot/checkpoint_step_0002200.pt \
  --canonicalizer-checkpoint artifacts/visual_canonicalizer_v23_evidence/checkpoint_selected_development.pt \
  --route-mode operation_blind \
  --out artifacts/visual_relation_circuit_v23_evidence/operation_blind

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/eval_visual_relation_circuit_development_v23.py \
  --candidate artifacts/visual_relation_circuit_v23_evidence/relation_aware/checkpoint_selected_development.pt \
  --query-blind artifacts/visual_relation_circuit_v23_evidence/query_blind/checkpoint_selected_development.pt \
  --operation-blind artifacts/visual_relation_circuit_v23_evidence/operation_blind/checkpoint_selected_development.pt \
  --pvf-checkpoint artifacts/predictive_visual_field_v16_memory_pilot/checkpoint_step_0002200.pt \
  --canonicalizer-checkpoint artifacts/visual_canonicalizer_v23_evidence/checkpoint_selected_development.pt \
  --out artifacts/visual_relation_circuit_v23_paired_audit

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/prepare_visual_relation_blinded_review_v23.py \
  --candidate artifacts/visual_relation_circuit_v23_evidence/relation_aware/checkpoint_selected_development.pt \
  --paired-audit artifacts/visual_relation_circuit_v23_paired_audit/paired_development_audit.json \
  --pvf-checkpoint artifacts/predictive_visual_field_v16_memory_pilot/checkpoint_step_0002200.pt \
  --canonicalizer-checkpoint artifacts/visual_canonicalizer_v23_evidence/checkpoint_selected_development.pt \
  --out artifacts/visual_relation_circuit_v23_blinded_review

# Fill responses.json only after inspecting the opaque review pages.
PYTHONPATH=. python scripts/score_visual_relation_blinded_review_v23.py \
  --receipt artifacts/visual_relation_circuit_v23_blinded_review/review_receipt.json \
  --answer-key artifacts/visual_relation_circuit_v23_blinded_review/sealed/answer_key.json \
  --responses artifacts/visual_relation_circuit_v23_blinded_review/responses.json \
  --reviewer codex-visual-review \
  --out artifacts/visual_relation_circuit_v23_blinded_review/review_result.json
```

The frozen command is recorded for audit but must **not** be rerun for this
evidence set:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/eval_visual_relation_circuit_frozen_v23.py \
  --candidate artifacts/visual_relation_circuit_v23_evidence/relation_aware/checkpoint_selected_development.pt \
  --paired-audit artifacts/visual_relation_circuit_v23_paired_audit/paired_development_audit.json \
  --review-result artifacts/visual_relation_circuit_v23_blinded_review/review_result.json \
  --pvf-checkpoint artifacts/predictive_visual_field_v16_memory_pilot/checkpoint_step_0002200.pt \
  --canonicalizer-checkpoint artifacts/visual_canonicalizer_v23_evidence/checkpoint_selected_development.pt \
  --out artifacts/visual_relation_circuit_v23_frozen
```

| Artifact | SHA-256 |
|---|---|
| Fixed protocol | `9cc11b78e97c692b6604ed7c0ab2253e2a4c272d2db505ad191a45913ccd2a70` |
| Fixed manifest | `76048753b52735d14c98f0d9e4eb8d751401fe8810e82eaaaebff517f6866c03` |
| Frozen V16 retina | `90791001203640f0de66316cf2e30b3e2c588480fef0e3d9d4f6283ba043ecbe` |
| Selected canonicalizer | `26cf1bab490abe867e7055a679eff6a9e26e81ad78e6cd9694afd3e425c06135` |
| Canonicalizer training log | `81d7fd72ad12d99eebb3bf8bda40b06607de26a653aee22d6de6f8f284c467ad` |
| Selected relation candidate | `69c5cb06a5f02b5bed26b8687042534e9481fec96bea6ab41e2e00df7c23df43` |
| Selected query-blind control | `4041358c12d500a2ea11f1e56c7bb4df8a4de848f635c733dd779e6168d35011` |
| Selected operation-blind control | `5ce09ddbb52279bd709cdc785c2ec361dc43e93887dacb9a87dc7efa7d9e20c0` |
| Relation candidate training log | `4f25b30e4540634e81a9ac3d9fff28cecf946ef3fd208ac15da859b9884455c3` |
| Query-blind training log | `857d6d4cd2f3b35dc6721711cf0a57effd3360426c68521e0acbff50221b654f` |
| Operation-blind training log | `b9715f9f44c60c1a1572a076b39cdeb853c957578c02a7ad1791b4254e666e7f` |
| Fresh paired audit | `8ee56109212b2a5f48ca4ab6b09dabaec4da6fc00db05e26dd262bd2c3dfdf17` |
| Opaque review receipt | `3fcdaa1584022430aac8fc99d709a2498996135327f1cf702dd8814b7ee0ae73` |
| Scored review result | `5f5aa82ff3fd92ebb0deabc9f98759fe3061507003498c979deeb4d4bf5df568` |
| Frozen evaluation result | `bdfe2040473550a3271173652124986cfa50e662fc186a7120f55768132b6814` |
| Frozen sample sheet | `74851b8d1e412eafcafc00425c44cedc232ee007e6fafce2566d209397be818e` |
| Tracked result figure | `c9f8b13fe148632c1ed3985a0f325a42d6a97cdd819fecb7f71cb568969faf61` |

