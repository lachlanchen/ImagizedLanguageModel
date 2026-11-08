# Separating readable writing from correct visual answers

The full-page QA training experiment remains unchanged. This CPU audit and
review preparation strengthen its interpretation; they do not change its
registered data, optimizer, loss, budget, final checkpoint or criteria.

## Reference recognition limits

The [complete evaluation audit](evidence/native_qa_evaluation_readiness.json)
binds all 64 canonical target images to their existing independent OCR
observations and all original accepted answers. Strict canonical matching
and source-accepted matching give the same counts:

| Clean-reference observation | Seen training partition | Held-out partition |
| --- | ---: | ---: |
| Exact answers | 23/32 | 26/32 |
| Both answers correct within a passage pair | 9/16 | 11/16 |

The prospective seen-fit OCR requirement is 24/32. Exact reproduction of
these canonical PNGs would therefore miss that particular requirement even
though it reproduced the targets. This is not an upper bound for other
generated typography, which the recognizer may read better or worse. Keep
this limitation visible rather than silently correcting OCR transcripts or
changing a threshold after seeing model answers. No automatic score alone
establishes reliable writing. All raw outputs still require inspection, and
recognition errors must remain distinct from model errors.

## What the paired control can identify

Each held-out passage has two questions with disjoint normalized accepted-answer
sets. Question erasure gives identical image bytes and identical noise seeds
within every pair. For a deterministic sampler, both erased requests must
therefore give the same answer image. A complete transcript cannot belong to
both disjoint answer sets: at most one question per pair can be exact, and
no pair can have both answers exact. The structural upper bounds are 16/32
answers and zero both-correct pairs. These are consequences of the fixed
control construction, not new neural measurements. The actual final and
initial native control tensors still must be compared by the assessor.

Normal both-correct pairs supply a stronger check than isolated answer counts.
Erasure changes layout as already disclosed in the protocol, so a score drop
alone does not isolate a unique semantic mechanism. Passage erasure tests
grounding in the supplied passage; an answer from external pretrained
knowledge is a separate possibility. This narrow experiment cannot establish
general language modelling or the wider ILM goal.

## Raw transcription before looking at targets

`scripts/prepare_blind_image_review.py` copies complete actual model PNGs
verbatim, authenticates them against retained neural tensors and shuffles them
under opaque review IDs. Contact sheets add labels outside answer images;
every answer-region pixel is unchanged. Source case IDs, model/condition names
and mappings stay in a separate private key. Expected answers and training
manifests are not loaded by the packager.

The [executed packaging check](evidence/native_qa_blind_review_packaging.json)
uses all eight actual earlier full-page preflight outputs. It is an artifact
check, not a new blind quality evaluation: those earlier outputs were already
inspected. The packager retains identical controls as separate cases and
never repairs, crops, replaces or deduplicates generated writing. Reviewers
should transcribe visible marks and mark uncertainty before comparing with
source answers. The preparer still has source access; opaque IDs do not prove
independent or perfect blinding.

A completion worker can wait for the exact live QA coordinator, then prepare
all 320 new answer images and the 64 reused initial copying images only after
its complete assessment. It does not launch or retry a model, score visual
annotations, publish results or declare a successful language learner.

Reproduce the CPU audit from the repository root:

```bash
PYTHONPATH=. python scripts/audit_native_qa_evaluation.py \
  --run artifacts/native_paired_qa_learning_v2 \
  --output artifacts/native_qa_evaluation_readiness_recheck.json
```

The review packager runs in the established model Python with CUDA hidden;
its `--index unique-label=/path/to/outputs.json` option accepts each complete
input-only inference index. Start the completion worker only with the exact
currently live coordinator PID. Its private runtime records that process's
start identity and owns only its CPU packaging child. No model or manuscript
source is changed by these helpers.
