# Return the next experiment to unseen-language learning

The completed 32-question experiment fits all its training answers exactly and
reproduces five cold-loaded streams, but its 24 normal development answers
remain unusable.
This supports the physical input/output implementation and training capacity.
It does not establish a successful language-learning recipe. The 7.54B source
body also supplies inherited language/vision knowledge; this is not a small
foundation learned from glyphs from scratch. More unchanged fitting of the
same questions and further manuscript expansion are not the next experiment.

The next hypothesis is that a varied conditional writing curriculum can teach
reusable glyph generation and answer selection through the now-working path.
More data is not a demonstrated repair: earlier larger-corpus image-code
writers already failed. Preserve the current normalized pixel architecture
and complete-image objective so that this test does not simultaneously change
the representation, decoder family and loss. Do not infer a semantic mechanism
from lower teacher-forced image loss.

The [4,000-update interim assessment](pixel_language_diversity_interim04000_result.md)
of the broader run now retains all 112 autonomous streams. Three of 64 assessed
training answers are pixel-exact; no normal development answer is pixel-exact.
Actual inspection of all 24 normal development outputs identifies no usable
correct complete answer. Two date-like outputs are readable enough to check,
but both give years different from their pictured passages. Whole-training
correct-history image NLL improves to 0.251025, while useful unseen language
generation remains unestablished. The fixed 16,000-update run continues;
this interim observation neither selects a checkpoint nor rejects all possible
longer or different learning recipes.

Prepare a fixed Chinese cohort of 512 natural questions in 256 passage pairs.
Keep the original 32 questions as visible positive controls. Add 240 different
training articles with two distinct-answer questions each, selected by a fixed
hash order before any new model call. Exclude the original 16 articles from
the added group. Select eligible questions by source order and distinct
complete reference pixels, never by generated quality or helper performance.
Retain exact existing prompt/answer images, including original source noise,
for this first diversity test. Cleaning, new fonts and new prompt templates
would be additional interventions and need separate accounting.

The preparation must independently verify all original 32 inputs and targets
against the completed run, deduplicate complete prompt pixels, bind every
source payload and save complete PNG inputs/targets without an OCR or lexical
interface. Preserve the same 24 normal/erased development cases and every
reference alternative for a comparable reused-development assessment. Keep
development paths separate from training records. This is not a sealed test.
No source strings, character inventory or answer IDs may enter deployed
generation.

The CPU preparation is implemented by
`scripts/prepare_pixel_language_diversity.py`. It does not train, load model
weights or qualify generalization. Inspect the actual selected prompt/answer
lengths before registering a GPU schedule. The old four-case length preflight
does not cover potentially longer new answers. A new actual variable-length
batch and memory qualification is necessary before sustained optimization.
Use one GPU0 owner, existing RAM/swap admission and the 22 GiB allocation cap.
Do not restart a live job on an observation timeout.

The training decision must retain complete autonomous outputs, actual image
feedback, learned stopping, original initial-state identity and matched
whole-cohort loss measurements. Report exposure per question and active
compute; a larger update budget is not a data-only causal comparison. New
prompt answer quality is the primary criterion. Exact training reproduction
is only a capacity control. English remains untrained transfer in this Chinese
stage, and historical forms remain outside its capability claims.

If varied training still fails to generate readable, correct unseen answers,
do not keep extending the same fit or add speculative physics terminology.
Examine the learned coupling of native context, visual history and glyph
emission with a controlled alternative, potentially using an offline teacher
that is absent from inference. Useful behavior must precede compression into
a smaller independent model or claims about LLM-equivalent efficiency.

CPU preparation has now completed at `artifacts/pixel_language_diversity_v1/`.
It contains 512 questions, 256 passage pairs and 1,657 complete target blocks,
with 498 unique complete answer images. Answers span one to seven blocks;
the largest prepared prompt has 393,216 pixels. All 32 original positive
controls preserve their exact input/target pixels. An independent NumPy
unpacking of all 512 original source bitstreams reproduces every complete
target, and all prepared PNG hashes, binary shapes, prompt uniqueness and
pairwise target distinctions pass. Prompt folding is not independently
reimplemented in this audit. The first two added passage pairs are actually
viewed, not every prompt in the corpus.

The [manifest](evidence/pixel_language_diversity_preparation/manifest.json),
[bitstream audit](evidence/pixel_language_diversity_preparation/audit.json)
and [sample review](evidence/pixel_language_diversity_preparation/review.json)
retain this preparation. It has zero model calls and zero optimizer updates.
No improved generalization is claimed by preparation.

The [registered learning protocol](pixel_language_diversity_training_protocol.md)
now specifies 16,000 batch-four updates, bounded 32-question native-context
windows, fixed autonomous assessments and complete answer-image supervision.
Actual GPU qualification has completed from the original initial state. The
1/3/4/7-block batch matches independent owning states with maximum normwise
error 3.135e-6; its objective is 10.195171870 versus independent mean
10.195172463. Future-image and padded-history exclusions are exact. Both the
largest-prompt and longest-answer stress batches give finite, nonzero required
gradient groups. Restoring the first update and executing the second again
reproduces model, Adam, sampler, RNG, losses and gradient measurements bit for
bit. This uses three actual optimizer calls and retains none for the main run.
Peak allocated GPU memory is 17.8857 GiB. The
[qualification record](evidence/pixel_language_diversity_qualification/report.json)
and [actual call ledger](evidence/pixel_language_diversity_qualification/optimizer_calls.jsonl)
retain these implementation checks. They do not measure language quality.

The main run starts freshly from the declared initial state and must replay
the qualification's first two updates exactly, after initial whole-cohort loss
measurement and autonomous image generation. Its final unseen-answer results
remain pending; the interim assessment is linked above. The first execution
stopped before its initial assessment or any
training update because state fingerprinting rejected a scalar buffer. The
corrected fingerprint now matches an independent NumPy byte hash of all 286
actual initial tensors. Preserve that failed execution and use a new prepared
directory with newly qualified source. Run preparation and the two GPU stages
separately:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. python scripts/train_pixel_language_diversity.py \
  --stage prepare --output artifacts/pixel_language_diversity_training_v2
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. \
  python scripts/train_pixel_language_diversity.py --stage qualify \
  --output artifacts/pixel_language_diversity_training_v2
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. \
  python scripts/train_pixel_language_diversity.py --stage run \
  --output artifacts/pixel_language_diversity_training_v2
```

Preparation requires a new directory and committed source. A completed
qualification cannot be rerun in its existing directory. Only explicitly
resume an interrupted main execution after its owner has exited, using
`--stage run --resume`; observation timeouts never authorize restarting it.
After completion, run the independent CPU audit:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. python scripts/audit_pixel_language_diversity.py \
  --output artifacts/pixel_language_diversity_training_v2
```

The [corrected-runner qualification](evidence/pixel_language_diversity_qualification/corrected_runner/report.json)
also passes every actual GPU check and repeats the original qualification's
first two update measurements exactly. Its peak is again 17.8857 GiB, with
three actual qualification optimizer calls. Both qualification attempts are
retained, totaling six such calls; the first failed main execution has zero.
The main v2 execution starts from the original initial state. These records
establish reproducible execution, not improved unseen-answer quality.

An additional [CPU source-coverage audit](evidence/pixel_language_answer_coverage/report.json)
reproduces every one of the 512 training answer PNGs and all sixteen Chinese
development reference sets by rendering their owning source text with the
authenticated original renderer. It makes 529 rendering calls and zero model
calls. The first 32 answers contain 174 distinct source code points in 265
occurrences; the full 512 contain 1,084 in 3,896 occurrences. Two of the sixteen
Chinese development cases have an entire reference composed of source code
points present in the old answer set. Eight do in the expanded set.

These counts describe current answer supervision, not prior foundation/raster
exposure or a visual glyph inventory. Code-point identity does not imply glyph
identity. Full supervision coverage cannot establish correct selection,
composition or readable generation. Missing coverage also does not remove a
case from the objective of generating unfamiliar forms. Every registered case
remains in the evaluation; no output is inspected or selected by this audit,
and no source text enters model inference. The original 32-question development
outcomes were already known, while the current run's final outputs are pending.
The audit is reproducible with `scripts/audit_pixel_answer_coverage.py` in the
existing Python 3.12 native rendering environment. It does not modify the
running training protocol, prepared pixels or checkpoint selection.

The [complete initial assessment review](evidence/pixel_language_diversity_initial_review/audit.json)
now independently checks all 112 original streams: 64 training examples and
24 normal plus 24 erased development inputs. Every saved PNG matches its
emitted tensor, every complete-image score matches, and all stopping/cap
traces are consistent. Initial exact matches are zero in every group. Chinese
normal/erased foreground F1 is 0.152883/0.095551; English is
0.050973/0.053070. Whole-training correct-history image NLL is 9.749791.
These are pre-learning baselines, not post-training quality measurements.

All 112 [raw output PNGs](evidence/pixel_language_diversity_initial_review/raw_outputs/)
and output-only sheets are retained. The sheets preserve every original pixel
and wrap long strips for display. They contain case labels without reference
answers. The actually viewed source-first Chinese and English sheets show
broken marks; the [review manifest](evidence/pixel_language_diversity_initial_review/retention_and_review.json)
records exactly which images were inspected. No formal semantic accuracy is
assigned. Feedback files are authenticated by their hashes, while CPU encoder
replay remains part of the required final audit after the model process exits.

Use `scripts/review_pixel_language_assessment.py` to review each completed
fixed assessment on CPU. At updates 4,000 and 8,000, it also requires the
owning learned-state receipt from `scripts/capture_pixel_language_milestones.py`.
The lightweight watcher preserves those two states after their assessment
reports appear, independently of quality. It shares the current tmux session,
uses no GPU and exits after both captures or an original-owner failure. The
owning 4,000-update state is now captured and authenticated against its complete
assessment; the 8,000-update assessment and state remain pending. The
interim review does not qualify a useful language model.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. python scripts/review_pixel_language_assessment.py \
  --run artifacts/pixel_language_diversity_training_v2 --update 4000 \
  --output artifacts/pixel_language_diversity_review04000_v1
```
